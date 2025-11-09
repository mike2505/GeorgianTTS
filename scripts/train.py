import os
import sys
import yaml
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime

mp.set_start_method('spawn', force=True)

sys.path.insert(0, str(Path(__file__).parent.parent / "chatterbox" / "src"))

from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from chatterbox.models.t3 import T3
from chatterbox.models.t3.modules.t3_config import T3Config
from chatterbox.models.t3.modules.cond_enc import T3Cond
from chatterbox.models.s3gen import S3Gen
from chatterbox.models.voice_encoder import VoiceEncoder
from chatterbox.models.tokenizers import MTLTokenizer

from dataset import create_dataloaders


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_model(config, device):
    print("Loading pretrained Chatterbox model...")
    
    multilingual_model = ChatterboxMultilingualTTS.from_pretrained(device=device)
    
    t3 = multilingual_model.t3
    s3gen = multilingual_model.s3gen
    ve = multilingual_model.ve
    tokenizer = multilingual_model.tokenizer
    s3_tokenizer = s3gen.tokenizer
    
    if not config['model']['components_to_train']['t3']:
        for param in t3.parameters():
            param.requires_grad = False
    
    if not config['model']['components_to_train']['s3gen']:
        for param in s3gen.parameters():
            param.requires_grad = False
    
    if not config['model']['components_to_train']['voice_encoder']:
        for param in ve.parameters():
            param.requires_grad = False
    
    trainable_params = sum(p.numel() for p in t3.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in t3.parameters())
    
    print(f"T3 Model: {trainable_params:,} trainable parameters out of {total_params:,}")
    
    return t3, s3gen, ve, tokenizer, s3_tokenizer


def setup_optimizer(model, config):
    opt_config = config['training']['optimizer']
    
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config['training']['learning_rate'],
        betas=opt_config['betas'],
        weight_decay=opt_config['weight_decay'],
        eps=opt_config['eps']
    )
    
    return optimizer


def setup_scheduler(optimizer, config, num_training_steps):
    scheduler_config = config['training']['scheduler']
    warmup_steps = config['training']['warmup_steps']
    
    if scheduler_config['type'] == 'cosine':
        from torch.optim.lr_scheduler import CosineAnnealingLR
        
        def warmup_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            return 1.0
        
        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_lambda)
        
        main_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=num_training_steps - warmup_steps,
            eta_min=scheduler_config['min_lr']
        )
        
        return warmup_scheduler, main_scheduler
    
    return None, None


def compute_t3_loss(t3_model, batch, device):
    text_tokens = batch['text_tokens'].to(device)
    speech_tokens = batch['speech_tokens'].to(device)
    speaker_embs = batch['speaker_embs'].to(device)
    text_masks = batch['text_masks'].to(device)
    
    batch_size = text_tokens.shape[0]
    
    t3_cond = T3Cond(
        speaker_emb=speaker_embs,
        cond_prompt_speech_tokens=None,
        emotion_adv=0.5 * torch.ones(batch_size, 1, 1, device=device)
    )
    
    sot = t3_model.hp.start_text_token
    eot = t3_model.hp.stop_text_token
    
    text_tokens = F.pad(text_tokens, (1, 0), value=sot)
    text_tokens = F.pad(text_tokens, (0, 1), value=eot)
    
    try:
        logits = t3_model.forward(
            text_tokens=text_tokens,
            speech_tokens=speech_tokens[:, :-1],
            t3_cond=t3_cond
        )
        
        targets = speech_tokens[:, 1:]
        
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=0
        )
        
        return loss
        
    except Exception as e:
        print(f"Error in forward pass: {e}")
        return None


def train_epoch(model, train_loader, optimizer, warmup_scheduler, main_scheduler, 
                config, device, epoch, writer, global_step):
    model.train()
    
    total_loss = 0
    num_batches = 0
    grad_accum_steps = config['training']['gradient_accumulation_steps']
    max_grad_norm = config['training']['max_grad_norm']
    log_every = config['logging']['log_every_n_steps']
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(progress_bar):
        try:
            loss = compute_t3_loss(model, batch, device)
            
            if loss is None:
                continue
            
            loss = loss / grad_accum_steps
            loss.backward()
            
            total_loss += loss.item() * grad_accum_steps
            num_batches += 1
            
            if (batch_idx + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                
                if warmup_scheduler and global_step < config['training']['warmup_steps']:
                    warmup_scheduler.step()
                elif main_scheduler:
                    main_scheduler.step()
                
                optimizer.zero_grad()
                global_step += 1
                
                if global_step % log_every == 0:
                    lr = optimizer.param_groups[0]['lr']
                    avg_loss = total_loss / num_batches
                    
                    if writer:
                        writer.add_scalar('train/loss', avg_loss, global_step)
                        writer.add_scalar('train/lr', lr, global_step)
                    
                    progress_bar.set_postfix({
                        'loss': f'{avg_loss:.4f}',
                        'lr': f'{lr:.2e}'
                    })
        
        except Exception as e:
            print(f"Error in batch {batch_idx}: {e}")
            continue
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    return avg_loss, global_step


@torch.no_grad()
def validate(model, val_loader, device):
    model.eval()
    
    total_loss = 0
    num_batches = 0
    
    for batch in tqdm(val_loader, desc="Validation"):
        try:
            loss = compute_t3_loss(model, batch, device)
            
            if loss is None:
                continue
            
            total_loss += loss.item()
            num_batches += 1
            
        except Exception as e:
            print(f"Validation error: {e}")
            continue
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    return avg_loss


def save_checkpoint(model, optimizer, epoch, global_step, config, best=False):
    save_dir = Path(config['checkpointing']['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config
    }
    
    if best:
        save_path = save_dir / "best_model.pt"
    else:
        save_path = save_dir / f"checkpoint_epoch_{epoch}.pt"
    
    torch.save(checkpoint, save_path)
    print(f"Saved checkpoint to {save_path}")
    
    if not best and config['checkpointing']['keep_last_n'] > 0:
        checkpoints = sorted(save_dir.glob("checkpoint_epoch_*.pt"))
        if len(checkpoints) > config['checkpointing']['keep_last_n']:
            for old_ckpt in checkpoints[:-config['checkpointing']['keep_last_n']]:
                old_ckpt.unlink()


def main():
    parser = argparse.ArgumentParser(description='Fine-tune Chatterbox TTS for Georgian')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    if args.resume:
        config['resume_from'] = args.resume
    
    set_seed(config['seed'])
    
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    t3, s3gen, ve, tokenizer, s3_tokenizer = setup_model(config, device)
    
    train_loader, val_loader = create_dataloaders(
        config['data']['train_csv'],
        config['data']['val_csv'],
        tokenizer,
        s3_tokenizer,
        ve,
        batch_size=config['training']['batch_size'],
        num_workers=config['data']['num_workers']
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    optimizer = setup_optimizer(t3, config)
    
    num_training_steps = len(train_loader) * config['training']['num_epochs'] // config['training']['gradient_accumulation_steps']
    warmup_scheduler, main_scheduler = setup_scheduler(optimizer, config, num_training_steps)
    
    writer = None
    if config['logging']['use_tensorboard']:
        log_dir = Path(config['logging']['log_dir']) / datetime.now().strftime('%Y%m%d_%H%M%S')
        writer = SummaryWriter(log_dir)
        print(f"Tensorboard logs: {log_dir}")
    
    start_epoch = 0
    global_step = 0
    best_val_loss = float('inf')
    
    if config.get('resume_from'):
        print(f"Resuming from {config['resume_from']}")
        checkpoint = torch.load(config['resume_from'], map_location=device)
        t3.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        global_step = checkpoint['global_step']
    
    print("\nStarting training...")
    for epoch in range(start_epoch, config['training']['num_epochs']):
        train_loss, global_step = train_epoch(
            t3, train_loader, optimizer, warmup_scheduler, main_scheduler,
            config, device, epoch, writer, global_step
        )
        
        print(f"Epoch {epoch} - Train Loss: {train_loss:.4f}")
        
        if (epoch + 1) % config['logging']['eval_every_n_epochs'] == 0:
            val_loss = validate(t3, val_loader, device)
            print(f"Epoch {epoch} - Val Loss: {val_loss:.4f}")
            
            if writer:
                writer.add_scalar('val/loss', val_loss, epoch)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(t3, optimizer, epoch, global_step, config, best=True)
        
        if (epoch + 1) % config['checkpointing']['save_every_n_epochs'] == 0:
            save_checkpoint(t3, optimizer, epoch, global_step, config, best=False)
    
    print("Training completed!")
    
    if writer:
        writer.close()


if __name__ == "__main__":
    main()

