import os
import sys
import yaml
import argparse
import warnings

warnings.filterwarnings('ignore')

import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

mp.set_start_method('spawn', force=True)

sys.path.insert(0, str(Path(__file__).parent.parent / "chatterbox" / "src"))

from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from chatterbox.models.t3.modules.cond_enc import T3Cond
from dataset import GeorgianTTSDataset, collate_fn


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def train_worker(rank, world_size, config_path):
    setup(rank, world_size)
    torch.cuda.set_device(rank)
    
    config = load_config(config_path)
    device = f'cuda:{rank}'
    
    if rank == 0:
        print("Loading model...")
    
    model = ChatterboxMultilingualTTS.from_pretrained(device=device)
    t3_ddp = DDP(model.t3, device_ids=[rank])
    
    if rank == 0:
        print(f"Model loaded. Training...")
    
    train_dataset = GeorgianTTSDataset(
        config['data']['train_csv'],
        model.tokenizer,
        model.s3gen.tokenizer,
        model.ve
    )
    
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        sampler=train_sampler,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=False
    )
    
    optimizer = torch.optim.AdamW(
        t3_ddp.parameters(),
        lr=config['training']['learning_rate']
    )
    
    if rank == 0:
        progress_bar = tqdm(train_loader, desc="Training")
    else:
        progress_bar = train_loader
    
    t3_ddp.train()
    
    for batch_idx, batch in enumerate(progress_bar):
        optimizer.zero_grad()
        
        text_tokens = batch['text_tokens'].to(device)
        speech_tokens = batch['speech_tokens'].to(device)
        speaker_embs = batch['speaker_embs'].to(device)
        text_lengths = batch['text_lengths'].to(device)
        speech_lengths = batch['speech_lengths'].to(device)
        
        batch_size = text_tokens.shape[0]
        
        t3_cond = T3Cond(
            speaker_emb=speaker_embs,
            cond_prompt_speech_tokens=None,
            emotion_adv=0.5 * torch.ones(batch_size, 1, 1, device=device)
        )
        
        sot = t3_ddp.module.hp.start_text_token
        eot = t3_ddp.module.hp.stop_text_token
        
        text_tokens = F.pad(text_tokens, (1, 0), value=sot)
        text_tokens = F.pad(text_tokens, (0, 1), value=eot)
        text_lengths = text_lengths + 2
        
        try:
            output = t3_ddp(
                text_tokens=text_tokens,
                text_token_lens=text_lengths,
                speech_tokens=speech_tokens[:, :-1],
                speech_token_lens=speech_lengths - 1,
                t3_cond=t3_cond,
                training=True
            )
            
            if isinstance(output, dict):
                speech_logits = output['speech_logits']
            else:
                speech_logits = output.speech_logits
            
            targets = speech_tokens[:, 1:]
            
            loss = F.cross_entropy(
                speech_logits.reshape(-1, speech_logits.size(-1)),
                targets.reshape(-1),
                ignore_index=0
            )
            
            loss.backward()
            optimizer.step()
            
            if rank == 0 and batch_idx % 10 == 0:
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
                
        except Exception as e:
            if rank == 0 and batch_idx % 100 == 0:
                print(f"Skipping batch {batch_idx}: {e}")
            continue
        
        if rank == 0 and batch_idx % 500 == 0 and batch_idx > 0:
            Path('checkpoints').mkdir(exist_ok=True)
            torch.save({
                'model_state_dict': t3_ddp.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, f'checkpoints/checkpoint_{batch_idx}.pt')
    
    cleanup()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--world_size', type=int, default=2)
    args = parser.parse_args()
    
    mp.spawn(train_worker, args=(args.world_size, args.config), nprocs=args.world_size, join=True)


if __name__ == "__main__":
    main()

