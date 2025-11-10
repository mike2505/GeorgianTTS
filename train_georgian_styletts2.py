import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "styletts2"))

import warnings
warnings.filterwarnings('ignore')

import yaml
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm
import shutil

from styletts2.meldataset import build_dataloader
from styletts2.models import *
from styletts2.losses import *
from styletts2.utils import *
from styletts2.optimizers import build_optimizer


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def train_georgian_styletts2(config_path):
    config = load_config(config_path)
    
    accelerator = Accelerator()
    device = accelerator.device
    
    if accelerator.is_main_process:
        print(f"Training Georgian TTS with StyleTTS2")
        print(f"Using {accelerator.num_processes} GPUs")
        print(f"Device: {device}")
    
    train_list = Path(config['data_params']['train_data'])
    val_list = Path(config['data_params']['val_data'])
    
    if not train_list.exists():
        print(f"Error: Train data file not found: {train_list}")
        print("Run: python scripts/prepare_styletts2_data.py first!")
        return
    
    train_dataloader = build_dataloader(
        train_list,
        root_path=config['data_params']['root_path'],
        OOD_data=config['data_params']['OOD_data'],
        min_length=config['data_params']['min_length'],
        batch_size=config['batch_size'],
        num_workers=0,
        dataset_config=config,
        device=device
    )
    
    if accelerator.is_main_process:
        print(f"Train batches: {len(train_dataloader)}")
    
    from styletts2.models import build_model
    model = build_model(config['model_params'], device)
    
    optimizer = build_optimizer(
        model.parameters(),
        lr=config['optimizer_params']['lr']
    )
    
    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )
    
    if accelerator.is_main_process:
        print("Starting training...")
        log_dir = Path(config['log_dir'])
        log_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(config['epochs_1st']):
        model.train()
        
        if accelerator.is_main_process:
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch}")
        else:
            progress_bar = train_dataloader
        
        total_loss = 0
        num_batches = 0
        
        for batch in progress_bar:
            optimizer.zero_grad()
            
            try:
                loss = model(batch)
                accelerator.backward(loss)
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                if accelerator.is_main_process:
                    progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
                    
            except Exception as e:
                if accelerator.is_main_process:
                    print(f"Error in batch: {e}")
                continue
        
        if accelerator.is_main_process:
            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            print(f"Epoch {epoch} - Loss: {avg_loss:.4f}")
            
            if (epoch + 1) % config['save_freq'] == 0:
                checkpoint_path = log_dir / f"checkpoint_epoch_{epoch}.pt"
                accelerator.save({
                    'model': accelerator.unwrap_model(model).state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch
                }, checkpoint_path)
                print(f"Saved checkpoint: {checkpoint_path}")
    
    if accelerator.is_main_process:
        print("Training complete!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train Georgian TTS with StyleTTS2')
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    args = parser.parse_args()
    
    train_georgian_styletts2(args.config)

