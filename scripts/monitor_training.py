import argparse
import time
import json
from pathlib import Path
from datetime import datetime
import pandas as pd


def parse_tensorboard_logs(log_dir):
    from tensorboard.backend.event_processing import event_accumulator
    
    log_path = Path(log_dir)
    if not log_path.exists():
        print(f"Log directory {log_dir} does not exist")
        return None
    
    event_files = list(log_path.rglob("events.out.tfevents.*"))
    if not event_files:
        print("No tensorboard event files found")
        return None
    
    latest_file = max(event_files, key=lambda x: x.stat().st_mtime)
    
    ea = event_accumulator.EventAccumulator(str(latest_file))
    ea.Reload()
    
    metrics = {}
    
    for tag in ea.Tags()['scalars']:
        events = ea.Scalars(tag)
        metrics[tag] = [(e.step, e.value) for e in events]
    
    return metrics


def watch_training(checkpoint_dir, log_dir, interval=30):
    print(f"Monitoring training...")
    print(f"Checkpoints: {checkpoint_dir}")
    print(f"Logs: {log_dir}")
    print(f"Refresh interval: {interval}s")
    print("\n" + "="*80)
    
    checkpoint_path = Path(checkpoint_dir)
    log_path = Path(log_dir)
    
    last_checkpoint = None
    
    while True:
        try:
            print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]")
            
            if checkpoint_path.exists():
                checkpoints = list(checkpoint_path.glob("*.pt"))
                if checkpoints:
                    latest_checkpoint = max(checkpoints, key=lambda x: x.stat().st_mtime)
                    
                    if latest_checkpoint != last_checkpoint:
                        last_checkpoint = latest_checkpoint
                        print(f"\n✓ New checkpoint: {latest_checkpoint.name}")
                        print(f"  Size: {latest_checkpoint.stat().st_size / 1024**2:.1f} MB")
                        print(f"  Modified: {datetime.fromtimestamp(latest_checkpoint.stat().st_mtime)}")
                        
                        try:
                            import torch
                            ckpt = torch.load(latest_checkpoint, map_location='cpu')
                            print(f"  Epoch: {ckpt.get('epoch', 'N/A')}")
                            print(f"  Global step: {ckpt.get('global_step', 'N/A')}")
                        except Exception as e:
                            print(f"  Could not read checkpoint: {e}")
                    
                    print(f"\nCheckpoints: {len(checkpoints)} total")
                    for ckpt in sorted(checkpoints, key=lambda x: x.stat().st_mtime)[-3:]:
                        print(f"  - {ckpt.name}")
            
            if log_path.exists():
                log_files = list(log_path.rglob("*.log"))
                if log_files:
                    latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
                    
                    with open(latest_log, 'r') as f:
                        lines = f.readlines()
                        if lines:
                            print(f"\nLatest log entries:")
                            for line in lines[-5:]:
                                print(f"  {line.rstrip()}")
            
            try:
                metrics = parse_tensorboard_logs(log_path)
                if metrics:
                    print(f"\nTraining metrics:")
                    if 'train/loss' in metrics:
                        train_loss = metrics['train/loss'][-1][1] if metrics['train/loss'] else None
                        if train_loss:
                            print(f"  Train loss: {train_loss:.4f}")
                    
                    if 'val/loss' in metrics:
                        val_loss = metrics['val/loss'][-1][1] if metrics['val/loss'] else None
                        if val_loss:
                            print(f"  Val loss: {val_loss:.4f}")
                    
                    if 'train/lr' in metrics:
                        lr = metrics['train/lr'][-1][1] if metrics['train/lr'] else None
                        if lr:
                            print(f"  Learning rate: {lr:.2e}")
            except Exception as e:
                pass
            
            print("\n" + "="*80)
            print(f"Waiting {interval}s... (Ctrl+C to stop)")
            time.sleep(interval)
            
        except KeyboardInterrupt:
            print("\n\nMonitoring stopped.")
            break
        except Exception as e:
            print(f"\nError: {e}")
            time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description='Monitor training progress')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                       help='Checkpoint directory to monitor')
    parser.add_argument('--log_dir', type=str, default='logs',
                       help='Log directory to monitor')
    parser.add_argument('--interval', type=int, default=30,
                       help='Refresh interval in seconds')
    
    args = parser.parse_args()
    
    watch_training(args.checkpoint_dir, args.log_dir, args.interval)


if __name__ == "__main__":
    main()

