import torch
import argparse
from pathlib import Path


def merge_checkpoints(gpu0_checkpoint, gpu1_checkpoint, output_path):
    print(f"Loading GPU 0 checkpoint: {gpu0_checkpoint}")
    ckpt0 = torch.load(gpu0_checkpoint, map_location='cpu')
    
    print(f"Loading GPU 1 checkpoint: {gpu1_checkpoint}")
    ckpt1 = torch.load(gpu1_checkpoint, map_location='cpu')
    
    state_dict0 = ckpt0['model_state_dict']
    state_dict1 = ckpt1['model_state_dict']
    
    print(f"\nAveraging parameters...")
    merged_state_dict = {}
    
    for key in state_dict0.keys():
        if key in state_dict1:
            merged_state_dict[key] = (state_dict0[key] + state_dict1[key]) / 2.0
        else:
            print(f"  Warning: {key} not found in GPU1 checkpoint, using GPU0 only")
            merged_state_dict[key] = state_dict0[key]
    
    merged_checkpoint = {
        'epoch': max(ckpt0['epoch'], ckpt1['epoch']),
        'global_step': ckpt0['global_step'] + ckpt1['global_step'],
        'model_state_dict': merged_state_dict,
        'config': ckpt0['config']
    }
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving merged checkpoint to: {output_path}")
    torch.save(merged_checkpoint, output_path)
    
    print(f"\nMerge complete!")
    print(f"  GPU 0 epoch: {ckpt0['epoch']}")
    print(f"  GPU 1 epoch: {ckpt1['epoch']}")
    print(f"  Merged epoch: {merged_checkpoint['epoch']}")
    print(f"  Total steps: {merged_checkpoint['global_step']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Merge checkpoints from dual-GPU training')
    parser.add_argument('--gpu0_checkpoint', type=str, required=True, help='Path to GPU 0 checkpoint')
    parser.add_argument('--gpu1_checkpoint', type=str, required=True, help='Path to GPU 1 checkpoint')
    parser.add_argument('--output', type=str, required=True, help='Output path for merged checkpoint')
    
    args = parser.parse_args()
    merge_checkpoints(args.gpu0_checkpoint, args.gpu1_checkpoint, args.output)

