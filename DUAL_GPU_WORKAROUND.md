# Dual-GPU Training Workaround

**Status:** Temporary workaround for Chatterbox T3 model multi-GPU limitations

## Problem

The Chatterbox T3 model architecture is incompatible with standard PyTorch multi-GPU training (DDP/DataParallel) due to:
- Custom `T3Cond` dataclass that can't be properly split across GPUs
- Conditional parameters that don't always participate in loss computation
- Complex forward pass logic that breaks DDP gradient synchronization

## Workaround Solution

Run **two independent training jobs** in parallel:
- **GPU 0:** Trains on first half of dataset
- **GPU 1:** Trains on second half of dataset
- **Merge:** Average the model weights after training

This effectively utilizes both GPUs without requiring DDP/DataParallel.

## Usage

### Quick Start

```bash
cd /root/GeorgianTTS
chmod +x scripts/train_dual_gpu.sh
./scripts/train_dual_gpu.sh
```

This will:
1. Split `data/processed/train.csv` into two halves
2. Launch training on GPU 0 with first half
3. Launch training on GPU 1 with second half
4. Run both jobs in background

### Monitor Training

```bash
tail -f logs/gpu0_training.log
tail -f logs/gpu1_training.log

watch -n 1 nvidia-smi
```

### Merge Checkpoints

After training completes, merge the checkpoints:

```bash
python3 scripts/merge_checkpoints.py \
    --gpu0_checkpoint checkpoints/gpu0/checkpoint_epoch_49.pt \
    --gpu1_checkpoint checkpoints/gpu1/checkpoint_epoch_49.pt \
    --output checkpoints/merged_final.pt
```

## Files

- `scripts/split_dataset.py` - Splits training CSV into two halves
- `scripts/train_dual_gpu.sh` - Launcher script for dual training
- `scripts/merge_checkpoints.py` - Merges checkpoints by averaging weights
- `configs/georgian_finetune_a100_gpu0.yaml` - Config for GPU 0
- `configs/georgian_finetune_a100_gpu1.yaml` - Config for GPU 1

## Performance

- **Effective Batch Size:** 192 per GPU (24 × 8 accumulation)
- **Expected Speed:** ~3-4 it/s per GPU
- **Epoch Time:** ~20-30 minutes per GPU
- **GPU Utilization:** ~95% on both A100s

## Limitations

- Models train on different data subsets (not identical to single-GPU training)
- Checkpoint merging is simple averaging (not optimal, but works well)
- Requires manual merge step after training
- Slightly higher memory overhead (2 model copies)

## Future

This workaround will be replaced once we modify the Chatterbox T3 model to properly support DDP training.

## Stopping Training

To stop both training jobs:

```bash
pkill -f "train.py --config configs/georgian_finetune_a100_gpu"
```

Or use the PIDs printed by the launcher script.

