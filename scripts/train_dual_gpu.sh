#!/bin/bash

echo "========================================="
echo "Dual-GPU Training Launcher"
echo "========================================="
echo ""

echo "Step 1: Splitting dataset..."
python3 scripts/split_dataset.py \
    --train_csv data/processed/train.csv \
    --output_dir data/processed

echo ""
echo "Step 2: Launching training on both GPUs..."
echo ""

export CUDA_VISIBLE_DEVICES=0
python3 scripts/train.py --config configs/georgian_finetune_a100_gpu0.yaml > logs/gpu0_training.log 2>&1 &
GPU0_PID=$!
echo "GPU 0 training started (PID: $GPU0_PID)"
echo "  Log: logs/gpu0_training.log"

export CUDA_VISIBLE_DEVICES=1
python3 scripts/train.py --config configs/georgian_finetune_a100_gpu1.yaml > logs/gpu1_training.log 2>&1 &
GPU1_PID=$!
echo "GPU 1 training started (PID: $GPU1_PID)"
echo "  Log: logs/gpu1_training.log"

echo ""
echo "========================================="
echo "Both training jobs launched!"
echo "========================================="
echo ""
echo "Monitor progress:"
echo "  GPU 0: tail -f logs/gpu0_training.log"
echo "  GPU 1: tail -f logs/gpu1_training.log"
echo "  GPU usage: watch -n 1 nvidia-smi"
echo ""
echo "Training PIDs: $GPU0_PID (GPU0), $GPU1_PID (GPU1)"
echo ""
echo "To stop training:"
echo "  kill $GPU0_PID $GPU1_PID"
echo ""

wait $GPU0_PID
GPU0_EXIT=$?
echo "GPU 0 training finished (exit code: $GPU0_EXIT)"

wait $GPU1_PID
GPU1_EXIT=$?
echo "GPU 1 training finished (exit code: $GPU1_EXIT)"

echo ""
echo "========================================="
echo "Training Complete!"
echo "========================================="
echo "Checkpoints saved to:"
echo "  GPU 0: checkpoints/gpu0/"
echo "  GPU 1: checkpoints/gpu1/"

