#!/bin/bash

echo "========================================="
echo "StyleTTS2 Georgian Setup"
echo "========================================="
echo ""

AUDIO_DIR="$(pwd)/data/processed/audio"
TRAIN_CSV="$(pwd)/data/processed/train.csv"
VAL_CSV="$(pwd)/data/processed/val.csv"

echo "Step 1: Preparing training data..."
python3 scripts/prepare_styletts2_data.py \
    --input_csv "$TRAIN_CSV" \
    --output_file Data/train_georgian.txt \
    --audio_base_dir "$AUDIO_DIR"

echo ""
echo "Step 2: Preparing validation data..."
python3 scripts/prepare_styletts2_data.py \
    --input_csv "$VAL_CSV" \
    --output_file Data/val_georgian.txt \
    --audio_base_dir "$AUDIO_DIR"

echo ""
echo "Step 3: Creating OOD texts..."
python3 scripts/create_ood_texts.py \
    --val_csv "$VAL_CSV" \
    --output Data/OOD_texts_georgian.txt \
    --num_samples 1000

echo ""
echo "========================================="
echo "Setup Complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Download pretrained models (see README)"
echo "2. Run: accelerate launch train_georgian_styletts2.py --config configs/styletts2_georgian.yml"
echo ""

