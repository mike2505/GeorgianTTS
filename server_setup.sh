#!/bin/bash

set -e

echo "========================================="
echo "Georgian TTS Server Setup & Training"
echo "========================================="

echo ""
echo "Step 1: Extracting Common Voice dataset..."
tar -xzf "Common Voice Scripted Speech 23.0 - Georgian.tar.gz"

echo ""
echo "Step 2: Installing Python dependencies..."
pip install -r requirements.txt

echo ""
echo "Step 3: Installing Chatterbox TTS..."
cd chatterbox
pip install -e .
cd ..

echo ""
echo "Step 4: Converting Common Voice format..."
python scripts/convert_commonvoice.py \
    --cv_dir cv-corpus-23.0-2025-09-05/ka \
    --output_dir data/raw

echo ""
echo "Step 5: Preprocessing audio data..."
python scripts/preprocess_data.py \
    --metadata_csv data/raw/commonvoice_metadata.csv \
    --input_dir cv-corpus-23.0-2025-09-05/ka \
    --output_dir data/processed \
    --num_workers 8

echo ""
echo "Step 6: Validating dataset..."
python scripts/validate_dataset.py \
    --metadata data/processed/metadata.csv \
    --output_dir data/validation

echo ""
echo "========================================="
echo "Setup Complete! Starting Training..."
echo "========================================="
echo ""

python scripts/train.py --config configs/georgian_finetune.yaml

