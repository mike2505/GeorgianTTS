#!/bin/bash

echo "========================================="
echo "Georgian TTS Fine-tuning - Quick Start"
echo "========================================="

cd "$(dirname "$0")/.."

echo ""
echo "Step 1: Installing dependencies..."
pip install -r requirements.txt
cd chatterbox && pip install -e . && cd ..

echo ""
echo "Step 2: Creating directory structure..."
mkdir -p data/{raw/{audio,transcripts},processed} checkpoints logs outputs

echo ""
echo "========================================="
echo "Setup complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Place your audio files in: data/raw/audio/"
echo "2. Place your transcripts in: data/raw/transcripts/"
echo "   OR create a metadata.csv file"
echo ""
echo "3. Preprocess your data:"
echo "   python scripts/preprocess_data.py --input_dir data/raw --output_dir data/processed"
echo ""
echo "4. Validate your dataset:"
echo "   python scripts/validate_dataset.py --metadata data/processed/metadata.csv --output_dir data/validation"
echo ""
echo "5. Start training:"
echo "   python scripts/train.py --config configs/georgian_finetune.yaml"
echo ""
echo "6. Generate speech:"
echo "   python scripts/inference.py --checkpoint checkpoints/best_model.pt --text 'გამარჯობა!'"
echo ""

