#!/bin/bash

set -e

echo "========================================="
echo "Georgian TTS Fine-tuning Pipeline"
echo "========================================="

cd "$(dirname "$0")"

COMMONVOICE_DIR="cv-corpus-23.0-2025-09-05/ka"
MAX_SAMPLES=${1:-""}

if [ ! -d "$COMMONVOICE_DIR" ]; then
    echo "Error: Common Voice directory not found: $COMMONVOICE_DIR"
    exit 1
fi

echo ""
echo "Step 1: Converting Common Voice dataset..."
if [ -n "$MAX_SAMPLES" ]; then
    echo "Using first $MAX_SAMPLES training samples for testing"
    python scripts/convert_commonvoice.py \
        --cv_dir "$COMMONVOICE_DIR" \
        --output_dir data/raw \
        --max_samples "$MAX_SAMPLES"
else
    echo "Using full dataset"
    python scripts/convert_commonvoice.py \
        --cv_dir "$COMMONVOICE_DIR" \
        --output_dir data/raw
fi

echo ""
echo "Step 2: Preprocessing data..."
python scripts/preprocess_data.py \
    --metadata_csv data/raw/commonvoice_metadata.csv \
    --input_dir "$COMMONVOICE_DIR" \
    --output_dir data/processed \
    --num_workers 8

echo ""
echo "Step 3: Validating dataset..."
python scripts/validate_dataset.py \
    --metadata data/processed/metadata.csv \
    --output_dir data/validation

echo ""
echo "========================================="
echo "Pipeline Complete!"
echo "========================================="
echo ""
echo "Dataset is ready for training!"
echo ""
echo "Training data: data/processed/train.csv"
echo "Validation data: data/processed/val.csv"
echo "Quality report: data/validation/"
echo ""
echo "Start training with:"
echo "  python scripts/train.py --config configs/georgian_finetune.yaml"
echo ""
echo "Or test the setup first:"
echo "  python scripts/test_setup.py"
echo ""

