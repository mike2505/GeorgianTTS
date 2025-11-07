# Quick Start - Georgian TTS Fine-tuning

## Your Dataset
You have Mozilla Common Voice Georgian v23.0 - Perfect for TTS training!

## Step-by-Step Guide

### 1. Convert the Common Voice Dataset (2 minutes)

```bash
python scripts/convert_commonvoice.py \
    --cv_dir cv-corpus-23.0-2025-09-05/ka \
    --output_dir data/raw
```

**Optional**: Test with fewer samples first
```bash
python scripts/convert_commonvoice.py \
    --cv_dir cv-corpus-23.0-2025-09-05/ka \
    --output_dir data/raw \
    --max_samples 1000
```

### 2. Preprocess the Data (10-30 minutes)

```bash
python scripts/preprocess_data.py \
    --metadata_csv data/raw/commonvoice_metadata.csv \
    --input_dir cv-corpus-23.0-2025-09-05/ka \
    --output_dir data/processed \
    --num_workers 8
```

This will:
- ✓ Validate Georgian text (already supports all 33 Georgian letters)
- ✓ Resample audio to 24kHz
- ✓ Normalize audio levels
- ✓ Create train/validation splits
- ✓ Generate statistics

### 3. Validate Dataset Quality (5 minutes)

```bash
python scripts/validate_dataset.py \
    --metadata data/processed/metadata.csv \
    --output_dir data/validation
```

Check the outputs:
- `data/validation/dataset_analysis.png` - Visualizations
- `data/validation/validation_stats.json` - Statistics
- `data/validation/*_errors.csv` - Any issues found

### 4. Start Training

```bash
python scripts/train.py --config configs/georgian_finetune.yaml
```

Monitor in another terminal:
```bash
python scripts/monitor_training.py
```

Or use TensorBoard:
```bash
tensorboard --logdir logs/
```

### 5. Generate Speech

After training (or even during):

```bash
python scripts/inference.py \
    --checkpoint checkpoints/best_model.pt \
    --text "გამარჯობა, როგორ ხარ?" \
    --output test_output.wav \
    --language_id ka
```

## Important: No Code Changes Needed! 🎉

**Georgian is already fully supported:**

✓ All 33 Georgian letters (ა-ჰ) recognized
✓ Text validation handles Georgian script
✓ Tokenizer supports Georgian characters
✓ Language ID "ka" is supported
✓ Punctuation normalized correctly

## Quick Test

Verify everything is set up:

```bash
python scripts/test_setup.py
```

## Expected Results

**Dataset Size**: ~50-100 hours of Georgian speech
**Training Time**: ~20-25 hours for 30 epochs (on A100)
**Quality**: 
- After 10 epochs: Basic intelligibility
- After 30 epochs: Good quality
- After 50 epochs: Near-native quality

## Common Commands

```bash
# Convert Common Voice data
python scripts/convert_commonvoice.py --cv_dir cv-corpus-23.0-2025-09-05/ka --output_dir data/raw

# Preprocess
python scripts/preprocess_data.py --metadata_csv data/raw/commonvoice_metadata.csv --input_dir cv-corpus-23.0-2025-09-05/ka --output_dir data/processed

# Validate
python scripts/validate_dataset.py --metadata data/processed/metadata.csv --output_dir data/validation

# Train
python scripts/train.py --config configs/georgian_finetune.yaml

# Generate speech
python scripts/inference.py --checkpoint checkpoints/best_model.pt --text "გამარჯობა!" --output output.wav --language_id ka

# Evaluate
python scripts/evaluate.py --checkpoint checkpoints/best_model.pt --test_csv data/processed/val.csv --output_dir outputs/eval
```

## Troubleshooting

See `USAGE.md` for detailed usage instructions
See `GEORGIAN_SETUP.md` for Georgian-specific information

## File Structure After Setup

```
GeorgianTTS/
├── cv-corpus-23.0-2025-09-05/    # Extracted Common Voice
│   └── ka/
│       ├── clips/                 # Audio files
│       └── *.tsv                  # Metadata
├── data/
│   ├── raw/
│   │   └── commonvoice_metadata.csv
│   ├── processed/
│   │   ├── audio/                 # Preprocessed 24kHz WAV files
│   │   ├── train.csv
│   │   ├── val.csv
│   │   └── metadata.csv
│   └── validation/
│       └── dataset_analysis.png
├── checkpoints/                   # Model checkpoints
├── logs/                          # Training logs
└── outputs/                       # Generated audio
```

## Next Steps

1. Run the conversion script
2. Check the validation output
3. Start training
4. Monitor progress
5. Generate test samples
6. Iterate and improve

Happy training! 🚀

