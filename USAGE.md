# Usage Guide - Georgian TTS Fine-tuning

## Prerequisites

- Python 3.11+
- CUDA-capable GPU (recommended: 16GB+ VRAM)
- 50GB+ free disk space
- Georgian language audio dataset

## Installation

```bash
chmod +x scripts/quick_start.sh
./scripts/quick_start.sh
```

Or manually:

```bash
pip install -r requirements.txt
cd chatterbox && pip install -e . && cd ..
```

## Dataset Preparation

### Option 1: Using metadata.csv

Create `data/raw/metadata.csv`:

```csv
audio_path,transcript,speaker_id,duration
data/raw/audio/sample1.wav,გამარჯობა როგორ ხარ?,speaker_001,2.5
data/raw/audio/sample2.wav,კარგად ვარ გმადლობთ,speaker_001,2.8
```

### Option 2: Directory structure

```
data/raw/
├── audio/
│   ├── file1.wav
│   ├── file2.wav
│   └── ...
└── transcripts/
    ├── file1.txt
    ├── file2.txt
    └── ...
```

### Preprocessing

```bash
python scripts/preprocess_data.py \
    --input_dir data/raw \
    --output_dir data/processed \
    --num_workers 4
```

This will:
- Validate audio files
- Normalize text
- Resample audio to 24kHz
- Split into train/val sets (90/10)
- Generate statistics

### Validation

```bash
python scripts/validate_dataset.py \
    --metadata data/processed/metadata.csv \
    --output_dir data/validation
```

Outputs:
- `audio_errors.csv` - Audio issues
- `text_errors.csv` - Text issues
- `dataset_analysis.png` - Visualizations
- `validation_stats.json` - Statistics

## Training

### Basic Training

```bash
python scripts/train.py --config configs/georgian_finetune.yaml
```

### Custom Configuration

Edit `configs/georgian_finetune.yaml`:

```yaml
training:
  num_epochs: 50
  batch_size: 4
  gradient_accumulation_steps: 4
  learning_rate: 5.0e-5
```

### Resume Training

```bash
python scripts/train.py \
    --config configs/georgian_finetune.yaml \
    --resume checkpoints/checkpoint_epoch_10.pt
```

### Monitor Training

With TensorBoard:

```bash
tensorboard --logdir logs/
```

### Training Tips

1. **Batch Size**: Adjust based on GPU memory
   - 24GB VRAM: batch_size=4-8
   - 16GB VRAM: batch_size=2-4
   - Use `gradient_accumulation_steps` for larger effective batch sizes

2. **Learning Rate**: Start with 5e-5
   - Decrease if loss diverges
   - Increase if training is too slow

3. **Epochs**: 
   - Small dataset (<1h): 100+ epochs
   - Medium dataset (1-10h): 50 epochs
   - Large dataset (>10h): 20-30 epochs

4. **Components to Train**:
   ```yaml
   components_to_train:
     t3: true
     s3gen: false
     voice_encoder: false
   ```
   - Only fine-tune T3 for new language
   - Train S3Gen for better audio quality
   - Train voice_encoder for speaker adaptation

## Inference

### Single Text

```bash
python scripts/inference.py \
    --checkpoint checkpoints/best_model.pt \
    --text "გამარჯობა, როგორ ხარ?" \
    --output output.wav \
    --language_id ka
```

### With Voice Cloning

```bash
python scripts/inference.py \
    --checkpoint checkpoints/best_model.pt \
    --text "გამარჯობა, როგორ ხარ?" \
    --audio_prompt reference_voice.wav \
    --output output.wav \
    --exaggeration 0.5 \
    --cfg_weight 0.5
```

### Batch Generation

Create `texts.txt`:
```
გამარჯობა
როგორ ხარ?
კარგად ვარ
```

```bash
python scripts/inference.py \
    --checkpoint checkpoints/best_model.pt \
    --input_file texts.txt \
    --output_dir outputs/batch \
    --language_id ka
```

### Inference Parameters

- `--exaggeration` (0.0-1.0): Emotion intensity
  - 0.3: Calm, neutral
  - 0.5: Natural (default)
  - 0.7+: Expressive, dramatic

- `--cfg_weight` (0.0-1.0): Guidance strength
  - 0.0: No guidance, more varied
  - 0.5: Balanced (default)
  - 1.0: Strong guidance, more consistent

- `--temperature` (0.1-1.5): Sampling randomness
  - 0.5: Conservative
  - 0.8: Natural (default)
  - 1.2: Creative

- `--repetition_penalty` (1.0-3.0): Prevent repetition
  - 1.0: No penalty
  - 2.0: Moderate (default)
  - 3.0: Strong penalty

## Evaluation

### Generate Test Samples

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/best_model.pt \
    --test_csv data/processed/val.csv \
    --output_dir outputs/evaluation \
    --num_samples 50 \
    --language_id ka
```

### Compare Models

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/epoch_10.pt \
    --compare_checkpoint checkpoints/epoch_50.pt \
    --test_csv data/processed/val.csv \
    --output_dir outputs/comparison \
    --num_samples 10
```

## Tokenizer Analysis

Check Georgian character support:

```bash
python scripts/tokenizer_utils.py \
    --tokenizer_path chatterbox/grapheme_mtl_merged_expanded_v1.json
```

## Troubleshooting

### Out of Memory

1. Reduce `batch_size` in config
2. Increase `gradient_accumulation_steps`
3. Use mixed precision (already enabled)
4. Reduce `max_text_len` and `max_speech_len`

### Poor Quality Output

1. Check dataset quality with validation script
2. Increase training epochs
3. Try different `exaggeration` and `cfg_weight` values
4. Ensure sufficient data (>1 hour recommended)

### Training Loss Not Decreasing

1. Check learning rate (try 1e-5 to 1e-4)
2. Verify dataset is correctly preprocessed
3. Check for data loading errors in logs
4. Try unfreezing more components

### Audio Artifacts

1. Check input audio quality
2. Adjust `cfg_weight` (try 0.3-0.7)
3. Try different reference audio for cloning
4. Consider training S3Gen component

## Performance Optimization

### Fast Training

```yaml
training:
  batch_size: 8
  gradient_accumulation_steps: 2
  use_compile: false
  mixed_precision: true

data:
  num_workers: 8
```

### High Quality

```yaml
training:
  batch_size: 2
  gradient_accumulation_steps: 8
  num_epochs: 100
  learning_rate: 2.0e-5

model:
  components_to_train:
    t3: true
    s3gen: true
    voice_encoder: true
```

## Best Practices

1. **Data Quality > Quantity**
   - Clean audio (no noise, clear speech)
   - Accurate transcripts
   - Consistent recording conditions

2. **Start Small**
   - Test with 100 samples first
   - Verify preprocessing works
   - Check one epoch completes

3. **Regular Evaluation**
   - Generate samples every 5 epochs
   - Listen to outputs
   - Compare with previous checkpoints

4. **Checkpoint Management**
   - Keep best model
   - Save every 5-10 epochs
   - Keep last 3 checkpoints

5. **Experiment Tracking**
   - Use meaningful checkpoint names
   - Log hyperparameters
   - Keep notes on what works

## Advanced Usage

### Multi-GPU Training

Coming soon - currently single GPU only.

### Mixed Language Training

To train on Georgian + other languages, merge datasets with appropriate `language_id` labels.

### Fine-tuning on Specific Voices

1. Create single-speaker dataset
2. Train with voice_encoder enabled
3. Use lower learning rate (1e-5)

### Export for Production

Convert to ONNX or TorchScript for deployment:

```python
import torch
model = load_model(checkpoint_path)
scripted = torch.jit.script(model.t3)
scripted.save("t3_georgian.pt")
```

## Support

For issues or questions:
1. Check logs in `logs/` directory
2. Review error messages carefully
3. Validate dataset first
4. Try with smaller batch size

For Chatterbox-specific issues, refer to:
https://github.com/resemble-ai/chatterbox

