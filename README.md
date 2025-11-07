# Georgian TTS - Chatterbox Fine-tuning

Fine-tuning [Chatterbox TTS](https://github.com/resemble-ai/chatterbox) for Georgian language.

## Project Structure

```
GeorgianTTS/
├── chatterbox/              # Original Chatterbox repository
├── data/                    # Dataset directory
│   ├── raw/                 # Raw audio and transcripts
│   ├── processed/           # Preprocessed data
│   └── metadata.csv         # Dataset metadata
├── checkpoints/             # Model checkpoints
├── logs/                    # Training logs
├── configs/                 # Training configurations
├── scripts/                 # Training and preprocessing scripts
├── outputs/                 # Generated audio samples
└── requirements.txt         # Python dependencies
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
cd chatterbox && pip install -e . && cd ..
```

2. Prepare your dataset:
   - Place audio files in `data/raw/audio/`
   - Place transcripts in `data/raw/transcripts/`

3. Preprocess the data:
```bash
python scripts/preprocess_data.py --input_dir data/raw --output_dir data/processed
```

4. Start training:
```bash
python scripts/train.py --config configs/georgian_finetune.yaml
```

## Dataset Format

Your dataset should contain:
- Audio files: `.wav`, `.mp3`, or `.flac` (16kHz or 24kHz recommended)
- Transcripts: `.txt` files with matching filenames or a single `metadata.csv`

### Metadata CSV format:
```csv
audio_path,transcript,speaker_id,duration
data/raw/audio/file1.wav,"Georgian text here",speaker_001,3.5
data/raw/audio/file2.wav,"More Georgian text",speaker_001,4.2
```

## Training Configuration

Edit `configs/georgian_finetune.yaml` to customize:
- Batch size
- Learning rate
- Number of epochs
- Model components to fine-tune
- Data augmentation settings

## Inference

After training, generate speech with:
```bash
python scripts/inference.py \
    --checkpoint checkpoints/best_model.pt \
    --text "Georgian text to synthesize" \
    --output outputs/sample.wav
```

## Model Architecture

Chatterbox uses:
- **T3**: 0.5B parameter Llama-based text-to-speech token model
- **S3Gen**: Token-to-audio generator with flow matching
- **Voice Encoder**: Speaker embedding extraction

For Georgian fine-tuning, we adapt:
1. Text tokenizer for Georgian characters
2. T3 model for Georgian phonetics
3. Optional: Voice encoder for Georgian speaker characteristics

