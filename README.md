# Georgian TTS with StyleTTS2

State-of-the-art Text-to-Speech for Georgian language using StyleTTS2.

## Why StyleTTS2?

- **Best Quality**: Human-level naturalness and prosody
- **Multi-GPU Ready**: Uses Accelerate library (no DDP issues)
- **Proven**: Works on 2-4x A100 GPUs efficiently
- **Georgian Support**: Custom phonetic romanization

## Setup

### 1. Prepare Data

Run the automated setup:

```bash
./setup_styletts2.sh
```

This will:
- Convert your Common Voice data to StyleTTS2 format
- Create train/val/OOD text files
- Apply Georgian romanization

### 2. Download Pretrained Models

StyleTTS2 needs pretrained modules. Download from:
- **LibriTTS checkpoint**: https://huggingface.co/yl4579/StyleTTS2-LibriTTS/tree/main
- Extract to `Models/LibriTTS/`

Or manually download:
```bash
mkdir -p Models/LibriTTS
cd Models/LibriTTS
wget https://huggingface.co/yl4579/StyleTTS2-LibriTTS/resolve/main/epoch_2nd_00100.pth
wget https://huggingface.co/yl4579/StyleTTS2-LibriTTS/resolve/main/epoch_1st_00080.pth
cd ../..
```

### 3. Train

**Single GPU:**
```bash
accelerate launch --mixed_precision=fp16 --num_processes=1 train_georgian_styletts2.py --config configs/styletts2_georgian.yml
```

**Dual GPU (2x A100):**
```bash
accelerate launch --mixed_precision=fp16 --num_processes=2 train_georgian_styletts2.py --config configs/styletts2_georgian.yml
```

## Georgian Romanization

Georgian characters are converted to phonetically accurate romanizations:

- Ejectives: კ→k', პ→p', ტ→t', ქ→q', ჩ→ch', ც→ts'
- Aspirated: თ→th, ფ→ph, ხ→kh
- Affricates: შ→sh, ჟ→zh, წ→tsh, ჭ→chh
- Others: All unique mappings (no collisions)

Example:
- `გამარჯობა` → `gamarjoba`
- `კარგად ხართ` → `k'argad kharth`

## Data Format

StyleTTS2 expects:
```
path/to/audio.wav|romanized_text|speaker_id
```

All conversions are handled automatically by the setup script.

## Performance

**Expected on 2x A100:**
- **Training speed**: ~5-10 it/s
- **1 epoch**: ~20-30 minutes
- **Total (50 epochs)**: ~1-2 days

## Project Structure

```
GeorgianTTS/
├── data/processed/          # Your Common Voice data
├── styletts2/              # StyleTTS2 codebase
├── Data/                   # StyleTTS2 data lists
├── Models/                 # Pretrained models
├── configs/                # Training configs
├── scripts/                # Data preparation
└── logs/                   # Training logs
```

## Original Dataset

Common Voice Georgian: 89k samples, ~150 hours

## Credits

- **StyleTTS2**: https://github.com/yl4579/StyleTTS2
- **Common Voice**: Mozilla Foundation
