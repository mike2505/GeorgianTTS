# Fresh Ubuntu Installation Guide

Complete setup guide for Georgian TTS on a fresh Ubuntu installation.

## Quick Setup (Automated)

```bash
# Download and run the setup script
wget https://raw.githubusercontent.com/mike2505/GeorgianTTS/main/ubuntu_setup.sh
chmod +x ubuntu_setup.sh
./ubuntu_setup.sh
```

This will install everything automatically!

## Manual Setup (Step by Step)

### 1. Update System

```bash
sudo apt update && sudo apt upgrade -y
```

### 2. Install Python 3.11

```bash
sudo apt install -y python3.11 python3.11-venv python3.11-dev python3-pip
```

### 3. Install Build Tools

```bash
sudo apt install -y build-essential git wget curl ca-certificates gnupg
```

### 4. Install Audio Libraries

```bash
sudo apt install -y \
    ffmpeg \
    libsndfile1 libsndfile1-dev \
    libportaudio2 portaudio19-dev \
    libasound2-dev \
    sox libsox-fmt-all
```

### 5. Install NVIDIA Driver (if not installed)

```bash
# Check if driver is installed
nvidia-smi

# If not, install automatically
sudo ubuntu-drivers autoinstall
sudo reboot
```

### 6. Install CUDA Toolkit 12.2

```bash
# Add NVIDIA repository
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update

# Install CUDA
sudo apt install cuda-toolkit-12-2 -y

# Add to PATH (add to ~/.bashrc)
echo 'export PATH=/usr/local/cuda-12.2/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### 7. Clone Repository

```bash
cd ~
mkdir -p Projects
cd Projects
git clone https://github.com/mike2505/GeorgianTTS.git
cd GeorgianTTS
```

### 8. Create Virtual Environment

```bash
python3.11 -m venv env
source env/bin/activate
pip install --upgrade pip setuptools wheel
```

### 9. Install PyTorch with CUDA

```bash
pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 10. Install Project Dependencies

```bash
pip install -r requirements.txt
cd chatterbox && pip install -e . && cd ..
```

### 11. Verify Installation

```bash
python scripts/test_setup.py
```

## System Requirements

### Minimum:
- **OS**: Ubuntu 20.04+ (22.04 recommended)
- **RAM**: 16GB
- **Storage**: 100GB free
- **GPU**: NVIDIA GPU with 8GB+ VRAM
- **CUDA**: 11.8 or 12.x

### Recommended:
- **OS**: Ubuntu 22.04 LTS
- **RAM**: 32GB+
- **Storage**: 500GB+ SSD
- **GPU**: RTX 3090/4090 or A100
- **CUDA**: 12.2

## Troubleshooting

### Issue: `nvidia-smi` not found

**Solution**: Install NVIDIA driver
```bash
sudo ubuntu-drivers autoinstall
sudo reboot
```

### Issue: CUDA version mismatch

**Solution**: Install correct CUDA version
```bash
# Check PyTorch CUDA version
python -c "import torch; print(torch.version.cuda)"

# Install matching CUDA toolkit
sudo apt install cuda-toolkit-12-1
```

### Issue: `libsndfile.so` not found

**Solution**: Install audio libraries
```bash
sudo apt install libsndfile1 libsndfile1-dev
```

### Issue: `ffmpeg` not found

**Solution**: Install ffmpeg
```bash
sudo apt install ffmpeg
```

### Issue: Python 3.11 not available

**Solution**: Add deadsnakes PPA (Ubuntu 20.04)
```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-dev
```

## Cloud Platform Specific Instructions

### RunPod / Vast.ai

Most GPU clouds come with:
- ✅ NVIDIA Driver pre-installed
- ✅ CUDA Toolkit pre-installed
- ✅ PyTorch environment

Just run:
```bash
git clone https://github.com/mike2505/GeorgianTTS.git
cd GeorgianTTS
pip install -r requirements.txt
cd chatterbox && pip install -e . && cd ..
```

### AWS EC2 (Deep Learning AMI)

Comes with everything pre-installed:
```bash
# Activate conda environment
conda activate pytorch

# Clone and install
git clone https://github.com/mike2505/GeorgianTTS.git
cd GeorgianTTS
pip install -r requirements.txt
cd chatterbox && pip install -e . && cd ..
```

### Google Colab

Not recommended (session limits), but possible:
```bash
!git clone https://github.com/mike2505/GeorgianTTS.git
%cd GeorgianTTS
!pip install -r requirements.txt
!cd chatterbox && pip install -e . && cd ..
```

## Verification Checklist

After installation, verify:

```bash
# Check Python version
python --version  # Should be 3.11+

# Check NVIDIA driver
nvidia-smi

# Check CUDA
nvcc --version

# Check PyTorch CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Check GPU
python -c "import torch; print(torch.cuda.get_device_name(0))"

# Run full test
python scripts/test_setup.py
```

All checks should pass! ✅

## Next Steps

After successful installation:

1. **Download Dataset**:
   ```bash
   # Place Common Voice Georgian in data/raw/
   ```

2. **Start Training**:
   ```bash
   ./run_pipeline.sh
   ```

3. **Monitor Progress**:
   ```bash
   # In separate terminal
   tensorboard --logdir logs/
   ```

For detailed usage, see `START_HERE.md` and `USAGE.md`.

