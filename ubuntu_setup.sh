#!/bin/bash

set -e

echo "========================================="
echo "Georgian TTS - Fresh Ubuntu Setup"
echo "========================================="

echo ""
echo "System: $(lsb_release -d | cut -f2)"
echo "Kernel: $(uname -r)"
echo ""

echo "Step 1: Updating system packages..."
sudo apt update
sudo apt upgrade -y

echo ""
echo "Step 2: Installing Python 3.11 and development tools..."
sudo apt install -y \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    build-essential \
    git \
    wget \
    curl \
    ca-certificates \
    gnupg

echo ""
echo "Step 3: Installing audio processing libraries..."
sudo apt install -y \
    ffmpeg \
    libsndfile1 \
    libsndfile1-dev \
    libportaudio2 \
    libportaudiocpp0 \
    portaudio19-dev \
    libasound2-dev \
    libsamplerate0-dev \
    sox \
    libsox-fmt-all

echo ""
echo "Step 4: Installing multimedia codecs..."
sudo apt install -y \
    ubuntu-restricted-extras \
    libavcodec-extra \
    libavformat-dev \
    libavutil-dev \
    libswscale-dev

echo ""
echo "Step 5: Installing system utilities..."
sudo apt install -y \
    tmux \
    screen \
    htop \
    nvtop \
    vim \
    nano \
    tree \
    zip \
    unzip \
    p7zip-full

echo ""
echo "Step 6: Checking NVIDIA GPU and CUDA..."
if command -v nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA Driver installed"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    
    if command -v nvcc &> /dev/null; then
        echo "✓ CUDA Toolkit installed: $(nvcc --version | grep release | awk '{print $5}')"
    else
        echo "⚠ CUDA Toolkit not found"
        echo ""
        echo "To install CUDA 12.2 (recommended for PyTorch 2.6):"
        echo "Visit: https://developer.nvidia.com/cuda-downloads"
        echo "Or run:"
        echo "  wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb"
        echo "  sudo dpkg -i cuda-keyring_1.1-1_all.deb"
        echo "  sudo apt update"
        echo "  sudo apt install cuda-toolkit-12-2 -y"
    fi
else
    echo "⚠ NVIDIA Driver not found"
    echo ""
    echo "To install NVIDIA Driver (Ubuntu 22.04+):"
    echo "  sudo ubuntu-drivers autoinstall"
    echo "  sudo reboot"
fi

echo ""
echo "Step 7: Setting up Python environment..."
cd ~
mkdir -p Projects
cd Projects

if [ -d "GeorgianTTS" ]; then
    echo "✓ GeorgianTTS directory already exists"
    cd GeorgianTTS
else
    echo "Cloning GeorgianTTS repository..."
    git clone https://github.com/mike2505/GeorgianTTS.git
    cd GeorgianTTS
fi

echo ""
echo "Creating Python virtual environment..."
python3.11 -m venv env

echo ""
echo "Activating virtual environment..."
source env/bin/activate

echo ""
echo "Upgrading pip and setuptools..."
pip install --upgrade pip setuptools wheel

echo ""
echo "Step 8: Installing PyTorch with CUDA support..."
pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo ""
echo "Step 9: Installing project dependencies..."
pip install -r requirements.txt

echo ""
echo "Step 10: Installing Chatterbox TTS..."
cd chatterbox
pip install -e .
cd ..

echo ""
echo "Step 11: Verifying installation..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
    python -c "import torch; print(f'CUDA Version: {torch.version.cuda}')"
    python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
fi

echo ""
echo "Step 12: Running setup test..."
python scripts/test_setup.py

echo ""
echo "========================================="
echo "Setup Complete!"
echo "========================================="
echo ""
echo "Virtual environment is activated."
echo ""
echo "To activate in future sessions:"
echo "  cd ~/Projects/GeorgianTTS"
echo "  source env/bin/activate"
echo ""
echo "Quick start:"
echo "  1. Place your dataset in data/raw/"
echo "  2. Run: ./run_pipeline.sh"
echo "  3. Or: python scripts/train.py --config configs/georgian_finetune.yaml"
echo ""
echo "See START_HERE.md for detailed instructions."
echo "========================================="

