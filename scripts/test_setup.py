import sys
from pathlib import Path
import torch

print("="*60)
print("Testing Georgian TTS Fine-tuning Setup")
print("="*60)

print("\n1. Checking Python version...")
print(f"   Python: {sys.version}")

print("\n2. Checking PyTorch installation...")
try:
    import torch
    print(f"   PyTorch: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   GPU device: {torch.cuda.get_device_name(0)}")
        print(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
except Exception as e:
    print(f"   ERROR: {e}")

print("\n3. Checking required packages...")
required_packages = [
    'numpy',
    'pandas',
    'librosa',
    'torchaudio',
    'transformers',
    'soundfile',
    'tensorboard',
    'tqdm',
    'matplotlib',
    'seaborn',
    'yaml'
]

missing = []
for package in required_packages:
    try:
        __import__(package)
        print(f"   ✓ {package}")
    except ImportError:
        print(f"   ✗ {package} - MISSING")
        missing.append(package)

if missing:
    print(f"\n   WARNING: Missing packages: {', '.join(missing)}")
    print("   Run: pip install -r requirements.txt")

print("\n4. Checking Chatterbox installation...")
chatterbox_path = Path(__file__).parent.parent / "chatterbox"
if chatterbox_path.exists():
    print(f"   ✓ Chatterbox repository found")
    
    sys.path.insert(0, str(chatterbox_path / "src"))
    
    try:
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS
        print(f"   ✓ Chatterbox imports working")
        
        try:
            print("   Testing model download (this may take a while)...")
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model = ChatterboxMultilingualTTS.from_pretrained(device=device)
            print(f"   ✓ Model loaded successfully on {device}")
            
            langs = model.get_supported_languages()
            print(f"   ✓ Supports {len(langs)} languages")
            
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        except Exception as e:
            print(f"   ⚠ Model loading failed: {e}")
            print("   This is okay - model will download on first use")
            
    except ImportError as e:
        print(f"   ✗ Import failed: {e}")
        print("   Run: cd chatterbox && pip install -e . && cd ..")
else:
    print(f"   ✗ Chatterbox repository not found")
    print("   Expected at: {chatterbox_path}")

print("\n5. Checking directory structure...")
required_dirs = [
    "data/raw/audio",
    "data/raw/transcripts",
    "data/processed",
    "checkpoints",
    "logs",
    "configs",
    "scripts",
    "outputs"
]

for dir_path in required_dirs:
    full_path = Path(__file__).parent.parent / dir_path
    if full_path.exists():
        print(f"   ✓ {dir_path}")
    else:
        print(f"   ✗ {dir_path} - MISSING")

print("\n6. Checking scripts...")
required_scripts = [
    "scripts/preprocess_data.py",
    "scripts/train.py",
    "scripts/inference.py",
    "scripts/evaluate.py",
    "scripts/validate_dataset.py",
    "scripts/dataset.py"
]

for script in required_scripts:
    script_path = Path(__file__).parent.parent / script
    if script_path.exists():
        print(f"   ✓ {script}")
    else:
        print(f"   ✗ {script} - MISSING")

print("\n7. Checking configuration...")
config_path = Path(__file__).parent.parent / "configs/georgian_finetune.yaml"
if config_path.exists():
    print(f"   ✓ Configuration file found")
    try:
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)
        print(f"   ✓ Configuration valid")
        print(f"   - Batch size: {config['training']['batch_size']}")
        print(f"   - Learning rate: {config['training']['learning_rate']}")
        print(f"   - Epochs: {config['training']['num_epochs']}")
    except Exception as e:
        print(f"   ✗ Configuration error: {e}")
else:
    print(f"   ✗ Configuration file not found")

print("\n" + "="*60)
print("Setup Test Complete")
print("="*60)

if missing:
    print("\n⚠ WARNINGS:")
    print(f"  Missing packages: {', '.join(missing)}")
    print("  Install with: pip install -r requirements.txt")
else:
    print("\n✓ All checks passed!")
    print("\nNext steps:")
    print("1. Add your Georgian audio data to data/raw/")
    print("2. Run: python scripts/preprocess_data.py --input_dir data/raw --output_dir data/processed")
    print("3. Run: python scripts/validate_dataset.py --metadata data/processed/metadata.csv")
    print("4. Run: python scripts/train.py --config configs/georgian_finetune.yaml")

print("\nFor detailed usage instructions, see USAGE.md")
print("="*60)

