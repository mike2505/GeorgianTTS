import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "styletts2"))

import warnings
warnings.filterwarnings('ignore')

import torch
import torchaudio
import yaml
from munch import Munch

from models import *
from utils import *
from text_utils import TextCleaner
from Utils.PLBERT.util import load_plbert

GEORGIAN_TO_ROMAN = {
    'ა': 'a', 'ბ': 'b', 'გ': 'g', 'დ': 'd', 'ე': 'e', 'ვ': 'v', 'ზ': 'z',
    'თ': 'th', 'ი': 'i', 'კ': "kk", 'ლ': 'l', 'მ': 'm', 'ნ': 'n', 'ო': 'o',
    'პ': "pp", 'ჟ': 'zh', 'რ': 'r', 'ს': 's', 'ტ': "tt", 'უ': 'u', 'ფ': 'ph',
    'ქ': "qq", 'ღ': 'gh', 'ყ': 'qh', 'შ': 'sh', 'ჩ': "chh", 'ც': "tss",
    'ძ': 'dz', 'წ': 'tsh', 'ჭ': 'chq', 'ხ': 'kh', 'ჯ': 'j', 'ჰ': 'h',
}

def georgian_to_roman(text):
    return ''.join([GEORGIAN_TO_ROMAN.get(c, c) for c in text]).lower()

device = 'cuda:1'
config_path = 'configs/styletts2_georgian.yml'
checkpoint_path = 'logs/styletts2/epoch_1st_00005.pth'

print("Loading config...")
with open(config_path) as f:
    config = Munch.fromDict(yaml.safe_load(f))

print("Loading models...")
text_aligner = load_ASR_models(config.ASR_path, config.ASR_config)
pitch_extractor = load_F0_models(config.F0_path)
plbert = load_plbert(config.PLBERT_dir)

model_params = Munch.fromDict(config.model_params)
model = build_model(model_params, text_aligner, pitch_extractor, plbert)

print("Loading checkpoint...")
checkpoint = torch.load(checkpoint_path, map_location='cpu')

for key in model:
    if model[key] is None or key not in checkpoint['net']:
        continue
    state_dict = checkpoint['net'][key]
    new_state_dict = {}
    for k, v in state_dict.items():
        new_state_dict[k[7:] if k.startswith('module.') else k] = v
    model[key].load_state_dict(new_state_dict, strict=False)
    model[key] = model[key].to(device)
    model[key].eval()

print("Model loaded!")

text = "გამარჯობა"
romanized = georgian_to_roman(text)
print(f"Text: {text}")
print(f"Romanized: {romanized}")

textclenaer = TextCleaner()
tokens = textclenaer(romanized)
tokens.insert(0, 0)
tokens = torch.LongTensor(tokens).unsqueeze(0).to(device)

print(f"Tokens: {tokens.shape}")

print("Generating...")
with torch.no_grad():
    asr_features = torch.randn(1, 100, 512).to(device)
    F0_features = torch.randn(1, 100, 1).to(device)
    N_features = torch.randn(1, 100, 1).to(device)
    style = torch.randn(1, 256).to(device)
    
    wav = model['decoder'](asr_features, F0_features, N_features, style).squeeze().cpu().numpy()

print(f"Generated wav shape: {wav.shape}")
torchaudio.save('test_simple.wav', torch.from_numpy(wav).unsqueeze(0), 24000)
print("Saved to test_simple.wav")

