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
from Utils.PLBERT.util import load_plbert


def load_checkpoint(checkpoint_path, config_path, device='cuda'):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    config = Munch(config)
    
    plbert = load_plbert(config.PLBERT_dir)
    
    model = build_model(Munch(config.model_params), None, None, plbert)
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['net'])
    
    model = model.to(device)
    model.eval()
    
    return model, config


def process_georgian_text(text):
    GEORGIAN_TO_ROMAN = {
        'ა': 'a', 'ბ': 'b', 'გ': 'g', 'დ': 'd', 'ე': 'e', 'ვ': 'v', 'ზ': 'z',
        'თ': 'th', 'ი': 'i', 'კ': "kk", 'ლ': 'l', 'მ': 'm', 'ნ': 'n', 'ო': 'o',
        'პ': "pp", 'ჟ': 'zh', 'რ': 'r', 'ს': 's', 'ტ': "tt", 'უ': 'u', 'ფ': 'ph',
        'ქ': "qq", 'ღ': 'gh', 'ყ': 'qh', 'შ': 'sh', 'ჩ': "chh", 'ც': "tss",
        'ძ': 'dz', 'წ': 'tsh', 'ჭ': 'chq', 'ხ': 'kh', 'ჯ': 'j', 'ჰ': 'h',
    }
    result = []
    for char in text:
        result.append(GEORGIAN_TO_ROMAN.get(char, char))
    return ''.join(result).lower()


def generate_speech(model, text, output_path='output.wav', device='cuda'):
    from text_utils import TextCleaner
    
    textclenaer = TextCleaner()
    
    text = process_georgian_text(text)
    print(f"Romanized: {text}")
    
    tokens = textclenaer(text)
    tokens = torch.LongTensor(tokens).unsqueeze(0).to(device)
    
    with torch.no_grad():
        input_lengths = torch.LongTensor([tokens.shape[-1]]).to(device)
        
        s_pred = model.style_encoder(None, tokens, input_lengths)
        
        wav = model.decoder(
            model.predictor.text_encoder(tokens, input_lengths, s_pred)[0],
            s_pred
        ).squeeze().cpu()
    
    torchaudio.save(output_path, wav.unsqueeze(0), 24000)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint')
    parser.add_argument('--config', type=str, required=True, help='Config file')
    parser.add_argument('--text', type=str, required=True, help='Georgian text')
    parser.add_argument('--output', type=str, default='output_georgian.wav', help='Output file')
    parser.add_argument('--device', type=str, default='cuda:1', help='Device')
    
    args = parser.parse_args()
    
    print("Loading model...")
    model, config = load_checkpoint(args.checkpoint, args.config, args.device)
    
    print("Generating speech...")
    generate_speech(model, args.text, args.output, args.device)
    
    print("Done!")

