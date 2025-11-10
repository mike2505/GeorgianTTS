import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "styletts2"))

import warnings
warnings.filterwarnings('ignore')

import torch
import torchaudio
import yaml
from munch import Munch
import numpy as np

from models import *
from Utils.PLBERT.util import load_plbert
from text_utils import TextCleaner
from utils import *

GEORGIAN_TO_ROMAN = {
    'ა': 'a', 'ბ': 'b', 'გ': 'g', 'დ': 'd', 'ე': 'e', 'ვ': 'v', 'ზ': 'z',
    'თ': 'th', 'ი': 'i', 'კ': "kk", 'ლ': 'l', 'მ': 'm', 'ნ': 'n', 'ო': 'o',
    'პ': "pp", 'ჟ': 'zh', 'რ': 'r', 'ს': 's', 'ტ': "tt", 'უ': 'u', 'ფ': 'ph',
    'ქ': "qq", 'ღ': 'gh', 'ყ': 'qh', 'შ': 'sh', 'ჩ': "chh", 'ც': "tss",
    'ძ': 'dz', 'წ': 'tsh', 'ჭ': 'chq', 'ხ': 'kh', 'ჯ': 'j', 'ჰ': 'h',
}

def georgian_to_roman(text):
    result = []
    for char in text:
        result.append(GEORGIAN_TO_ROMAN.get(char, char))
    return ''.join(result).lower()


def load_model(checkpoint_path, config_path, device='cuda'):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    config = Munch.fromDict(config)
    model_params = Munch.fromDict(config.model_params)
    
    text_aligner = load_ASR_models(config.ASR_path, config.ASR_config)
    pitch_extractor = load_F0_models(config.F0_path)
    plbert = load_plbert(config.PLBERT_dir)
    
    model = build_model(model_params, text_aligner, pitch_extractor, plbert)
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    for key in model:
        state_dict = checkpoint['net'][key]
        
        new_state_dict = {}
        for param_key, param_value in state_dict.items():
            if param_key.startswith('module.'):
                new_state_dict[param_key[7:]] = param_value
            else:
                new_state_dict[param_key] = param_value
        
        model[key].load_state_dict(new_state_dict, strict=False)
        model[key] = model[key].to(device)
        model[key].eval()
    
    return model, config


def generate(model, text, device='cuda'):
    textclenaer = TextCleaner()
    
    romanized = georgian_to_roman(text)
    print(f"Input: {text}")
    print(f"Romanized: {romanized}")
    
    tokens = textclenaer(romanized)
    tokens = torch.LongTensor(tokens).unsqueeze(0).to(device)
    input_lengths = torch.LongTensor([tokens.shape[-1]]).to(device)
    
    with torch.no_grad():
        mask = length_to_mask(input_lengths).to(device)
        
        ppgs, s2s_pred, s2s_attn = model.text_aligner(
            model.text_encoder(tokens, input_lengths, None)[0],
            mask,
            tokens
        )
        
        s_pred = model.style_encoder(None, tokens, input_lengths)
        
        mel = model.predictor.text_encoder(tokens, input_lengths, s_pred)
        
        wav = model.decoder(mel[0], s_pred).squeeze().cpu().numpy()
    
    return wav


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--text', type=str, default='გამარჯობა, როგორ ხართ? მე ვარ ხელოვნური ინტელექტი და ვსაუბრობ ქართულად. თბილისი არის საქართველოს დედაქალაქი.')
    parser.add_argument('--output', type=str, default='output_georgian.wav')
    parser.add_argument('--device', type=str, default='cuda:1')
    
    args = parser.parse_args()
    
    print("Loading model...")
    model, config = load_model(args.checkpoint, args.config, args.device)
    
    print("Generating speech...")
    wav = generate(model, args.text, args.device)
    
    print(f"Saving to {args.output}...")
    torchaudio.save(args.output, torch.from_numpy(wav).unsqueeze(0), 24000)
    
    print("Done!")
