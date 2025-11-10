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
        if model[key] is None or key not in checkpoint['net']:
            continue
            
        state_dict = checkpoint['net'][key]
        
        new_state_dict = {}
        for param_key, param_value in state_dict.items():
            if param_key.startswith('module.'):
                new_state_dict[param_key[7:]] = param_value
            else:
                new_state_dict[param_key] = param_value
        
        print(f"Loading {key}...")
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
    tokens.insert(0, 0)
    tokens = torch.LongTensor(tokens).unsqueeze(0).to(device)
    input_lengths = torch.LongTensor([tokens.shape[-1]]).to(device)
    
    with torch.no_grad():
        text_mask = length_to_mask(input_lengths).to(device)
        
        t_en = model['text_encoder'](tokens, input_lengths, text_mask)
        bert_dur = model['bert'](tokens, attention_mask=(~text_mask).int())
        d_en = model['bert_encoder'](bert_dur).transpose(-1, -2)
        
        s_pred = torch.randn((1, 256)).to(device)
        
        d = model['predictor'].text_encoder(d_en, s_pred, input_lengths, text_mask)
        
        x, _ = model['predictor'].lstm(d)
        duration = model['predictor'].duration_proj(x)
        duration = torch.sigmoid(duration).sum(axis=-1)
        pred_dur = torch.round(duration.squeeze()).clamp(min=1)
        
        pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
        c_frame = 0
        for i in range(pred_aln_trg.size(0)):
            pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
            c_frame += int(pred_dur[i].data)
        
        en = d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(device)
        
        F0_pred, N_pred = model['predictor'].F0Ntrain(en, s_pred)
        
        asr = t_en @ pred_aln_trg.unsqueeze(0).to(device)
        
        wav = model['decoder'](asr, F0_pred, N_pred, s_pred).squeeze().cpu().numpy()
    
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
