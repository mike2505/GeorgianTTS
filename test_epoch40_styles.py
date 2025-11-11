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

print("This tests if different random styles produce variation")
print("If all 3 samples sound identical = need Stage 2 training")
print("If they sound different = style encoder is working!")
print()

device = 'cuda:0'
print("Loading...")

with open('configs/styletts2_georgian.yml') as f:
    config = Munch.fromDict(yaml.safe_load(f))

text_aligner = load_ASR_models(config.ASR_path, config.ASR_config)
pitch_extractor = load_F0_models(config.F0_path)
plbert = load_plbert(config.PLBERT_dir)

model_params = Munch.fromDict(config.model_params)
model = build_model(model_params, text_aligner, pitch_extractor, plbert)

checkpoint = torch.load('logs/styletts2/epoch_1st_00040.pth', map_location='cpu')

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

print("Model loaded! Generating 3 samples with different styles...")

text = "გამარჯობა"
romanized = georgian_to_roman(text)
print(f"Text: {text} → {romanized}\n")

textclenaer = TextCleaner()

for style_id in range(3):
    print(f"Sample {style_id + 1}/3...")
    
    tokens = textclenaer(romanized)
    tokens.insert(0, 0)
    tokens = torch.LongTensor(tokens).unsqueeze(0).to(device)
    
    with torch.no_grad():
        input_lengths = torch.LongTensor([tokens.shape[-1]]).to(device)
        text_mask = length_to_mask(input_lengths).to(device)
        
        torch.manual_seed(style_id * 100)
        style = torch.randn(1, 256).to(device) * 2.0
        
        t_en = model['text_encoder'](tokens, input_lengths, text_mask)
        bert_dur = model['bert'](tokens, attention_mask=(~text_mask).int())
        d_en = model['bert_encoder'](bert_dur).transpose(-1, -2)
        
        d = model['predictor'].text_encoder(d_en, style, input_lengths, text_mask)
        x, _ = model['predictor'].lstm(d)
        duration = model['predictor'].duration_proj(x)
        duration = torch.sigmoid(duration).sum(axis=-1)
        pred_dur = torch.round(duration.squeeze()).clamp(min=1)
        
        pred_aln_trg = torch.zeros(input_lengths[0], int(pred_dur.sum().data))
        c_frame = 0
        for i in range(pred_aln_trg.size(0)):
            pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
            c_frame += int(pred_dur[i].data)
        
        en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(device)).transpose(-1, -2)
        
        F0_pred, N_pred = model['predictor'].F0Ntrain(en, style)
        
        asr = (t_en.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(device)).transpose(-1, -2)
        
        wav = model['decoder'](asr, F0_pred, N_pred, style).squeeze().cpu().numpy()
    
    output_file = f'test_epoch40_style{style_id}.wav'
    torchaudio.save(output_file, torch.from_numpy(wav).unsqueeze(0), 24000)
    print(f"  Saved: {output_file}")

print("\nDone! Listen to all 3 samples.")
print("If they sound different, style is working (even if robotic)!")
print("Stage 2 will make them natural and expressive!")

