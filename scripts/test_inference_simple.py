import sys
import torch
import torchaudio as ta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "chatterbox" / "src"))

from chatterbox.mtl_tts import ChatterboxMultilingualTTS

checkpoint_path = "checkpoints/best_model.pt"
device = torch.device('cuda:1')

print("Loading model...")
model = ChatterboxMultilingualTTS.from_pretrained(device=device)

print("Loading checkpoint...")
checkpoint = torch.load(checkpoint_path, map_location=device)
model.t3.load_state_dict(checkpoint['model_state_dict'])

print("Generating...")
text = "Hello, this is a test."

wav = model.generate(
    text=text,
    language_id="en",
    exaggeration=0.5,
    cfg_weight=0.5,
    temperature=0.8,
    repetition_penalty=2.0
)

output_path = "test_simple.wav"
ta.save(output_path, wav, model.sr)
print(f"Saved to {output_path}")

