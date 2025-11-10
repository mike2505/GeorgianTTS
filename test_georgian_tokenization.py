import sys
import warnings
from pathlib import Path

warnings.filterwarnings('ignore', category=UserWarning, module='perth')
warnings.filterwarnings('ignore', message='.*pkg_resources.*')

sys.path.insert(0, str(Path(__file__).parent / "chatterbox" / "src"))

from chatterbox.mtl_tts import ChatterboxMultilingualTTS

print("Loading tokenizer...")
model = ChatterboxMultilingualTTS.from_pretrained(device='cpu')
tokenizer = model.tokenizer

test_texts = [
    "გამარჯობა",
    "მე ვარ ხელოვნური ინტელექტი",
    "კარგად ხართ",
]

print("\nTesting Georgian tokenization:")
print("=" * 60)

for text in test_texts:
    tokens = tokenizer.text_to_tokens(text, language_id='ka')
    token_list = tokens.flatten().tolist()
    
    print(f"\nOriginal: {text}")
    print(f"Tokens shape: {tokens.shape}")
    print(f"Token IDs: {token_list[:20]}")
    print(f"Unique tokens: {len(set(token_list))}")
    print(f"All tokens: {token_list}")
    
    decoded = tokenizer.decode(tokens.flatten())
    print(f"Decoded: {decoded}")

print("\n" + "=" * 60)
print("✓ If you see diverse token IDs (not all 1s), tokenization works!")

