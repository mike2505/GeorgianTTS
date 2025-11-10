import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "chatterbox" / "src"))

from chatterbox.models.tokenizers.tokenizer import georgian_to_roman

test_texts = [
    "გამარჯობა",
    "მე ვარ ხელოვნური ინტელექტი",
    "კარგად ხართ",
    "თბილისი",
    "პატარა ქალაქი",
]

print("Testing Georgian to Roman conversion:")
print("=" * 70)

for text in test_texts:
    romanized = georgian_to_roman(text)
    print(f"Georgian:   {text}")
    print(f"Romanized:  {romanized}")
    print(f"Unique:     {len(set(romanized.replace(' ', '').replace(',', '').replace('.', '')))}")
    print()

print("=" * 70)
print("✓ Check if romanizations are unique and phonetically distinct!")
print("✓ No collisions means each Georgian sound has unique representation!")

