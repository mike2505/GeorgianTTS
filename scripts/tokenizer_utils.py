import sys
from pathlib import Path
import torch
import json

sys.path.insert(0, str(Path(__file__).parent.parent / "chatterbox" / "src"))

from chatterbox.models.tokenizers import MTLTokenizer


GEORGIAN_CHARS = list('აბგდევზთიკლმნოპჟრსტუფქღყშჩცძწჭხჯჰ')


def analyze_tokenizer(tokenizer_path):
    print(f"Loading tokenizer from {tokenizer_path}...")
    
    with open(tokenizer_path, 'r', encoding='utf-8') as f:
        tokenizer_data = json.load(f)
    
    vocab = tokenizer_data.get('model', {}).get('vocab', {})
    
    print(f"\nTokenizer vocabulary size: {len(vocab)}")
    
    georgian_tokens = {}
    for char in GEORGIAN_CHARS:
        if char in vocab:
            georgian_tokens[char] = vocab[char]
    
    print(f"Georgian characters in vocabulary: {len(georgian_tokens)}/{len(GEORGIAN_CHARS)}")
    
    missing_chars = [char for char in GEORGIAN_CHARS if char not in vocab]
    if missing_chars:
        print(f"Missing Georgian characters: {missing_chars}")
    
    print("\nGeorgian character tokens:")
    for char, token_id in sorted(georgian_tokens.items(), key=lambda x: x[1]):
        print(f"  {char}: {token_id}")
    
    return georgian_tokens


def test_tokenizer_on_georgian(tokenizer, test_texts):
    print("\nTesting tokenizer on Georgian texts:")
    
    for text in test_texts:
        try:
            tokens = tokenizer.text_to_tokens(text, language_id="ka")
            print(f"\nText: {text}")
            print(f"Tokens: {tokens}")
            print(f"Token count: {tokens.shape[-1]}")
        except Exception as e:
            print(f"\nError tokenizing '{text}': {e}")


def create_georgian_vocab_extension():
    georgian_vocab = {
        char: 10000 + idx 
        for idx, char in enumerate(GEORGIAN_CHARS)
    }
    
    print("Georgian vocabulary extension:")
    for char, token_id in georgian_vocab.items():
        print(f"  '{char}': {token_id}")
    
    return georgian_vocab


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze tokenizer for Georgian support')
    parser.add_argument('--tokenizer_path', type=str, 
                       default='chatterbox/grapheme_mtl_merged_expanded_v1.json',
                       help='Path to tokenizer JSON file')
    
    args = parser.parse_args()
    
    analyze_tokenizer(args.tokenizer_path)
    
    test_texts = [
        "გამარჯობა",
        "როგორ ხარ?",
        "კარგად ვარ, გმადლობთ",
        "საქართველო",
    ]
    
    try:
        from chatterbox.models.tokenizers import MTLTokenizer
        tokenizer = MTLTokenizer(args.tokenizer_path)
        test_tokenizer_on_georgian(tokenizer, test_texts)
    except Exception as e:
        print(f"\nCouldn't load tokenizer: {e}")
    
    print("\n" + "="*50)
    create_georgian_vocab_extension()


if __name__ == "__main__":
    main()

