import pandas as pd
import argparse
import random
from pathlib import Path
import re

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
    return ''.join(result)

def process_georgian_text(text):
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    text = georgian_to_roman(text)
    text = text.lower()
    return text


def create_ood_texts(val_csv, output_file, num_samples=1000):
    print(f"Loading validation data from {val_csv}...")
    df = pd.read_csv(val_csv)
    
    print(f"Found {len(df)} validation samples")
    
    if len(df) < num_samples:
        num_samples = len(df)
    
    sampled = df.sample(n=num_samples, random_state=42)
    
    lines = []
    for idx, row in sampled.iterrows():
        transcript = row['transcript']
        try:
            processed_text = process_georgian_text(transcript)
            if len(processed_text.strip()) > 0:
                lines.append(f"{processed_text}|OOD")
        except Exception as e:
            print(f"Error processing: {e}")
            continue
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print(f"Created {len(lines)} OOD text samples")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create OOD texts from validation set')
    parser.add_argument('--val_csv', type=str, required=True, help='Validation CSV file')
    parser.add_argument('--output', type=str, required=True, help='Output OOD text file')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of OOD samples')
    
    args = parser.parse_args()
    create_ood_texts(args.val_csv, args.output, args.num_samples)

