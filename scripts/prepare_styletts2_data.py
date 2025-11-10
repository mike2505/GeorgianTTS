import pandas as pd
import argparse
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


def prepare_styletts2_data(input_csv, output_file, audio_base_dir):
    print(f"Loading dataset from {input_csv}...")
    df = pd.read_csv(input_csv)
    
    print(f"Found {len(df)} samples")
    
    unique_speakers = df['speaker_id'].unique()
    speaker_to_id = {speaker: idx for idx, speaker in enumerate(unique_speakers)}
    
    print(f"Found {len(unique_speakers)} unique speakers")
    
    lines = []
    skipped = 0
    
    for idx, row in df.iterrows():
        audio_path = row['audio_path']
        transcript = row['transcript']
        speaker_id = row.get('speaker_id', 'default')
        
        try:
            processed_text = process_georgian_text(transcript)
            
            if len(processed_text.strip()) == 0:
                skipped += 1
                continue
            
            abs_audio_path = Path(audio_base_dir) / Path(audio_path).name
            
            speaker_int_id = speaker_to_id[speaker_id]
            
            line = f"{abs_audio_path}|{processed_text}|{speaker_int_id}"
            lines.append(line)
            
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            skipped += 1
            continue
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print(f"\nSaved {len(lines)} samples to {output_path}")
    print(f"Skipped {skipped} samples")
    print(f"\nFormat: filename.wav|transcription|speaker")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare Georgian dataset for StyleTTS2')
    parser.add_argument('--input_csv', type=str, required=True, help='Input CSV file')
    parser.add_argument('--output_file', type=str, required=True, help='Output text file')
    parser.add_argument('--audio_base_dir', type=str, required=True, help='Base directory for audio files')
    
    args = parser.parse_args()
    prepare_styletts2_data(args.input_csv, args.output_file, args.audio_base_dir)

