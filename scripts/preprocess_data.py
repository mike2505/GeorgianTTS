import os
import argparse
import json
import librosa
import soundfile as sf
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')


GEORGIAN_ALPHABET = set('აბგდევზთიკლმნოპჟრსტუფქღყშჩცძწჭხჯჰ')
TARGET_SR = 24000
MIN_DURATION = 1.0
MAX_DURATION = 15.0


def is_valid_georgian_text(text):
    if not text or len(text.strip()) == 0:
        return False
    
    text = text.strip()
    georgian_chars = sum(1 for char in text if char in GEORGIAN_ALPHABET)
    total_chars = len([c for c in text if c.isalpha()])
    
    if total_chars == 0:
        return False
    
    return georgian_chars / total_chars > 0.5


def normalize_text(text):
    text = text.strip()
    text = ' '.join(text.split())
    
    punc_to_replace = [
        ("...", ", "),
        ("…", ", "),
        (":", ","),
        (" - ", ", "),
        (";", ", "),
        ("—", "-"),
        ("–", "-"),
        (" ,", ","),
        (""", "\""),
        (""", "\""),
        ("'", "'"),
        ("'", "'"),
    ]
    for old_char, new_char in punc_to_replace:
        text = text.replace(old_char, new_char)
    
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    
    sentence_enders = {".", "!", "?"}
    if not any(text.endswith(p) for p in sentence_enders):
        text += "."
    
    return text


def process_audio(audio_path, target_sr=TARGET_SR):
    try:
        wav, sr = librosa.load(audio_path, sr=None)
        
        if sr != target_sr:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
        
        if wav.ndim > 1:
            wav = wav.mean(axis=0)
        
        wav = wav / (np.abs(wav).max() + 1e-8)
        
        duration = len(wav) / target_sr
        
        return wav, target_sr, duration, None
    except Exception as e:
        return None, None, None, str(e)


def process_single_file(args):
    audio_path, transcript, speaker_id, output_dir, sample_id = args
    
    try:
        if not is_valid_georgian_text(transcript):
            return None, f"Invalid Georgian text: {transcript[:50]}"
        
        wav, sr, duration, error = process_audio(audio_path)
        
        if error:
            return None, f"Audio processing error: {error}"
        
        if duration < MIN_DURATION or duration > MAX_DURATION:
            return None, f"Duration out of range: {duration:.2f}s"
        
        normalized_text = normalize_text(transcript)
        
        output_audio_path = output_dir / "audio" / f"{sample_id}.wav"
        output_audio_path.parent.mkdir(parents=True, exist_ok=True)
        
        sf.write(str(output_audio_path), wav, sr)
        
        return {
            'audio_path': str(output_audio_path),
            'transcript': normalized_text,
            'speaker_id': speaker_id,
            'duration': duration,
            'sample_rate': sr,
            'sample_id': sample_id
        }, None
        
    except Exception as e:
        return None, f"Unexpected error: {str(e)}"


def load_metadata_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    
    required_columns = ['audio_path', 'transcript']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    if 'speaker_id' not in df.columns:
        df['speaker_id'] = 'default'
    
    return df


def load_metadata_from_directory(audio_dir, transcript_dir):
    audio_files = {}
    for ext in ['*.wav', '*.mp3', '*.flac']:
        for audio_path in Path(audio_dir).rglob(ext):
            audio_files[audio_path.stem] = audio_path
    
    metadata = []
    
    if transcript_dir and Path(transcript_dir).exists():
        for txt_path in Path(transcript_dir).rglob('*.txt'):
            stem = txt_path.stem
            if stem in audio_files:
                with open(txt_path, 'r', encoding='utf-8') as f:
                    transcript = f.read().strip()
                
                metadata.append({
                    'audio_path': str(audio_files[stem]),
                    'transcript': transcript,
                    'speaker_id': 'default'
                })
    
    if not metadata:
        for stem, audio_path in audio_files.items():
            metadata.append({
                'audio_path': str(audio_path),
                'transcript': stem,
                'speaker_id': 'default'
            })
    
    return pd.DataFrame(metadata)


def preprocess_dataset(input_dir, output_dir, metadata_csv=None, num_workers=4):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("Loading metadata...")
    if metadata_csv and Path(metadata_csv).exists():
        df = load_metadata_from_csv(metadata_csv)
    else:
        audio_dir = input_path / "audio" if (input_path / "audio").exists() else input_path
        transcript_dir = input_path / "transcripts" if (input_path / "transcripts").exists() else None
        df = load_metadata_from_directory(audio_dir, transcript_dir)
    
    print(f"Found {len(df)} samples")
    
    tasks = []
    for idx, row in df.iterrows():
        audio_path = row['audio_path']
        transcript = row['transcript']
        speaker_id = row.get('speaker_id', 'default')
        sample_id = f"{speaker_id}_{idx:06d}"
        
        tasks.append((audio_path, transcript, speaker_id, output_path, sample_id))
    
    print("Processing audio files...")
    processed_data = []
    errors = []
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_single_file, task): task for task in tasks}
        
        for future in tqdm(as_completed(futures), total=len(futures)):
            result, error = future.result()
            if result:
                processed_data.append(result)
            else:
                task = futures[future]
                errors.append({'audio_path': task[0], 'error': error})
    
    print(f"\nProcessed {len(processed_data)} samples successfully")
    print(f"Failed to process {len(errors)} samples")
    
    if processed_data:
        metadata_df = pd.DataFrame(processed_data)
        metadata_output = output_path / "metadata.csv"
        metadata_df.to_csv(metadata_output, index=False)
        print(f"Saved metadata to {metadata_output}")
        
        train_size = int(0.9 * len(metadata_df))
        train_df = metadata_df[:train_size]
        val_df = metadata_df[train_size:]
        
        train_df.to_csv(output_path / "train.csv", index=False)
        val_df.to_csv(output_path / "val.csv", index=False)
        
        print(f"Train samples: {len(train_df)}")
        print(f"Validation samples: {len(val_df)}")
        
        stats = {
            'total_samples': len(metadata_df),
            'train_samples': len(train_df),
            'val_samples': len(val_df),
            'total_duration': float(metadata_df['duration'].sum()),
            'avg_duration': float(metadata_df['duration'].mean()),
            'min_duration': float(metadata_df['duration'].min()),
            'max_duration': float(metadata_df['duration'].max()),
            'num_speakers': int(metadata_df['speaker_id'].nunique()),
            'sample_rate': int(metadata_df['sample_rate'].iloc[0])
        }
        
        with open(output_path / "stats.json", 'w') as f:
            json.dump(stats, f, indent=2)
        
        print("\nDataset Statistics:")
        print(f"  Total duration: {stats['total_duration']/3600:.2f} hours")
        print(f"  Average duration: {stats['avg_duration']:.2f}s")
        print(f"  Duration range: {stats['min_duration']:.2f}s - {stats['max_duration']:.2f}s")
        print(f"  Number of speakers: {stats['num_speakers']}")
    
    if errors:
        errors_df = pd.DataFrame(errors)
        errors_df.to_csv(output_path / "errors.csv", index=False)
        print(f"\nSaved error log to {output_path / 'errors.csv'}")


def main():
    parser = argparse.ArgumentParser(description='Preprocess Georgian TTS dataset')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Input directory containing audio and transcripts')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for processed data')
    parser.add_argument('--metadata_csv', type=str, default=None,
                        help='Optional CSV file with metadata')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of parallel workers')
    
    args = parser.parse_args()
    
    preprocess_dataset(
        args.input_dir,
        args.output_dir,
        args.metadata_csv,
        args.num_workers
    )


if __name__ == "__main__":
    main()

