import argparse
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import numpy as np
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


GEORGIAN_ALPHABET = set('აბგდევზთიკლმნოპჟრსტუფქღყშჩცძწჭხჯჰ')


def validate_audio_file(audio_path):
    errors = []
    
    if not Path(audio_path).exists():
        errors.append("File does not exist")
        return False, errors, None
    
    try:
        wav, sr = librosa.load(audio_path, sr=None)
        duration = len(wav) / sr
        
        if duration < 0.5:
            errors.append(f"Audio too short: {duration:.2f}s")
        if duration > 20.0:
            errors.append(f"Audio too long: {duration:.2f}s")
        
        if np.abs(wav).max() < 0.01:
            errors.append("Audio too quiet")
        
        if np.abs(wav).max() > 0.99:
            errors.append("Audio may be clipping")
        
        return len(errors) == 0, errors, duration
        
    except Exception as e:
        errors.append(f"Failed to load audio: {str(e)}")
        return False, errors, None


def validate_text(text):
    errors = []
    
    if not text or len(text.strip()) == 0:
        errors.append("Empty text")
        return False, errors
    
    text = text.strip()
    
    georgian_chars = sum(1 for char in text if char in GEORGIAN_ALPHABET)
    total_chars = len([c for c in text if c.isalpha()])
    
    if total_chars == 0:
        errors.append("No alphabetic characters")
        return False, errors
    
    if georgian_chars / total_chars < 0.5:
        errors.append(f"Not enough Georgian characters: {georgian_chars}/{total_chars}")
    
    if len(text) < 5:
        errors.append(f"Text too short: {len(text)} characters")
    
    if len(text) > 500:
        errors.append(f"Text too long: {len(text)} characters")
    
    return len(errors) == 0, errors


def analyze_dataset(metadata_csv, output_dir):
    print(f"Loading dataset from {metadata_csv}...")
    df = pd.read_csv(metadata_csv)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Total samples: {len(df)}")
    
    print("\nValidating dataset...")
    audio_errors = []
    text_errors = []
    valid_samples = []
    durations = []
    text_lengths = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        audio_path = row['audio_path']
        transcript = row['transcript']
        
        audio_valid, audio_errs, duration = validate_audio_file(audio_path)
        text_valid, text_errs = validate_text(transcript)
        
        if not audio_valid:
            audio_errors.append({
                'index': idx,
                'audio_path': audio_path,
                'errors': '; '.join(audio_errs)
            })
        
        if not text_valid:
            text_errors.append({
                'index': idx,
                'transcript': transcript[:100],
                'errors': '; '.join(text_errs)
            })
        
        if audio_valid and text_valid:
            valid_samples.append(idx)
            if duration:
                durations.append(duration)
            text_lengths.append(len(transcript))
    
    print(f"\nValidation Results:")
    print(f"  Valid samples: {len(valid_samples)} ({len(valid_samples)/len(df)*100:.1f}%)")
    print(f"  Audio errors: {len(audio_errors)}")
    print(f"  Text errors: {len(text_errors)}")
    
    if audio_errors:
        audio_errors_df = pd.DataFrame(audio_errors)
        audio_errors_df.to_csv(output_path / "audio_errors.csv", index=False)
        print(f"  Saved audio errors to {output_path / 'audio_errors.csv'}")
    
    if text_errors:
        text_errors_df = pd.DataFrame(text_errors)
        text_errors_df.to_csv(output_path / "text_errors.csv", index=False)
        print(f"  Saved text errors to {output_path / 'text_errors.csv'}")
    
    print("\nDataset Statistics:")
    if durations:
        print(f"  Total duration: {sum(durations)/3600:.2f} hours")
        print(f"  Average duration: {np.mean(durations):.2f}s")
        print(f"  Duration std: {np.std(durations):.2f}s")
        print(f"  Duration range: [{np.min(durations):.2f}s, {np.max(durations):.2f}s]")
    
    if text_lengths:
        print(f"  Average text length: {np.mean(text_lengths):.1f} chars")
        print(f"  Text length range: [{np.min(text_lengths)}, {np.max(text_lengths)}]")
    
    if 'speaker_id' in df.columns:
        speaker_counts = df['speaker_id'].value_counts()
        print(f"  Number of speakers: {len(speaker_counts)}")
        print(f"  Samples per speaker: {speaker_counts.mean():.1f} ± {speaker_counts.std():.1f}")
    
    print("\nGenerating visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    if durations:
        axes[0, 0].hist(durations, bins=50, edgecolor='black')
        axes[0, 0].set_xlabel('Duration (seconds)')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('Audio Duration Distribution')
        axes[0, 0].axvline(np.mean(durations), color='r', linestyle='--', label='Mean')
        axes[0, 0].legend()
    
    if text_lengths:
        axes[0, 1].hist(text_lengths, bins=50, edgecolor='black')
        axes[0, 1].set_xlabel('Text Length (characters)')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('Text Length Distribution')
        axes[0, 1].axvline(np.mean(text_lengths), color='r', linestyle='--', label='Mean')
        axes[0, 1].legend()
    
    if 'speaker_id' in df.columns:
        speaker_counts = df['speaker_id'].value_counts().head(20)
        axes[1, 0].bar(range(len(speaker_counts)), speaker_counts.values)
        axes[1, 0].set_xlabel('Speaker (top 20)')
        axes[1, 0].set_ylabel('Number of Samples')
        axes[1, 0].set_title('Samples per Speaker')
        axes[1, 0].tick_params(axis='x', rotation=45)
    
    all_chars = []
    for text in df['transcript'].dropna():
        all_chars.extend(list(text.lower()))
    
    char_counts = Counter(all_chars)
    georgian_char_counts = {char: count for char, count in char_counts.items() 
                           if char in GEORGIAN_ALPHABET}
    
    if georgian_char_counts:
        top_chars = sorted(georgian_char_counts.items(), key=lambda x: x[1], reverse=True)[:30]
        chars, counts = zip(*top_chars)
        
        axes[1, 1].bar(range(len(chars)), counts)
        axes[1, 1].set_xticks(range(len(chars)))
        axes[1, 1].set_xticklabels(chars, fontsize=12)
        axes[1, 1].set_xlabel('Georgian Characters')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Top 30 Georgian Character Frequencies')
    
    plt.tight_layout()
    plt.savefig(output_path / "dataset_analysis.png", dpi=300, bbox_inches='tight')
    print(f"Saved visualizations to {output_path / 'dataset_analysis.png'}")
    
    stats = {
        'total_samples': len(df),
        'valid_samples': len(valid_samples),
        'audio_errors': len(audio_errors),
        'text_errors': len(text_errors),
        'total_duration_hours': float(sum(durations) / 3600) if durations else 0,
        'avg_duration_sec': float(np.mean(durations)) if durations else 0,
        'avg_text_length': float(np.mean(text_lengths)) if text_lengths else 0,
        'num_speakers': int(df['speaker_id'].nunique()) if 'speaker_id' in df.columns else 0,
    }
    
    import json
    with open(output_path / "validation_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nValidation complete! Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Validate and analyze Georgian TTS dataset')
    parser.add_argument('--metadata', type=str, required=True, 
                       help='Path to metadata CSV file')
    parser.add_argument('--output_dir', type=str, default='data/validation',
                       help='Output directory for validation results')
    
    args = parser.parse_args()
    
    analyze_dataset(args.metadata, args.output_dir)


if __name__ == "__main__":
    main()

