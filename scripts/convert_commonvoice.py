import pandas as pd
import argparse
from pathlib import Path
import shutil
from tqdm import tqdm


def convert_commonvoice_dataset(cv_dir, output_dir, splits=['train', 'dev', 'test'], max_samples=None):
    cv_path = Path(cv_dir)
    output_path = Path(output_dir)
    
    if not cv_path.exists():
        raise ValueError(f"Directory {cv_dir} does not exist")
    
    clips_dir = cv_path / "clips"
    if not clips_dir.exists():
        raise ValueError(f"Clips directory {clips_dir} does not exist")
    
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "audio").mkdir(exist_ok=True)
    
    all_data = []
    
    for split in splits:
        tsv_file = cv_path / f"{split}.tsv"
        
        if not tsv_file.exists():
            print(f"Warning: {tsv_file} not found, skipping...")
            continue
        
        print(f"\nProcessing {split}.tsv...")
        df = pd.read_csv(tsv_file, sep='\t')
        
        print(f"  Total samples in {split}: {len(df)}")
        
        if max_samples and split == 'train':
            df = df.head(max_samples)
            print(f"  Limited to {max_samples} samples")
        
        required_cols = ['path', 'sentence']
        for col in required_cols:
            if col not in df.columns:
                print(f"Error: Missing column '{col}' in {tsv_file}")
                continue
        
        valid_count = 0
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"  Converting {split}"):
            audio_filename = row['path']
            sentence = row['sentence']
            
            if pd.isna(sentence) or len(sentence.strip()) == 0:
                continue
            
            source_audio = clips_dir / audio_filename
            
            if not source_audio.exists():
                continue
            
            speaker_id = row.get('client_id', 'unknown')[:16]
            gender = row.get('gender', 'unknown')
            age = row.get('age', 'unknown')
            
            new_filename = f"{split}_{valid_count:06d}.mp3"
            
            all_data.append({
                'audio_path': str(clips_dir / audio_filename),
                'transcript': sentence.strip(),
                'speaker_id': speaker_id,
                'gender': gender,
                'age': age,
                'split': split,
                'original_file': audio_filename
            })
            
            valid_count += 1
        
        print(f"  Valid samples: {valid_count}")
    
    if not all_data:
        print("\nError: No valid data found!")
        return
    
    metadata_df = pd.DataFrame(all_data)
    
    output_csv = output_path / "commonvoice_metadata.csv"
    metadata_df.to_csv(output_csv, index=False)
    
    print(f"\n" + "="*60)
    print(f"Conversion complete!")
    print(f"  Total samples: {len(metadata_df)}")
    print(f"  Metadata saved to: {output_csv}")
    print(f"\nDataset statistics:")
    print(f"  Train samples: {len(metadata_df[metadata_df['split'] == 'train'])}")
    print(f"  Dev samples: {len(metadata_df[metadata_df['split'] == 'dev'])}")
    print(f"  Test samples: {len(metadata_df[metadata_df['split'] == 'test'])}")
    print(f"  Unique speakers: {metadata_df['speaker_id'].nunique()}")
    
    if 'gender' in metadata_df.columns:
        print(f"\nGender distribution:")
        print(metadata_df['gender'].value_counts())
    
    print(f"\n" + "="*60)
    print(f"Next steps:")
    print(f"1. Preprocess the data:")
    print(f"   python scripts/preprocess_data.py \\")
    print(f"       --metadata_csv {output_csv} \\")
    print(f"       --input_dir {cv_dir} \\")
    print(f"       --output_dir data/processed")
    print(f"\n2. Validate the dataset:")
    print(f"   python scripts/validate_dataset.py \\")
    print(f"       --metadata data/processed/metadata.csv \\")
    print(f"       --output_dir data/validation")


def main():
    parser = argparse.ArgumentParser(description='Convert Mozilla Common Voice dataset to training format')
    parser.add_argument('--cv_dir', type=str, required=True,
                       help='Path to Common Voice dataset directory (e.g., cv-corpus-23.0-2025-09-05/ka)')
    parser.add_argument('--output_dir', type=str, default='data/raw',
                       help='Output directory for converted metadata')
    parser.add_argument('--splits', type=str, nargs='+', default=['train', 'dev', 'test'],
                       help='Which splits to include (train, dev, test)')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of training samples to use (for testing)')
    
    args = parser.parse_args()
    
    convert_commonvoice_dataset(
        args.cv_dir,
        args.output_dir,
        args.splits,
        args.max_samples
    )


if __name__ == "__main__":
    main()

