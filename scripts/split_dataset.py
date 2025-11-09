import pandas as pd
import argparse
from pathlib import Path


def split_dataset(train_csv, output_dir):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading dataset from {train_csv}...")
    df = pd.read_csv(train_csv)
    
    total_samples = len(df)
    half = total_samples // 2
    
    df_gpu0 = df.iloc[:half]
    df_gpu1 = df.iloc[half:]
    
    gpu0_path = output_path / "train_gpu0.csv"
    gpu1_path = output_path / "train_gpu1.csv"
    
    df_gpu0.to_csv(gpu0_path, index=False)
    df_gpu1.to_csv(gpu1_path, index=False)
    
    print(f"Split complete:")
    print(f"  GPU 0: {len(df_gpu0)} samples -> {gpu0_path}")
    print(f"  GPU 1: {len(df_gpu1)} samples -> {gpu1_path}")
    print(f"  Total: {total_samples} samples")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split dataset for dual-GPU training')
    parser.add_argument('--train_csv', type=str, required=True, help='Path to training CSV')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    
    args = parser.parse_args()
    split_dataset(args.train_csv, args.output_dir)

