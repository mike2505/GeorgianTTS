import os
import sys
import argparse
import torch
import torchaudio as ta
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "chatterbox" / "src"))

from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from chatterbox.models.t3 import T3
from chatterbox.models.t3.modules.t3_config import T3Config


def load_model(checkpoint_path, device):
    print(f"Loading base model...")
    base_model = ChatterboxMultilingualTTS.from_pretrained(device=device)
    
    if checkpoint_path:
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        base_model.t3.load_state_dict(checkpoint['model_state_dict'])
    
    return base_model


def evaluate_model(model, test_csv, output_dir, language_id="ka", num_samples=None, device='cuda'):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading test data from {test_csv}...")
    df = pd.read_csv(test_csv)
    
    if num_samples:
        df = df.head(num_samples)
    
    print(f"Evaluating on {len(df)} samples...")
    
    results = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        audio_path = row['audio_path']
        transcript = row['transcript']
        speaker_id = row.get('speaker_id', 'unknown')
        
        try:
            model.prepare_conditionals(audio_path, exaggeration=0.5)
            
            generated_wav = model.generate(
                text=transcript,
                language_id=language_id,
                exaggeration=0.5,
                cfg_weight=0.5,
                temperature=0.8
            )
            
            output_file = output_path / f"eval_{idx:04d}.wav"
            ta.save(str(output_file), generated_wav, model.sr)
            
            results.append({
                'index': idx,
                'audio_path': audio_path,
                'generated_path': str(output_file),
                'transcript': transcript,
                'speaker_id': speaker_id,
                'success': True,
                'error': None
            })
            
        except Exception as e:
            print(f"Error on sample {idx}: {e}")
            results.append({
                'index': idx,
                'audio_path': audio_path,
                'generated_path': None,
                'transcript': transcript,
                'speaker_id': speaker_id,
                'success': False,
                'error': str(e)
            })
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path / "evaluation_results.csv", index=False)
    
    success_rate = results_df['success'].mean()
    print(f"\nEvaluation Results:")
    print(f"  Total samples: {len(results_df)}")
    print(f"  Successful: {results_df['success'].sum()}")
    print(f"  Failed: {(~results_df['success']).sum()}")
    print(f"  Success rate: {success_rate*100:.2f}%")
    
    stats = {
        'total_samples': len(results_df),
        'successful': int(results_df['success'].sum()),
        'failed': int((~results_df['success']).sum()),
        'success_rate': float(success_rate)
    }
    
    with open(output_path / "evaluation_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nResults saved to {output_path}")


def compare_models(checkpoint1, checkpoint2, test_csv, output_dir, language_id="ka", num_samples=10, device='cuda'):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("Loading models...")
    model1 = load_model(checkpoint1, device)
    model2 = load_model(checkpoint2, device)
    
    df = pd.read_csv(test_csv).head(num_samples)
    
    print(f"Comparing models on {len(df)} samples...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        audio_path = row['audio_path']
        transcript = row['transcript']
        
        try:
            model1.prepare_conditionals(audio_path, exaggeration=0.5)
            wav1 = model1.generate(text=transcript, language_id=language_id)
            output1 = output_path / f"model1_sample_{idx:04d}.wav"
            ta.save(str(output1), wav1, model1.sr)
            
            model2.prepare_conditionals(audio_path, exaggeration=0.5)
            wav2 = model2.generate(text=transcript, language_id=language_id)
            output2 = output_path / f"model2_sample_{idx:04d}.wav"
            ta.save(str(output2), wav2, model2.sr)
            
        except Exception as e:
            print(f"Error on sample {idx}: {e}")
    
    print(f"\nComparison complete! Files saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate fine-tuned Georgian TTS model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--test_csv', type=str, required=True,
                       help='Path to test dataset CSV')
    parser.add_argument('--output_dir', type=str, default='outputs/evaluation',
                       help='Output directory for generated samples')
    parser.add_argument('--num_samples', type=int, default=None,
                       help='Number of samples to evaluate (default: all)')
    parser.add_argument('--language_id', type=str, default='ka',
                       help='Language ID (ka for Georgian)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--compare_checkpoint', type=str, default=None,
                       help='Optional second checkpoint for comparison')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if args.compare_checkpoint:
        compare_models(
            args.checkpoint,
            args.compare_checkpoint,
            args.test_csv,
            args.output_dir,
            language_id=args.language_id,
            num_samples=args.num_samples or 10,
            device=device
        )
    else:
        model = load_model(args.checkpoint, device)
        evaluate_model(
            model,
            args.test_csv,
            args.output_dir,
            language_id=args.language_id,
            num_samples=args.num_samples,
            device=device
        )


if __name__ == "__main__":
    main()

