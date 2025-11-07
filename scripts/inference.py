import os
import sys
import argparse
import torch
import torchaudio as ta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "chatterbox" / "src"))

from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from chatterbox.models.t3 import T3
from chatterbox.models.t3.modules.t3_config import T3Config


def load_finetuned_model(checkpoint_path, device):
    print(f"Loading base model...")
    base_model = ChatterboxMultilingualTTS.from_pretrained(device=device)
    
    print(f"Loading fine-tuned checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    base_model.t3.load_state_dict(checkpoint['model_state_dict'])
    
    print("Model loaded successfully!")
    return base_model


def generate_speech(
    model,
    text,
    audio_prompt_path=None,
    output_path="output.wav",
    exaggeration=0.5,
    cfg_weight=0.5,
    temperature=0.8,
    repetition_penalty=2.0,
    language_id="ka"
):
    print(f"Generating speech for: {text}")
    
    if audio_prompt_path:
        print(f"Using audio prompt: {audio_prompt_path}")
        model.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)
    
    wav = model.generate(
        text=text,
        language_id=language_id,
        audio_prompt_path=audio_prompt_path,
        exaggeration=exaggeration,
        cfg_weight=cfg_weight,
        temperature=temperature,
        repetition_penalty=repetition_penalty
    )
    
    ta.save(output_path, wav, model.sr)
    print(f"Saved audio to {output_path}")


def batch_generate(
    model,
    input_file,
    output_dir,
    audio_prompt_path=None,
    exaggeration=0.5,
    cfg_weight=0.5,
    temperature=0.8,
    language_id="ka"
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    print(f"Generating {len(lines)} samples...")
    
    for idx, text in enumerate(lines):
        output_file = output_path / f"sample_{idx:04d}.wav"
        
        try:
            generate_speech(
                model,
                text,
                audio_prompt_path=audio_prompt_path,
                output_path=str(output_file),
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                temperature=temperature,
                language_id=language_id
            )
        except Exception as e:
            print(f"Error generating sample {idx}: {e}")
            continue
    
    print(f"Batch generation complete! Files saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Generate speech with fine-tuned Georgian TTS')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to fine-tuned model checkpoint')
    parser.add_argument('--text', type=str, default=None,
                       help='Text to synthesize')
    parser.add_argument('--input_file', type=str, default=None,
                       help='Text file with multiple lines to synthesize')
    parser.add_argument('--output', type=str, default='output.wav',
                       help='Output audio file path')
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='Output directory for batch generation')
    parser.add_argument('--audio_prompt', type=str, default=None,
                       help='Reference audio file for voice cloning')
    parser.add_argument('--exaggeration', type=float, default=0.5,
                       help='Emotion exaggeration level (0.0-1.0)')
    parser.add_argument('--cfg_weight', type=float, default=0.5,
                       help='Classifier-free guidance weight')
    parser.add_argument('--temperature', type=float, default=0.8,
                       help='Sampling temperature')
    parser.add_argument('--repetition_penalty', type=float, default=2.0,
                       help='Repetition penalty')
    parser.add_argument('--language_id', type=str, default='ka',
                       help='Language ID (ka for Georgian)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = load_finetuned_model(args.checkpoint, device)
    
    if args.input_file:
        batch_generate(
            model,
            args.input_file,
            args.output_dir,
            audio_prompt_path=args.audio_prompt,
            exaggeration=args.exaggeration,
            cfg_weight=args.cfg_weight,
            temperature=args.temperature,
            language_id=args.language_id
        )
    elif args.text:
        generate_speech(
            model,
            args.text,
            audio_prompt_path=args.audio_prompt,
            output_path=args.output,
            exaggeration=args.exaggeration,
            cfg_weight=args.cfg_weight,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
            language_id=args.language_id
        )
    else:
        print("Error: Either --text or --input_file must be provided")
        return


if __name__ == "__main__":
    main()

