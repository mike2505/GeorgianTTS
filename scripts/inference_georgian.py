import os
import sys
import argparse
import torch
import torchaudio as ta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "chatterbox" / "src"))

from chatterbox.mtl_tts import ChatterboxMultilingualTTS


def load_finetuned_model(checkpoint_path, device):
    print(f"Loading base model...")
    base_model = ChatterboxMultilingualTTS.from_pretrained(device=device)
    
    print(f"Loading fine-tuned checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    base_model.t3.load_state_dict(checkpoint['model_state_dict'])
    
    print("Model loaded successfully!")
    return base_model


def generate_speech_direct(
    model,
    text,
    audio_prompt_path=None,
    output_path="output.wav",
    exaggeration=0.5,
    cfg_weight=3.0,
    temperature=0.7,
    top_k=210,
    top_p=0.9,
    repetition_penalty=1.5,
    max_new_tokens=2048
):
    print(f"Generating speech for: {text}")
    
    if audio_prompt_path:
        print(f"Using audio prompt: {audio_prompt_path}")
        audio_prompt, sr = ta.load(audio_prompt_path)
        if audio_prompt.shape[0] > 1:
            audio_prompt = audio_prompt.mean(dim=0, keepdim=True)
        if sr != model.sr:
            audio_prompt = ta.functional.resample(audio_prompt, sr, model.sr)
        
        speaker_emb = model.ve(audio_prompt.to(model.device)).mean(dim=1)
    else:
        print("Using default speaker embedding")
        speaker_emb = torch.zeros(1, 192, device=model.device)
    
    text_tokens = model.tokenizer.text_to_tokens(text).to(model.device)
    print(f"Text tokens shape: {text_tokens.shape}")
    print(f"Text tokens (first 20): {text_tokens.flatten()[:20].tolist()}")
    
    print("Generating speech tokens...")
    with torch.no_grad():
        speech_tokens = model.t3.generate(
            text_tokens=text_tokens,
            speaker_emb=speaker_emb,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_new_tokens
        )
    
    print("Converting to audio...")
    wav = model.s3gen.decode(speech_tokens)
    
    if wav.dim() == 3:
        wav = wav.squeeze(0)
    
    ta.save(output_path, wav.cpu(), model.sr)
    print(f"Saved audio to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate Georgian speech with fine-tuned TTS')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to fine-tuned model checkpoint')
    parser.add_argument('--text', type=str, required=True,
                       help='Georgian text to synthesize')
    parser.add_argument('--output', type=str, default='output.wav',
                       help='Output audio file path')
    parser.add_argument('--audio_prompt', type=str, default=None,
                       help='Reference audio file for voice cloning')
    parser.add_argument('--exaggeration', type=float, default=0.5,
                       help='Emotion exaggeration level (0.0-1.0)')
    parser.add_argument('--cfg_weight', type=float, default=3.0,
                       help='Classifier-free guidance weight')
    parser.add_argument('--temperature', type=float, default=0.7,
                       help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=210,
                       help='Top-k sampling')
    parser.add_argument('--top_p', type=float, default=0.9,
                       help='Top-p (nucleus) sampling')
    parser.add_argument('--repetition_penalty', type=float, default=1.5,
                       help='Repetition penalty')
    parser.add_argument('--max_new_tokens', type=int, default=2048,
                       help='Maximum tokens to generate')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = load_finetuned_model(args.checkpoint, device)
    
    generate_speech_direct(
        model,
        args.text,
        audio_prompt_path=args.audio_prompt,
        output_path=args.output,
        exaggeration=args.exaggeration,
        cfg_weight=args.cfg_weight,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        max_new_tokens=args.max_new_tokens
    )


if __name__ == "__main__":
    main()

