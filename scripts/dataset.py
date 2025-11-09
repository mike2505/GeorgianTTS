import torch
import torchaudio
import pandas as pd
import librosa
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


class GeorgianTTSDataset(Dataset):
    def __init__(
        self,
        metadata_csv,
        tokenizer,
        s3_tokenizer,
        voice_encoder,
        max_text_len=512,
        max_speech_len=2048,
        sample_rate=24000,
        ref_audio_len_sec=6
    ):
        self.df = pd.read_csv(metadata_csv)
        self.tokenizer = tokenizer
        self.s3_tokenizer = s3_tokenizer
        self.voice_encoder = voice_encoder
        self.max_text_len = max_text_len
        self.max_speech_len = max_speech_len
        self.sample_rate = sample_rate
        self.ref_audio_len = int(ref_audio_len_sec * 16000)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        audio_path = row['audio_path']
        transcript = row['transcript']
        speaker_id = row['speaker_id']
        
        try:
            wav, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            if len(wav) > self.sample_rate * 15:
                wav = wav[:self.sample_rate * 15]
            
            ref_wav_16k = librosa.resample(wav, orig_sr=self.sample_rate, target_sr=16000)
            ref_wav_16k = ref_wav_16k[:self.ref_audio_len]
            
            text_tokens = self.tokenizer.text_to_tokens(transcript)
            
            if text_tokens.shape[-1] > self.max_text_len:
                text_tokens = text_tokens[:, :self.max_text_len]
            
            speech_tokens, _ = self.s3_tokenizer.forward([ref_wav_16k], max_len=self.max_speech_len)
            speech_tokens = torch.tensor(speech_tokens) if not isinstance(speech_tokens, torch.Tensor) else speech_tokens
            
            speaker_emb = torch.from_numpy(
                self.voice_encoder.embeds_from_wavs([ref_wav_16k], sample_rate=16000)
            )
            speaker_emb = speaker_emb.mean(axis=0)
            
            return {
                'text_tokens': text_tokens.squeeze(0) if text_tokens.dim() > 1 else text_tokens,
                'speech_tokens': speech_tokens.squeeze(0) if speech_tokens.dim() > 1 else speech_tokens,
                'speaker_emb': speaker_emb,
                'audio_path': audio_path,
                'transcript': transcript,
                'speaker_id': speaker_id
            }
            
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            return {
                'text_tokens': torch.zeros(10, dtype=torch.long),
                'speech_tokens': torch.zeros(10, dtype=torch.long),
                'speaker_emb': torch.zeros(192),
                'audio_path': audio_path,
                'transcript': transcript,
                'speaker_id': speaker_id
            }


def collate_fn(batch):
    text_tokens = []
    speech_tokens = []
    speaker_embs = []
    text_lengths = []
    speech_lengths = []
    
    for item in batch:
        text_tokens.append(item['text_tokens'])
        speech_tokens.append(item['speech_tokens'])
        speaker_embs.append(item['speaker_emb'])
        text_lengths.append(item['text_tokens'].shape[0])
        speech_lengths.append(item['speech_tokens'].shape[0])
    
    max_text_len = max(text_lengths)
    max_speech_len = max(speech_lengths)
    
    text_tokens_padded = []
    speech_tokens_padded = []
    text_masks = []
    speech_masks = []
    
    for i, item in enumerate(batch):
        text_pad_len = max_text_len - text_lengths[i]
        speech_pad_len = max_speech_len - speech_lengths[i]
        
        text_tokens_padded.append(
            F.pad(item['text_tokens'], (0, text_pad_len), value=0)
        )
        speech_tokens_padded.append(
            F.pad(item['speech_tokens'], (0, speech_pad_len), value=0)
        )
        
        text_mask = torch.ones(max_text_len, dtype=torch.bool)
        text_mask[text_lengths[i]:] = False
        text_masks.append(text_mask)
        
        speech_mask = torch.ones(max_speech_len, dtype=torch.bool)
        speech_mask[speech_lengths[i]:] = False
        speech_masks.append(speech_mask)
    
    return {
        'text_tokens': torch.stack(text_tokens_padded),
        'speech_tokens': torch.stack(speech_tokens_padded),
        'speaker_embs': torch.stack(speaker_embs),
        'text_masks': torch.stack(text_masks),
        'speech_masks': torch.stack(speech_masks),
        'text_lengths': torch.tensor(text_lengths),
        'speech_lengths': torch.tensor(speech_lengths),
    }


def create_dataloaders(
    train_csv,
    val_csv,
    tokenizer,
    s3_tokenizer,
    voice_encoder,
    batch_size=4,
    num_workers=0
):
    train_dataset = GeorgianTTSDataset(
        train_csv,
        tokenizer,
        s3_tokenizer,
        voice_encoder
    )
    
    val_dataset = GeorgianTTSDataset(
        val_csv,
        tokenizer,
        s3_tokenizer,
        voice_encoder
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=False
    )
    
    return train_loader, val_loader

