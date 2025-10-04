"""
Data preprocessing utilities for video caption generation
Handles vocabulary creation, tokenization, and data loading
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import json
import pickle
import numpy as np
from collections import Counter
import re
from typing import List, Dict, Tuple, Optional


class Vocabulary:
    """Vocabulary class for handling word-to-index mapping"""
    
    def __init__(self):
        # Special tokens
        self.PAD_TOKEN = '<PAD>'
        self.BOS_TOKEN = '<BOS>'  # Begin of sentence
        self.EOS_TOKEN = '<EOS>'  # End of sentence
        self.UNK_TOKEN = '<UNK>'  # Unknown word
        
        # Initialize mappings
        self.word2idx = {}
        self.idx2word = {}
        self.word_count = Counter()
        
        # Add special tokens
        self._add_special_tokens()
    
    def _add_special_tokens(self):
        """Add special tokens to vocabulary"""
        special_tokens = [self.PAD_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN, self.UNK_TOKEN]
        for token in special_tokens:
            self.word2idx[token] = len(self.word2idx)
            self.idx2word[len(self.idx2word)] = token
    
    def add_word(self, word: str):
        """Add a word to vocabulary"""
        self.word_count[word] += 1
        if word not in self.word2idx:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
    
    def build_vocabulary(self, captions: List[str], min_count: int = 2):
        """
        Build vocabulary from caption list
        
        Args:
            captions: List of caption strings
            min_count: Minimum word frequency to include in vocabulary
        """
        # Tokenize and count words
        for caption in captions:
            tokens = self.tokenize(caption)
            for token in tokens:
                self.word_count[token] += 1
        
        # Add words that meet minimum count threshold
        for word, count in self.word_count.items():
            if count >= min_count:
                self.add_word(word)
        
        print(f"Vocabulary built with {len(self.word2idx)} words")
        print(f"Most common words: {self.word_count.most_common(10)}")
    
    def tokenize(self, caption: str) -> List[str]:
        """
        Tokenize caption string
        
        Args:
            caption: Input caption string
            
        Returns:
            List of tokens
        """
        # Convert to lowercase and remove extra spaces
        caption = caption.lower().strip()
        
        # Remove punctuation and split
        caption = re.sub(r'[^\w\s]', '', caption)
        tokens = caption.split()
        
        return tokens
    
    def encode_caption(self, caption: str, max_length: int = 20) -> List[int]:
        """
        Convert caption to sequence of indices
        
        Args:
            caption: Input caption string
            max_length: Maximum sequence length
            
        Returns:
            List of token indices
        """
        tokens = self.tokenize(caption)
        
        # Add BOS and EOS tokens
        tokens = [self.BOS_TOKEN] + tokens + [self.EOS_TOKEN]
        
        # Convert to indices
        indices = []
        for token in tokens:
            if token in self.word2idx:
                indices.append(self.word2idx[token])
            else:
                indices.append(self.word2idx[self.UNK_TOKEN])
        
        # Pad or truncate to max_length
        if len(indices) < max_length:
            indices.extend([self.word2idx[self.PAD_TOKEN]] * (max_length - len(indices)))
        else:
            indices = indices[:max_length-1] + [self.word2idx[self.EOS_TOKEN]]
        
        return indices
    
    def decode_caption(self, indices: List[int], remove_special: bool = True) -> str:
        """
        Convert sequence of indices back to caption
        
        Args:
            indices: List of token indices
            remove_special: Whether to remove special tokens
            
        Returns:
            Decoded caption string
        """
        tokens = []
        for idx in indices:
            if idx in self.idx2word:
                token = self.idx2word[idx]
                if remove_special and token in [self.PAD_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN]:
                    if token == self.EOS_TOKEN:
                        break
                    continue
                tokens.append(token)
        
        return ' '.join(tokens)
    
    def __len__(self):
        return len(self.word2idx)
    
    def save(self, filepath: str):
        """Save vocabulary to file"""
        vocab_data = {
            'word2idx': self.word2idx,
            'idx2word': self.idx2word,
            'word_count': dict(self.word_count)
        }
        with open(filepath, 'wb') as f:
            pickle.dump(vocab_data, f)
        print(f"Vocabulary saved to {filepath}")
    
    def load(self, filepath: str):
        """Load vocabulary from file"""
        with open(filepath, 'rb') as f:
            vocab_data = pickle.load(f)
        
        self.word2idx = vocab_data['word2idx']
        self.idx2word = vocab_data['idx2word']
        self.word_count = Counter(vocab_data['word_count'])
        print(f"Vocabulary loaded from {filepath}")


class VideoCaptionDataset(Dataset):
    """Dataset class for video caption data"""
    
    def __init__(self, 
                 video_features: np.ndarray,
                 captions: List[str],
                 vocabulary: Vocabulary,
                 max_caption_length: int = 20,
                 max_frames: int = 80):
        """
        Initialize dataset
        
        Args:
            video_features: Array of shape (num_videos, num_frames, feature_dim)
            captions: List of caption strings
            vocabulary: Vocabulary object
            max_caption_length: Maximum caption length
            max_frames: Maximum number of video frames
        """
        self.video_features = video_features
        self.captions = captions
        self.vocabulary = vocabulary
        self.max_caption_length = max_caption_length
        self.max_frames = max_frames
        
        # Encode captions
        self.encoded_captions = []
        for caption in captions:
            encoded = vocabulary.encode_caption(caption, max_caption_length)
            self.encoded_captions.append(encoded)
        
        print(f"Dataset initialized with {len(self.video_features)} samples")
    
    def __len__(self):
        return len(self.video_features)
    
    def __getitem__(self, idx):
        """Get a single sample"""
        # Get video features
        video_feat = self.video_features[idx]  # (num_frames, feature_dim)
        
        # Pad or truncate video frames
        num_frames = video_feat.shape[0]
        if num_frames < self.max_frames:
            # Pad with zeros
            padding = np.zeros((self.max_frames - num_frames, video_feat.shape[1]))
            video_feat = np.concatenate([video_feat, padding], axis=0)
        else:
            # Truncate
            video_feat = video_feat[:self.max_frames]
        
        # Get caption
        caption = torch.tensor(self.encoded_captions[idx], dtype=torch.long)
        
        # Convert video features to tensor
        video_feat = torch.tensor(video_feat, dtype=torch.float32)
        
        return {
            'video_features': video_feat,
            'caption': caption,
            'caption_text': self.captions[idx]
        }


def create_data_loader(dataset: VideoCaptionDataset, 
                      batch_size: int = 32, 
                      shuffle: bool = True,
                      num_workers: int = 4) -> DataLoader:
    """Create DataLoader for the dataset"""
    
    def collate_fn(batch):
        """Custom collate function"""
        video_features = torch.stack([item['video_features'] for item in batch])
        captions = torch.stack([item['caption'] for item in batch])
        caption_texts = [item['caption_text'] for item in batch]
        
        return {
            'video_features': video_features,
            'captions': captions,
            'caption_texts': caption_texts
        }
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )


def load_sample_data(data_path: str) -> Tuple[np.ndarray, List[str]]:
    """
    Load sample video features and captions
    This is a placeholder function - replace with your actual data loading logic
    
    Args:
        data_path: Path to data directory
        
    Returns:
        Tuple of (video_features, captions)
    """
    # This is a sample implementation
    # Replace with your actual data loading code
    
    # Sample random video features (num_videos, num_frames, feature_dim)
    num_videos = 1000
    num_frames = 60
    feature_dim = 4096
    
    video_features = np.random.randn(num_videos, num_frames, feature_dim).astype(np.float32)
    
    # Sample captions
    sample_captions = [
        "a man is walking down the street",
        "a woman is cooking in the kitchen",
        "children are playing in the park",
        "a dog is running in the yard",
        "people are dancing at a party",
        "a car is driving on the highway",
        "birds are flying in the sky",
        "a cat is sleeping on the bed",
        "students are studying in the library",
        "workers are building a house"
    ]
    
    # Repeat and shuffle captions to match number of videos
    captions = []
    for i in range(num_videos):
        captions.append(sample_captions[i % len(sample_captions)])
    
    return video_features, captions


def preprocess_captions(captions: List[str]) -> List[str]:
    """
    Preprocess caption text
    
    Args:
        captions: List of raw captions
        
    Returns:
        List of preprocessed captions
    """
    processed = []
    for caption in captions:
        # Convert to lowercase
        caption = caption.lower().strip()
        
        # Remove extra whitespace
        caption = re.sub(r'\s+', ' ', caption)
        
        # Remove or replace special characters if needed
        caption = re.sub(r'[^\w\s]', '', caption)
        
        processed.append(caption)
    
    return processed


if __name__ == "__main__":
    # Example usage
    print("Testing data preprocessing...")
    
    # Load sample data
    video_features, captions = load_sample_data("./data")
    
    # Preprocess captions
    captions = preprocess_captions(captions)
    
    # Build vocabulary
    vocab = Vocabulary()
    vocab.build_vocabulary(captions, min_count=1)
    
    # Create dataset
    dataset = VideoCaptionDataset(video_features, captions, vocab)
    
    # Create data loader
    data_loader = create_data_loader(dataset, batch_size=8)
    
    # Test data loading
    for batch in data_loader:
        print(f"Batch video features shape: {batch['video_features'].shape}")
        print(f"Batch captions shape: {batch['captions'].shape}")
        print(f"Sample caption: {batch['caption_texts'][0]}")
        break
    
    print("Data preprocessing test completed!")