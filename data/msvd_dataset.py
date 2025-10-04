"""
MSVD Dataset Loader for S2VT Video Caption Generation
Handles the MLDS_hw2_1_data dataset format with pre-extracted video features
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional
import pickle
from collections import Counter
import random

# Import base classes
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.preprocessing import Vocabulary


class MSVDDataset(Dataset):
    """MSVD Dataset class for video caption data"""
    
    def __init__(self, 
                 data_root: str,
                 split: str = 'train',
                 vocabulary: Optional[Vocabulary] = None,
                 max_caption_length: int = 20,
                 max_frames: int = 80,
                 caption_per_video: int = 1):
        """
        Initialize MSVD dataset
        
        Args:
            data_root: Root directory of MLDS_hw2_1_data
            split: 'train' or 'test'
            vocabulary: Vocabulary object (if None, will build from training data)
            max_caption_length: Maximum caption length
            max_frames: Maximum number of video frames
            caption_per_video: Number of captions to use per video during training
        """
        self.data_root = data_root
        self.split = split
        self.max_caption_length = max_caption_length
        self.max_frames = max_frames
        self.caption_per_video = caption_per_video
        
        # Set paths based on split
        if split == 'train':
            self.feat_dir = os.path.join(data_root, 'training_data', 'feat')
            self.label_file = os.path.join(data_root, 'training_label.json')
        else:  # test
            self.feat_dir = os.path.join(data_root, 'testing_data', 'feat')
            self.label_file = os.path.join(data_root, 'testing_label.json')
        
        # Load data
        self.data = self._load_data()
        
        # Handle vocabulary
        if vocabulary is None and split == 'train':
            # Build vocabulary from training data
            all_captions = []
            for item in self.data:
                all_captions.extend(item['captions'])
            
            self.vocabulary = Vocabulary()
            self.vocabulary.build_vocabulary(all_captions, min_count=2)
        else:
            self.vocabulary = vocabulary
        
        # Encode captions
        self._encode_captions()
        
        print(f"MSVD {split} dataset initialized:")
        print(f"  - {len(self.data)} videos")
        print(f"  - Vocabulary size: {len(self.vocabulary) if self.vocabulary else 'None'}")
        print(f"  - Average captions per video: {np.mean([len(item['captions']) for item in self.data]):.1f}")
    
    def _load_data(self) -> List[Dict]:
        """Load video data and captions"""
        print(f"Loading {self.split} data from {self.data_root}")
        
        # Load labels
        with open(self.label_file, 'r', encoding='utf-8') as f:
            labels = json.load(f)
        
        # Process each video
        data = []
        missing_features = 0
        
        for item in labels:
            video_id = item['id']
            captions = item['caption']
            
            # Check if feature file exists
            feat_file = os.path.join(self.feat_dir, f"{video_id}.npy")
            
            if os.path.exists(feat_file):
                data.append({
                    'video_id': video_id,
                    'captions': captions,
                    'feat_file': feat_file
                })
            else:
                missing_features += 1
        
        if missing_features > 0:
            print(f"Warning: {missing_features} videos have missing feature files")
        
        print(f"Loaded {len(data)} videos with features")
        return data
    
    def _encode_captions(self):
        """Encode captions using vocabulary"""
        if self.vocabulary is None:
            return
        
        for item in self.data:
            encoded_captions = []
            for caption in item['captions']:
                encoded = self.vocabulary.encode_caption(caption, self.max_caption_length)
                encoded_captions.append(encoded)
            item['encoded_captions'] = encoded_captions
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """Get a single sample"""
        item = self.data[idx]
        
        # Load video features
        video_features = np.load(item['feat_file'])  # Shape: (num_frames, feature_dim)
        
        # Handle frame padding/truncation
        num_frames = video_features.shape[0]
        feature_dim = video_features.shape[1]
        
        if num_frames < self.max_frames:
            # Pad with zeros
            padding = np.zeros((self.max_frames - num_frames, feature_dim), dtype=video_features.dtype)
            video_features = np.concatenate([video_features, padding], axis=0)
        else:
            # Truncate (take evenly spaced frames)
            indices = np.linspace(0, num_frames - 1, self.max_frames, dtype=int)
            video_features = video_features[indices]
        
        # Select caption(s)
        if self.split == 'train':
            # For training, randomly select one or multiple captions
            if self.caption_per_video == 1:
                # Select one random caption
                caption_idx = random.randint(0, len(item['captions']) - 1)
                caption_text = item['captions'][caption_idx]
                
                if self.vocabulary and 'encoded_captions' in item:
                    caption_encoded = item['encoded_captions'][caption_idx]
                else:
                    caption_encoded = None
            else:
                # Select multiple random captions (for multi-reference training)
                num_captions = min(self.caption_per_video, len(item['captions']))
                caption_indices = random.sample(range(len(item['captions'])), num_captions)
                caption_text = [item['captions'][i] for i in caption_indices]
                
                if self.vocabulary and 'encoded_captions' in item:
                    caption_encoded = [item['encoded_captions'][i] for i in caption_indices]
                else:
                    caption_encoded = None
        else:
            # For testing, return all captions
            caption_text = item['captions']
            if self.vocabulary and 'encoded_captions' in item:
                caption_encoded = item['encoded_captions']
            else:
                caption_encoded = None
        
        # Convert to tensors
        video_features = torch.tensor(video_features, dtype=torch.float32)
        
        if caption_encoded is not None:
            if isinstance(caption_encoded[0], list):
                # Multiple captions
                caption_tensor = torch.tensor(caption_encoded, dtype=torch.long)
            else:
                # Single caption
                caption_tensor = torch.tensor(caption_encoded, dtype=torch.long)
        else:
            caption_tensor = None
        
        return {
            'video_id': item['video_id'],
            'video_features': video_features,
            'caption_text': caption_text,
            'caption_encoded': caption_tensor,
            'num_original_frames': min(num_frames, self.max_frames)
        }


def create_msvd_data_loaders(data_root: str,
                            batch_size: int = 32,
                            max_caption_length: int = 20,
                            max_frames: int = 80,
                            num_workers: int = 4,
                            vocab_save_path: Optional[str] = None) -> Tuple[DataLoader, DataLoader, Vocabulary]:
    """
    Create MSVD data loaders for training and testing
    
    Args:
        data_root: Root directory of MLDS_hw2_1_data
        batch_size: Batch size
        max_caption_length: Maximum caption length
        max_frames: Maximum number of frames
        num_workers: Number of worker processes
        vocab_save_path: Path to save vocabulary
        
    Returns:
        Tuple of (train_loader, test_loader, vocabulary)
    """
    
    # Create training dataset and build vocabulary
    print("Creating training dataset...")
    train_dataset = MSVDDataset(
        data_root=data_root,
        split='train',
        vocabulary=None,  # Will build vocabulary
        max_caption_length=max_caption_length,
        max_frames=max_frames,
        caption_per_video=1
    )
    
    # Get vocabulary from training dataset
    vocabulary = train_dataset.vocabulary
    
    # Save vocabulary if path provided
    if vocab_save_path:
        os.makedirs(os.path.dirname(vocab_save_path), exist_ok=True)
        vocabulary.save(vocab_save_path)
    
    # Create testing dataset
    print("Creating testing dataset...")
    test_dataset = MSVDDataset(
        data_root=data_root,
        split='test',
        vocabulary=vocabulary,
        max_caption_length=max_caption_length,
        max_frames=max_frames
    )
    
    # Custom collate function
    def collate_fn(batch):
        """Custom collate function to handle variable caption lengths"""
        video_ids = [item['video_id'] for item in batch]
        video_features = torch.stack([item['video_features'] for item in batch])
        caption_texts = [item['caption_text'] for item in batch]
        num_frames = torch.tensor([item['num_original_frames'] for item in batch])
        
        # Handle encoded captions
        caption_encoded = []
        for item in batch:
            if item['caption_encoded'] is not None:
                if len(item['caption_encoded'].shape) == 1:
                    # Single caption
                    caption_encoded.append(item['caption_encoded'])
                else:
                    # Multiple captions - take first one for batch processing
                    caption_encoded.append(item['caption_encoded'][0])
            else:
                # Create dummy encoding if not available
                caption_encoded.append(torch.zeros(max_caption_length, dtype=torch.long))
        
        caption_encoded = torch.stack(caption_encoded)
        
        return {
            'video_ids': video_ids,
            'video_features': video_features,
            'captions': caption_encoded,
            'caption_texts': caption_texts,
            'num_frames': num_frames
        }
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return train_loader, test_loader, vocabulary


def analyze_msvd_dataset(data_root: str):
    """
    Analyze MSVD dataset statistics
    
    Args:
        data_root: Root directory of MLDS_hw2_1_data
    """
    print("Analyzing MSVD Dataset...")
    print("=" * 50)
    
    # Load training and test data
    train_dataset = MSVDDataset(data_root, split='train')
    test_dataset = MSVDDataset(data_root, split='test', vocabulary=train_dataset.vocabulary)
    
    # Dataset statistics
    print(f"Training videos: {len(train_dataset)}")
    print(f"Testing videos: {len(test_dataset)}")
    
    # Caption statistics
    train_captions = []
    train_caption_lengths = []
    
    for item in train_dataset.data:
        train_captions.extend(item['captions'])
        train_caption_lengths.extend([len(cap.split()) for cap in item['captions']])
    
    print(f"\nCaption Statistics:")
    print(f"Total training captions: {len(train_captions)}")
    print(f"Average captions per video: {len(train_captions) / len(train_dataset):.1f}")
    print(f"Average caption length: {np.mean(train_caption_lengths):.1f} words")
    print(f"Caption length std: {np.std(train_caption_lengths):.1f} words")
    print(f"Min caption length: {min(train_caption_lengths)} words")
    print(f"Max caption length: {max(train_caption_lengths)} words")
    
    # Video feature statistics
    sample_features = []
    frame_counts = []
    
    for i in range(min(100, len(train_dataset))):  # Sample first 100 videos
        item = train_dataset[i]
        features = np.load(train_dataset.data[i]['feat_file'])
        sample_features.append(features)
        frame_counts.append(features.shape[0])
    
    print(f"\nVideo Feature Statistics (sample of {len(sample_features)} videos):")
    print(f"Feature dimension: {sample_features[0].shape[1]}")
    print(f"Average frames per video: {np.mean(frame_counts):.1f}")
    print(f"Frame count std: {np.std(frame_counts):.1f}")
    print(f"Min frames: {min(frame_counts)}")
    print(f"Max frames: {max(frame_counts)}")
    
    # Vocabulary statistics
    vocab = train_dataset.vocabulary
    print(f"\nVocabulary Statistics:")
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Most common words: {vocab.word_count.most_common(10)}")
    
    return train_dataset, test_dataset


if __name__ == "__main__":
    # Example usage
    data_root = "E:/imgsynth/MLDS_hw2_1_data"
    
    if os.path.exists(data_root):
        # Analyze dataset
        train_dataset, test_dataset = analyze_msvd_dataset(data_root)
        
        # Create data loaders
        train_loader, test_loader, vocabulary = create_msvd_data_loaders(
            data_root=data_root,
            batch_size=8,
            vocab_save_path="./msvd_vocabulary.pkl"
        )
        
        # Test data loading
        print(f"\nTesting data loading...")
        for batch in train_loader:
            print(f"Video features shape: {batch['video_features'].shape}")
            print(f"Captions shape: {batch['captions'].shape}")
            print(f"Sample video ID: {batch['video_ids'][0]}")
            print(f"Sample caption: {batch['caption_texts'][0]}")
            break
        
        print("MSVD dataset loading test completed!")
    else:
        print(f"Dataset not found at {data_root}")
        print("Please extract MLDS_hw2_1_data.tar.gz first")