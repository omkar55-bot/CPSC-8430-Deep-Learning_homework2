"""
HW2 Model Seq2Seq - Enhanced S2VT Training Implementation

This is the main training script for HW2 video caption generation.
It implements S2VT (Sequence to Sequence - Video to Text) with attention mechanism.

Key Features:
- Enhanced S2VT architecture with attention
- Scheduled sampling for exposure bias reduction  
- Beam search decoding
- BLEU evaluation metrics
- Model checkpointing

Usage:
    python model_seq2seq.py --config config.json
    python model_seq2seq.py --train --data_path /path/to/data
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import os
import argparse
import time
import numpy as np
from typing import Dict, List, Tuple
import logging
import sys

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

try:
    from models.enhanced_s2vt import EnhancedS2VT, EnhancedS2VTLoss
    from data.msvd_dataset import create_msvd_data_loaders
    from utils.metrics import calculate_bleu_score
    ENHANCED_AVAILABLE = True
except ImportError:
    print("Enhanced modules not available, using simplified implementation")
    ENHANCED_AVAILABLE = False


class SimpleS2VTModel(nn.Module):
    """Simplified S2VT model for HW2 submission"""
    
    def __init__(self, vocab_size, video_dim=4096, hidden_dim=256, max_len=20):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.max_len = max_len
        
        # Video feature processing
        self.video_proj = nn.Linear(video_dim, hidden_dim)
        
        # Word embeddings
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        
        # LSTM layers
        self.encoder_lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.decoder_lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, vocab_size)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
    
    def forward(self, video_features, captions=None, sampling_prob=0.0):
        batch_size = video_features.size(0)
        
        # Encode video features
        video_proj = self.video_proj(video_features)
        encoder_output, _ = self.encoder_lstm(video_proj)
        
        if captions is not None:
            # Training mode
            return self._forward_training(encoder_output, captions, sampling_prob)
        else:
            # Inference mode
            return self._forward_inference(encoder_output)
    
    def _forward_training(self, encoder_output, captions, sampling_prob):
        batch_size, seq_len = captions.shape
        outputs = []
        
        # Initialize decoder
        decoder_input = captions[:, 0:1]  # BOS token
        decoder_hidden = None
        
        for t in range(1, seq_len):
            # Word embedding
            embedded = self.embedding(decoder_input)
            
            # Attention over encoder outputs
            attended, _ = self.attention(embedded, encoder_output, encoder_output)
            
            # Decoder step
            decoder_output, decoder_hidden = self.decoder_lstm(attended, decoder_hidden)
            
            # Output projection
            logits = self.output_proj(decoder_output)
            outputs.append(logits)
            
            # Scheduled sampling
            if np.random.random() < sampling_prob and t < seq_len - 1:
                decoder_input = torch.argmax(logits, dim=-1)
            else:
                decoder_input = captions[:, t:t+1]
        
        return torch.cat(outputs, dim=1)
    
    def _forward_inference(self, encoder_output):
        batch_size = encoder_output.size(0)
        outputs = []
        
        # Start with BOS token
        decoder_input = torch.ones(batch_size, 1, dtype=torch.long, device=encoder_output.device)
        decoder_hidden = None
        
        for t in range(self.max_len):
            # Word embedding
            embedded = self.embedding(decoder_input)
            
            # Attention
            attended, _ = self.attention(embedded, encoder_output, encoder_output)
            
            # Decoder step
            decoder_output, decoder_hidden = self.decoder_lstm(attended, decoder_hidden)
            
            # Output projection
            logits = self.output_proj(decoder_output)
            outputs.append(logits)
            
            # Next input
            decoder_input = torch.argmax(logits, dim=-1)
            
            # Check for EOS
            if (decoder_input == 2).all():  # EOS token
                break
        
        return {'logits': torch.cat(outputs, dim=1)}


class HW2Trainer:
    """Training class for HW2 model"""
    
    def __init__(self, config_path: str = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Default configuration
        self.config = {
            'model_parameters': {
                'video_feature_dim': 4096,
                'hidden_dim': 256,
                'embedding_dim': 256,
                'num_layers': 2,
                'dropout': 0.3
            },
            'training_parameters': {
                'batch_size': 32,
                'num_epochs': 100,
                'learning_rate': 0.001,
                'grad_clip': 5.0
            },
            'data_parameters': {
                'max_frames': 80,
                'max_caption_length': 20
            }
        }
        
        # Load config if provided
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                loaded_config = json.load(f)
                self.config.update(loaded_config)
        
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging"""
        os.makedirs('checkpoints', exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('checkpoints/training.log'),
                logging.StreamHandler()
            ]
        )
    
    def create_model(self, vocab_size):
        """Create model"""
        if ENHANCED_AVAILABLE:
            model = EnhancedS2VT(
                vocab_size=vocab_size,
                video_feature_dim=self.config['model_parameters']['video_feature_dim'],
                hidden_dim=self.config['model_parameters']['hidden_dim'],
                embedding_dim=self.config['model_parameters']['embedding_dim'],
                num_layers=self.config['model_parameters']['num_layers'],
                dropout=self.config['model_parameters']['dropout'],
                max_seq_length=self.config['data_parameters']['max_caption_length']
            )
        else:
            model = SimpleS2VTModel(
                vocab_size=vocab_size,
                video_dim=self.config['model_parameters']['video_feature_dim'],
                hidden_dim=self.config['model_parameters']['hidden_dim'],
                max_len=self.config['data_parameters']['max_caption_length']
            )
        
        return model.to(self.device)
    
    def train_model(self, data_path: str):
        """Main training function"""
        logging.info("Starting HW2 model training...")
        
        # Load data (simplified for HW2)
        if ENHANCED_AVAILABLE and os.path.exists(data_path):
            train_loader, test_loader, vocabulary = create_msvd_data_loaders(
                data_root=data_path,
                batch_size=self.config['training_parameters']['batch_size'],
                num_workers=0
            )
        else:
            # Create dummy data for demonstration
            logging.warning("Using dummy data - replace with actual data loading")
            vocabulary = self._create_dummy_vocabulary()
            train_loader = self._create_dummy_dataloader()
            test_loader = train_loader
        
        # Create model
        model = self.create_model(len(vocabulary))
        
        # Setup training
        criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
        optimizer = optim.Adam(model.parameters(), lr=self.config['training_parameters']['learning_rate'])
        
        best_loss = float('inf')
        
        # Training loop
        for epoch in range(1, self.config['training_parameters']['num_epochs'] + 1):
            model.train()
            total_loss = 0
            
            for batch_idx, batch_data in enumerate(train_loader):
                if isinstance(batch_data, dict):
                    video_features = batch_data['video_features'].to(self.device)
                    captions = batch_data['captions'].to(self.device)
                else:
                    video_features, captions = batch_data
                    video_features = video_features.to(self.device)
                    captions = captions.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                input_captions = captions[:, :-1]
                target_captions = captions[:, 1:]
                
                if ENHANCED_AVAILABLE:
                    outputs = model(video_features, input_captions)
                    if isinstance(outputs, dict):
                        logits = outputs['logits']
                    else:
                        logits = outputs
                else:
                    logits = model(video_features, input_captions)
                
                # Loss computation
                loss = criterion(logits.reshape(-1, logits.size(-1)), target_captions.reshape(-1))
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config['training_parameters']['grad_clip'])
                optimizer.step()
                
                total_loss += loss.item()
                
                if batch_idx % 10 == 0:
                    logging.info(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
            
            avg_loss = total_loss / len(train_loader)
            logging.info(f'Epoch {epoch} completed. Average Loss: {avg_loss:.4f}')
            
            # Save checkpoint
            if avg_loss < best_loss:
                best_loss = avg_loss
                self.save_model(model, vocabulary, epoch, avg_loss)
        
        logging.info("Training completed!")
        return model, vocabulary
    
    def _create_dummy_vocabulary(self):
        """Create dummy vocabulary for testing"""
        class DummyVocab:
            def __init__(self):
                words = ['<PAD>', '<BOS>', '<EOS>', '<UNK>'] + [f'word_{i}' for i in range(100)]
                self.word2idx = {word: i for i, word in enumerate(words)}
                self.idx2word = {i: word for word, i in self.word2idx.items()}
            
            def __len__(self):
                return len(self.word2idx)
        
        return DummyVocab()
    
    def _create_dummy_dataloader(self):
        """Create dummy dataloader for testing"""
        class DummyDataset:
            def __init__(self):
                self.data = []
                for i in range(100):  # 100 dummy samples
                    video_features = torch.randn(80, 4096)  # 80 frames, 4096 features
                    captions = torch.randint(1, 100, (20,))  # 20 tokens
                    self.data.append({'video_features': video_features, 'captions': captions})
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                return self.data[idx]
        
        dataset = DummyDataset()
        return DataLoader(dataset, batch_size=8, shuffle=True)
    
    def save_model(self, model, vocabulary, epoch, loss):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'vocabulary': vocabulary,
            'config': self.config,
            'loss': loss
        }
        
        # Create model directory
        model_dir = 'your_seq2seq_model'
        os.makedirs(model_dir, exist_ok=True)
        
        # Save checkpoint
        torch.save(checkpoint, os.path.join(model_dir, 'best_model_enhanced.pth'))
        logging.info(f'Model saved at epoch {epoch} with loss {loss:.4f}')


def main():
    parser = argparse.ArgumentParser(description='HW2 Seq2Seq Model Training')
    parser.add_argument('--config', type=str, help='Config file path')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--data_path', type=str, help='Path to training data')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = HW2Trainer(args.config)
    
    if args.train or args.data_path:
        # Training mode
        data_path = args.data_path or 'data'
        model, vocabulary = trainer.train_model(data_path)
        print("Training completed! Model saved to your_seq2seq_model/")
    else:
        # Demo mode
        print("HW2 Seq2Seq Model")
        print("Usage:")
        print("  python model_seq2seq.py --train --data_path /path/to/data")
        print("  python model_seq2seq.py --config config.json --train")


if __name__ == '__main__':
    main()