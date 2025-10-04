"""
Training script for S2VT Video Caption Generation Model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import json
from tqdm import tqdm
import argparse
from datetime import datetime

# Import our modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.s2vt_model import S2VTModel, S2VTLoss
from data.preprocessing import Vocabulary, VideoCaptionDataset, create_data_loader, load_sample_data, preprocess_captions
from utils.metrics import calculate_bleu_score, calculate_meteor_score, calculate_cider_score
from utils.evaluation import evaluate_model


class S2VTTrainer:
    """Trainer class for S2VT model"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model components
        self.vocabulary = None
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        
        # Training state
        self.current_epoch = 0
        self.best_score = 0.0
        self.train_losses = []
        self.val_losses = []
        
        print(f"Using device: {self.device}")
    
    def setup_model(self, vocab_size):
        """Initialize model, loss, and optimizer"""
        self.model = S2VTModel(
            vocab_size=vocab_size,
            max_frames=self.config['max_frames'],
            video_feature_dim=self.config['video_feature_dim'],
            hidden_dim=self.config['hidden_dim'],
            embedding_dim=self.config['embedding_dim'],
            num_layers=self.config['num_layers'],
            dropout=self.config['dropout']
        ).to(self.device)
        
        self.criterion = S2VTLoss(
            vocab_size=vocab_size,
            pad_token_id=self.vocabulary.word2idx[self.vocabulary.PAD_TOKEN]
        )
        
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            verbose=True
        )
        
        print(f"Model initialized with {sum(p.numel() for p in self.model.parameters())} parameters")
    
    def load_data(self):
        """Load and preprocess data"""
        print("Loading data...")
        
        # Load video features and captions
        video_features, captions = load_sample_data(self.config['data_path'])
        captions = preprocess_captions(captions)
        
        # Build vocabulary
        self.vocabulary = Vocabulary()
        self.vocabulary.build_vocabulary(captions, min_count=self.config['min_word_count'])
        
        # Save vocabulary
        os.makedirs(self.config['save_dir'], exist_ok=True)
        vocab_path = os.path.join(self.config['save_dir'], 'vocabulary.pkl')
        self.vocabulary.save(vocab_path)
        
        # Split data (simple train/val split)
        split_idx = int(0.8 * len(video_features))
        
        train_features = video_features[:split_idx]
        train_captions = captions[:split_idx]
        val_features = video_features[split_idx:]
        val_captions = captions[split_idx:]
        
        # Create datasets
        train_dataset = VideoCaptionDataset(
            train_features, train_captions, self.vocabulary,
            max_caption_length=self.config['max_caption_length'],
            max_frames=self.config['max_frames']
        )
        
        val_dataset = VideoCaptionDataset(
            val_features, val_captions, self.vocabulary,
            max_caption_length=self.config['max_caption_length'],
            max_frames=self.config['max_frames']
        )
        
        # Create data loaders
        self.train_loader = create_data_loader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['num_workers']
        )
        
        self.val_loader = create_data_loader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers']
        )
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        
        return len(self.vocabulary)
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch+1}')
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move data to device
            video_features = batch['video_features'].to(self.device)
            captions = batch['captions'].to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(video_features, captions)
            
            # Calculate loss (exclude BOS token from targets)
            targets = captions[:, 1:]  # Remove BOS token
            outputs = outputs[:, :targets.size(1), :]  # Match sequence length
            
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
            
            # Update weights
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{avg_loss:.4f}'
            })
        
        return total_loss / num_batches
    
    def validate(self):
        """Validate model"""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                # Move data to device
                video_features = batch['video_features'].to(self.device)
                captions = batch['captions'].to(self.device)
                
                # Forward pass
                outputs = self.model(video_features, captions)
                
                # Calculate loss
                targets = captions[:, 1:]  # Remove BOS token
                outputs = outputs[:, :targets.size(1), :]  # Match sequence length
                
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
        
        return total_loss / len(self.val_loader)
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_score': self.best_score,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.config['save_dir'], f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.config['save_dir'], 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"New best model saved at epoch {epoch}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_score = checkpoint['best_score']
        
        print(f"Checkpoint loaded from epoch {self.current_epoch}")
    
    def train(self):
        """Main training loop"""
        print("Starting training...")
        
        # Load data and setup model
        vocab_size = self.load_data()
        self.setup_model(vocab_size)
        
        # Training loop
        for epoch in range(self.current_epoch, self.config['num_epochs']):
            self.current_epoch = epoch
            
            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate()
            self.val_losses.append(val_loss)
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            # Check if this is the best model
            is_best = val_loss < self.best_score if self.best_score > 0 else True
            if is_best:
                self.best_score = val_loss
            
            # Save checkpoint
            if (epoch + 1) % self.config['save_every'] == 0:
                self.save_checkpoint(epoch + 1, is_best)
            
            # Print epoch summary
            print(f"Epoch {epoch+1}/{self.config['num_epochs']}")
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            print("-" * 50)
        
        print("Training completed!")
        
        # Save final model
        self.save_checkpoint(self.config['num_epochs'], is_best=False)


def get_default_config():
    """Get default training configuration"""
    return {
        # Data parameters
        'data_path': './data',
        'max_frames': 80,
        'max_caption_length': 20,
        'min_word_count': 2,
        
        # Model parameters
        'video_feature_dim': 4096,
        'hidden_dim': 512,
        'embedding_dim': 512,
        'num_layers': 2,
        'dropout': 0.5,
        
        # Training parameters
        'batch_size': 32,
        'num_epochs': 50,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'grad_clip': 5.0,
        'num_workers': 4,
        
        # Save parameters
        'save_dir': './checkpoints',
        'save_every': 5,
    }


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train S2VT Video Caption Generation Model')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    # Load configuration
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = get_default_config()
    
    # Create trainer
    trainer = S2VTTrainer(config)
    
    # Resume from checkpoint if specified
    if args.resume and os.path.exists(args.resume):
        trainer.load_checkpoint(args.resume)
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()