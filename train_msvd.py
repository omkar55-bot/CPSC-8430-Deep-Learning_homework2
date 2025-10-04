"""
Training script for S2VT Video Caption Generation Model with MSVD Dataset
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
from data.msvd_dataset import create_msvd_data_loaders, analyze_msvd_dataset
from utils.metrics import evaluate_captions
from utils.evaluation import evaluate_model


class MSVDTrainer:
    """Trainer class for S2VT model with MSVD dataset"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model components
        self.vocabulary = None
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        
        # Data loaders
        self.train_loader = None
        self.test_loader = None
        
        # Training state
        self.current_epoch = 0
        self.best_score = float('inf')  # Lower is better for loss
        self.train_losses = []
        self.val_losses = []
        
        print(f"Using device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    def load_data(self):
        """Load MSVD dataset"""
        print("Loading MSVD dataset...")
        
        # Analyze dataset first
        if self.config.get('analyze_dataset', True):
            analyze_msvd_dataset(self.config['data_root'])
        
        # Create data loaders
        self.train_loader, self.test_loader, self.vocabulary = create_msvd_data_loaders(
            data_root=self.config['data_root'],
            batch_size=self.config['batch_size'],
            max_caption_length=self.config['max_caption_length'],
            max_frames=self.config['max_frames'],
            num_workers=self.config['num_workers'],
            vocab_save_path=os.path.join(self.config['save_dir'], 'vocabulary.pkl')
        )
        
        print(f"Training batches: {len(self.train_loader)}")
        print(f"Testing batches: {len(self.test_loader)}")
        print(f"Vocabulary size: {len(self.vocabulary)}")
        
        return len(self.vocabulary)
    
    def setup_model(self, vocab_size):
        """Initialize model, loss, and optimizer"""
        print("Setting up model...")
        
        self.model = S2VTModel(
            vocab_size=vocab_size,
            max_frames=self.config['max_frames'],
            video_feature_dim=self.config['video_feature_dim'],
            hidden_dim=self.config['hidden_dim'],
            embedding_dim=self.config['embedding_dim'],
            num_layers=self.config['num_layers'],
            dropout=self.config['dropout']
        ).to(self.device)
        
        # Loss function
        self.criterion = S2VTLoss(
            vocab_size=vocab_size,
            pad_token_id=self.vocabulary.word2idx[self.vocabulary.PAD_TOKEN]
        )
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay'],
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=self.config.get('scheduler_patience', 3),
            verbose=True,
            min_lr=1e-6
        )
        
        # Print model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"Model initialized:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Model size: {total_params * 4 / 1e6:.1f} MB")
    
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
            
            # Check for NaN loss
            if torch.isnan(loss):
                print(f"Warning: NaN loss detected at batch {batch_idx}")
                continue
            
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
                'Avg Loss': f'{avg_loss:.4f}',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
            
            # Log every N batches
            if batch_idx % self.config.get('log_interval', 100) == 0:
                print(f"Batch {batch_idx}/{num_batches}, Loss: {loss.item():.4f}")
        
        return total_loss / num_batches
    
    def validate(self):
        """Validate model"""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc='Validation'):
                # Move data to device
                video_features = batch['video_features'].to(self.device)
                captions = batch['captions'].to(self.device)
                
                # Forward pass
                outputs = self.model(video_features, captions)
                
                # Calculate loss
                targets = captions[:, 1:]  # Remove BOS token
                outputs = outputs[:, :targets.size(1), :]  # Match sequence length
                
                loss = self.criterion(outputs, targets)
                
                if not torch.isnan(loss):
                    total_loss += loss.item()
        
        return total_loss / len(self.test_loader)
    
    def evaluate_captions_quality(self, num_samples=100):
        """Evaluate caption generation quality"""
        self.model.eval()
        
        predictions = []
        references = []
        
        print(f"Evaluating caption quality on {num_samples} samples...")
        
        with torch.no_grad():
            sample_count = 0
            for batch in self.test_loader:
                if sample_count >= num_samples:
                    break
                
                video_features = batch['video_features'].to(self.device)
                caption_texts = batch['caption_texts']
                
                # Generate captions for each video in batch
                for i in range(video_features.size(0)):
                    if sample_count >= num_samples:
                        break
                    
                    single_video = video_features[i:i+1]  # Keep batch dimension
                    
                    # Generate caption
                    outputs = self.model(single_video, max_length=self.config['max_caption_length'])
                    predicted_ids = outputs.argmax(dim=-1).squeeze(0)
                    
                    # Decode caption
                    predicted_caption = self.vocabulary.decode_caption(
                        predicted_ids.cpu().numpy().tolist()
                    )
                    
                    predictions.append(predicted_caption)
                    
                    # Handle multiple reference captions
                    if isinstance(caption_texts[i], list):
                        references.append(caption_texts[i])
                    else:
                        references.append([caption_texts[i]])
                    
                    sample_count += 1
        
        # Calculate evaluation metrics
        scores = evaluate_captions(predictions, references)
        
        # Print some examples
        print("\nExample predictions:")
        for i in range(min(5, len(predictions))):
            print(f"Reference: {references[i][0]}")
            print(f"Predicted: {predictions[i]}")
            print("-" * 50)
        
        return scores
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_score': self.best_score,
            'config': self.config,
            'vocabulary_size': len(self.vocabulary)
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.config['save_dir'], f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.config['save_dir'], 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"New best model saved at epoch {epoch} with validation loss: {self.best_score:.4f}")
        
        # Keep only last N checkpoints
        self._cleanup_checkpoints(keep_last=self.config.get('keep_checkpoints', 5))
    
    def _cleanup_checkpoints(self, keep_last=5):
        """Remove old checkpoints to save disk space"""
        checkpoint_dir = self.config['save_dir']
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_epoch_')]
        
        if len(checkpoints) > keep_last:
            # Sort by epoch number
            checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
            
            # Remove oldest checkpoints
            for checkpoint in checkpoints[:-keep_last]:
                checkpoint_path = os.path.join(checkpoint_dir, checkpoint)
                try:
                    os.remove(checkpoint_path)
                    print(f"Removed old checkpoint: {checkpoint}")
                except OSError:
                    pass
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        print(f"Loading checkpoint from {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_score = checkpoint['best_score']
        
        print(f"Checkpoint loaded from epoch {self.current_epoch}")
        print(f"Best validation loss so far: {self.best_score:.4f}")
    
    def train(self):
        """Main training loop"""
        print("Starting S2VT training on MSVD dataset...")
        print("=" * 60)
        
        # Load data and setup model
        vocab_size = self.load_data()
        self.setup_model(vocab_size)
        
        # Create save directory
        os.makedirs(self.config['save_dir'], exist_ok=True)
        
        # Save configuration
        config_path = os.path.join(self.config['save_dir'], 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        # Training loop
        for epoch in range(self.current_epoch, self.config['num_epochs']):
            self.current_epoch = epoch
            
            print(f"\nEpoch {epoch+1}/{self.config['num_epochs']}")
            print("-" * 40)
            
            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate()
            self.val_losses.append(val_loss)
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            # Check if this is the best model
            is_best = val_loss < self.best_score
            if is_best:
                self.best_score = val_loss
            
            # Print epoch summary
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # Evaluate caption quality periodically
            if (epoch + 1) % self.config.get('eval_every', 5) == 0:
                caption_scores = self.evaluate_captions_quality()
                print(f"Caption Quality Metrics:")
                for metric, score in caption_scores.items():
                    print(f"  {metric}: {score:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % self.config.get('save_every', 2) == 0:
                self.save_checkpoint(epoch + 1, is_best)
            
            print("-" * 40)
        
        print("\nTraining completed!")
        print(f"Best validation loss: {self.best_score:.4f}")
        
        # Save final model
        self.save_checkpoint(self.config['num_epochs'], is_best=False)


def get_msvd_config():
    """Get default configuration for MSVD dataset"""
    return {
        # Data parameters
        'data_root': 'E:/imgsynth/MLDS_hw2_1_data',
        'max_frames': 80,
        'max_caption_length': 20,
        'analyze_dataset': True,
        
        # Model parameters (based on MSVD dataset features)
        'video_feature_dim': 4096,  # VGG/ResNet features
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
        'scheduler_patience': 3,
        
        # Logging and saving
        'save_dir': './checkpoints_msvd',
        'save_every': 2,
        'eval_every': 5,
        'log_interval': 50,
        'keep_checkpoints': 5,
    }


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train S2VT on MSVD Dataset')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--data_root', type=str, default='E:/imgsynth/MLDS_hw2_1_data', 
                       help='Root directory of MLDS_hw2_1_data')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=None, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=None, help='Number of epochs')
    args = parser.parse_args()
    
    # Load configuration
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = get_msvd_config()
    
    # Override config with command line arguments
    if args.data_root:
        config['data_root'] = args.data_root
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.learning_rate:
        config['learning_rate'] = args.learning_rate
    if args.num_epochs:
        config['num_epochs'] = args.num_epochs
    
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Create trainer
    trainer = MSVDTrainer(config)
    
    # Resume from checkpoint if specified
    if args.resume and os.path.exists(args.resume):
        trainer.load_checkpoint(args.resume)
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()