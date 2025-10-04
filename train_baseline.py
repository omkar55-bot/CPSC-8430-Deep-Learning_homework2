"""
Enhanced Training Script with Advanced Training Tips
Implements attention, scheduled sampling, and baseline-reaching configurations
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
import math

# Import our modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.attention_s2vt import AttentionS2VT, ScheduledSamplingS2VT, BeamSearchDecoder
from models.s2vt_model import S2VTLoss
from data.msvd_dataset import create_msvd_data_loaders, analyze_msvd_dataset
from utils.metrics import evaluate_captions
from utils.evaluation import evaluate_model


class AdvancedMSVDTrainer:
    """
    Advanced trainer implementing all training tips for reaching baseline performance
    """
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model components
        self.vocabulary = None
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.beam_decoder = None
        
        # Data loaders
        self.train_loader = None
        self.test_loader = None
        
        # Training state
        self.current_epoch = 0
        self.best_bleu_score = 0.0
        self.train_losses = []
        self.val_losses = []
        self.bleu_scores = []
        
        # Scheduled sampling
        self.use_scheduled_sampling = config.get('use_scheduled_sampling', True)
        self.initial_sampling_prob = config.get('initial_sampling_prob', 0.0)
        self.final_sampling_prob = config.get('final_sampling_prob', 0.25)
        self.sampling_schedule_epochs = config.get('sampling_schedule_epochs', 100)
        
        print(f"Advanced MSVD Trainer initialized on {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    def load_data(self):
        """Load MSVD dataset with enhanced preprocessing"""
        print("Loading MSVD dataset...")
        
        # Create data loaders with specific parameters for baseline
        self.train_loader, self.test_loader, self.vocabulary = create_msvd_data_loaders(
            data_root=self.config['data_root'],
            batch_size=self.config['batch_size'],
            max_caption_length=self.config['max_caption_length'],
            max_frames=self.config['max_frames'],
            num_workers=self.config['num_workers'],
            vocab_save_path=os.path.join(self.config['save_dir'], 'vocabulary.pkl')
        )
        
        # Filter vocabulary based on min_count for baseline
        if self.config.get('vocab_min_count', 3) > 2:
            print(f"Filtering vocabulary with min_count={self.config['vocab_min_count']}")
            self._filter_vocabulary()
        
        print(f"Training batches: {len(self.train_loader)}")
        print(f"Testing batches: {len(self.test_loader)}")
        print(f"Vocabulary size: {len(self.vocabulary)}")
        
        return len(self.vocabulary)
    
    def _filter_vocabulary(self):
        """Filter vocabulary based on minimum word count (baseline requirement)"""
        min_count = self.config['vocab_min_count']
        
        # Create new vocabulary with higher threshold
        new_word2idx = {}
        new_idx2word = {}
        
        # Add special tokens first
        special_tokens = [
            self.vocabulary.PAD_TOKEN,
            self.vocabulary.BOS_TOKEN,
            self.vocabulary.EOS_TOKEN,
            self.vocabulary.UNK_TOKEN
        ]
        
        for token in special_tokens:
            idx = len(new_word2idx)
            new_word2idx[token] = idx
            new_idx2word[idx] = token
        
        # Add words that meet the threshold
        for word, count in self.vocabulary.word_count.items():
            if count >= min_count and word not in special_tokens:
                idx = len(new_word2idx)
                new_word2idx[word] = idx
                new_idx2word[idx] = word
        
        # Update vocabulary
        self.vocabulary.word2idx = new_word2idx
        self.vocabulary.idx2word = new_idx2word
        
        print(f"Vocabulary filtered: {len(new_word2idx)} words (min_count >= {min_count})")
    
    def setup_model(self, vocab_size):
        """Setup model with advanced features"""
        print("Setting up advanced S2VT model...")
        
        model_type = self.config.get('model_type', 'attention')
        
        if model_type == 'attention':
            self.model = AttentionS2VT(
                vocab_size=vocab_size,
                max_frames=self.config['max_frames'],
                video_feature_dim=self.config['video_feature_dim'],
                hidden_dim=self.config['hidden_dim'],
                embedding_dim=self.config['embedding_dim'],
                num_layers=self.config['num_layers'],
                dropout=self.config['dropout'],
                attention_dim=self.config.get('attention_dim', 256)
            ).to(self.device)
            
        elif model_type == 'scheduled_sampling':
            self.model = ScheduledSamplingS2VT(
                vocab_size=vocab_size,
                max_frames=self.config['max_frames'],
                video_feature_dim=self.config['video_feature_dim'],
                hidden_dim=self.config['hidden_dim'],
                embedding_dim=self.config['embedding_dim'],
                num_layers=self.config['num_layers'],
                dropout=self.config['dropout'],
                attention_dim=self.config.get('attention_dim', 256)
            ).to(self.device)
        
        else:
            # Basic S2VT
            from models.s2vt_model import S2VTModel
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
        
        # Optimizer - Use Adam as per baseline
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config.get('weight_decay', 0),
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler - not used in baseline
        if self.config.get('use_scheduler', False):
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',  # Maximize BLEU score
                factor=0.5,
                patience=self.config.get('scheduler_patience', 5),
                verbose=True,
                min_lr=1e-6
            )
        
        # Beam search decoder
        self.beam_decoder = BeamSearchDecoder(
            self.model,
            self.vocabulary,
            beam_size=self.config.get('beam_size', 5),
            max_length=self.config['max_caption_length'],
            length_penalty=self.config.get('length_penalty', 1.0)
        )
        
        # Print model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"Model type: {model_type}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Model size: {total_params * 4 / 1e6:.1f} MB")
    
    def update_scheduled_sampling(self, epoch):
        """Update scheduled sampling probability"""
        if not self.use_scheduled_sampling:
            return
        
        if hasattr(self.model, 'set_sampling_prob'):
            # Linear schedule
            progress = min(epoch / self.sampling_schedule_epochs, 1.0)
            current_prob = self.initial_sampling_prob + progress * (
                self.final_sampling_prob - self.initial_sampling_prob
            )
            
            self.model.set_sampling_prob(current_prob)
            print(f"Scheduled sampling probability: {current_prob:.3f}")
    
    def train_epoch(self):
        """Train for one epoch with advanced features"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        # Update scheduled sampling
        self.update_scheduled_sampling(self.current_epoch)
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch+1}')
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move data to device
            video_features = batch['video_features'].to(self.device)
            captions = batch['captions'].to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            if hasattr(self.model, 'forward') and 'attention' in str(type(self.model)).lower():
                # Attention model returns outputs and attention weights
                outputs, attention_weights = self.model(video_features, captions)
            else:
                # Basic model
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
                if hasattr(self.model, 'forward') and 'attention' in str(type(self.model)).lower():
                    outputs, _ = self.model(video_features, captions)
                else:
                    outputs = self.model(video_features, captions)
                
                # Calculate loss
                targets = captions[:, 1:]  # Remove BOS token
                outputs = outputs[:, :targets.size(1), :]  # Match sequence length
                
                loss = self.criterion(outputs, targets)
                
                if not torch.isnan(loss):
                    total_loss += loss.item()
        
        return total_loss / len(self.test_loader)
    
    def evaluate_bleu_score(self, num_samples=None, use_beam_search=True):
        """Evaluate BLEU score for baseline comparison"""
        self.model.eval()
        
        predictions = []
        references = []
        
        if num_samples is None:
            num_samples = len(self.test_loader.dataset)
        
        print(f"Evaluating BLEU score on {num_samples} samples...")
        
        sample_count = 0
        with torch.no_grad():
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
                    
                    if use_beam_search and hasattr(self, 'beam_decoder'):
                        # Use beam search
                        predicted_ids, _, _ = self.beam_decoder.decode(single_video)
                        predicted_caption = self.vocabulary.decode_caption(predicted_ids)
                    else:
                        # Use greedy decoding
                        if hasattr(self.model, 'forward') and 'attention' in str(type(self.model)).lower():
                            outputs, _ = self.model(single_video, max_length=self.config['max_caption_length'])
                        else:
                            outputs = self.model(single_video, max_length=self.config['max_caption_length'])
                        
                        predicted_ids = outputs.argmax(dim=-1).squeeze(0)
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
        
        return scores, predictions, references
    
    def save_checkpoint(self, epoch, bleu_score, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_bleu_score': self.best_bleu_score,
            'config': self.config,
            'vocabulary_size': len(self.vocabulary),
            'bleu_score': bleu_score
        }
        
        if hasattr(self, 'scheduler') and self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.config['save_dir'], f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.config['save_dir'], 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"New best model saved at epoch {epoch} with BLEU-1: {bleu_score:.4f}")
    
    def train(self):
        """Main training loop following baseline setup"""
        print("Starting Advanced S2VT training...")
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
            
            # Evaluate BLEU score periodically
            bleu_score = 0.0
            if (epoch + 1) % self.config.get('eval_every', 10) == 0:
                scores, _, _ = self.evaluate_bleu_score(
                    num_samples=self.config.get('eval_samples', 200),
                    use_beam_search=True
                )
                bleu_score = scores['BLEU-1']  # Use BLEU-1 for baseline comparison
                self.bleu_scores.append(bleu_score)
                
                print(f"BLEU Evaluation:")
                for metric, score in scores.items():
                    print(f"  {metric}: {score:.4f}")
                
                # Target: BLEU@1 = 0.6 (Captions Avg.)
                if bleu_score >= 0.6:
                    print(f"🎉 Baseline BLEU@1 = 0.6 reached! Current: {bleu_score:.4f}")
            
            # Update learning rate scheduler
            if hasattr(self, 'scheduler') and self.scheduler:
                self.scheduler.step(bleu_score if bleu_score > 0 else val_loss)
            
            # Check if this is the best model
            is_best = bleu_score > self.best_bleu_score
            if is_best:
                self.best_bleu_score = bleu_score
            
            # Print epoch summary
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            if bleu_score > 0:
                print(f"BLEU-1: {bleu_score:.4f} (Target: 0.60)")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # Save checkpoint
            if (epoch + 1) % self.config.get('save_every', 10) == 0:
                self.save_checkpoint(epoch + 1, bleu_score, is_best)
            
            print("-" * 40)
        
        print("\nTraining completed!")
        print(f"Best BLEU-1 score: {self.best_bleu_score:.4f}")
        
        # Final evaluation
        final_scores, predictions, references = self.evaluate_bleu_score(use_beam_search=True)
        print("\nFinal Evaluation Results:")
        for metric, score in final_scores.items():
            print(f"  {metric}: {score:.4f}")
        
        # Save final results
        results = {
            'final_scores': final_scores,
            'best_bleu_score': self.best_bleu_score,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'bleu_scores': self.bleu_scores,
            'sample_predictions': predictions[:10],
            'sample_references': references[:10]
        }
        
        results_path = os.path.join(self.config['save_dir'], 'final_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save final model
        self.save_checkpoint(self.config['num_epochs'], final_scores['BLEU-1'], is_best=False)


def get_baseline_config():
    """Get baseline configuration to reach BLEU@1 = 0.6"""
    return {
        # Data parameters
        'data_root': 'E:/imgsynth/MLDS_hw2_1_data',
        'max_frames': 80,
        'max_caption_length': 20,
        'vocab_min_count': 3,  # As per baseline requirement
        
        # Model parameters - Baseline settings
        'model_type': 'attention',  # or 'scheduled_sampling', 'basic'
        'video_feature_dim': 4096,
        'hidden_dim': 256,  # Baseline LSTM dimension = 256
        'embedding_dim': 256,
        'num_layers': 2,
        'dropout': 0.3,
        'attention_dim': 256,
        
        # Training parameters - Baseline settings
        'batch_size': 32,
        'num_epochs': 200,  # Baseline training epochs = 200
        'learning_rate': 0.001,  # Baseline learning rate = 0.001
        'weight_decay': 0.0,  # No weight decay in baseline
        'grad_clip': 5.0,
        'num_workers': 4,
        
        # Advanced training features
        'use_scheduled_sampling': True,
        'initial_sampling_prob': 0.0,
        'final_sampling_prob': 0.25,
        'sampling_schedule_epochs': 100,
        
        # Beam search parameters
        'beam_size': 5,
        'length_penalty': 1.0,
        
        # Evaluation and saving
        'save_dir': './checkpoints_baseline',
        'save_every': 10,
        'eval_every': 10,
        'eval_samples': 200,
        'use_scheduler': False,  # Baseline doesn't use scheduler
    }


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Advanced S2VT Training for Baseline')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--model_type', type=str, default='attention',
                       choices=['basic', 'attention', 'scheduled_sampling'],
                       help='Type of S2VT model to use')
    parser.add_argument('--data_root', type=str, default='E:/imgsynth/MLDS_hw2_1_data')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--num_epochs', type=int, default=None)
    parser.add_argument('--hidden_dim', type=int, default=None)
    args = parser.parse_args()
    
    # Load configuration
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = get_baseline_config()
    
    # Override with command line arguments
    if args.model_type:
        config['model_type'] = args.model_type
    if args.data_root:
        config['data_root'] = args.data_root
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.learning_rate:
        config['learning_rate'] = args.learning_rate
    if args.num_epochs:
        config['num_epochs'] = args.num_epochs
    if args.hidden_dim:
        config['hidden_dim'] = args.hidden_dim
    
    print("Baseline Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Create trainer
    trainer = AdvancedMSVDTrainer(config)
    
    # Resume from checkpoint if specified
    if args.resume and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume)
        trainer.current_epoch = checkpoint['epoch']
        trainer.best_bleu_score = checkpoint.get('best_bleu_score', 0.0)
        print(f"Resuming from epoch {trainer.current_epoch}")
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()