"""
Model Seq2Seq - Enhanced S2VT Training Implementation

This file contains the main training code for the HW2 video caption generation model.
It implements the enhanced S2VT architecture with attention mechanism and scheduled sampling.

Key Components:
1. Enhanced S2VT model with coverage attention
2. Scheduled sampling for exposure bias reduction
3. Beam search decoding with coverage penalty
4. BLEU evaluation metrics
5. Model checkpointing and saving

Usage:
    python model_seq2seq.py --config config_enhanced.json
    python model_seq2seq.py --config config_baseline.json --resume checkpoint.pth
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

# Model imports
from models.enhanced_s2vt import EnhancedS2VT, EnhancedS2VTLoss, CoverageBeamSearch
from models.attention_s2vt import AttentionS2VT, ScheduledSamplingS2VT, BeamSearchDecoder

# Data imports
from data.msvd_dataset import MSVDDataset, create_msvd_data_loaders
from utils.metrics import calculate_bleu_score


class Seq2SeqModelTrainer:
    """Main trainer class for HW2 Seq2Seq model"""
    
    def __init__(self, config_path: str):
        """Initialize trainer with configuration"""
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup logging
        self.setup_logging()
        
        # Load data
        self.train_loader, self.test_loader, self.vocabulary = self.load_data()
        
        # Create model
        self.model = self.create_model()
        
        # Setup training components
        self.setup_training()
        
        # Training state
        self.current_epoch = 0
        self.best_bleu = 0.0
        self.save_dir = self.config['logging']['save_dir']
        os.makedirs(self.save_dir, exist_ok=True)
        
        logging.info(f"Trainer initialized on {self.device}")
        logging.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_dir = self.config['logging']['save_dir']
        os.makedirs(log_dir, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(log_dir, 'training.log')),
                logging.StreamHandler()
            ]
        )
    
    def load_data(self):
        """Load MSVD dataset"""
        logging.info("Loading MSVD dataset...")
        
        train_loader, test_loader, vocabulary = create_msvd_data_loaders(
            data_root=self.config['data_parameters']['data_root'],
            batch_size=self.config['training_parameters']['batch_size'],
            max_frames=self.config['data_parameters']['max_frames'],
            max_caption_length=self.config['data_parameters']['max_caption_length'],
            num_workers=0  # Disable multiprocessing for compatibility
        )
        
        logging.info(f"Training samples: {len(train_loader.dataset)}")
        logging.info(f"Test samples: {len(test_loader.dataset)}")
        
        return train_loader, test_loader, vocabulary
    
    def create_model(self):
        """Create S2VT model based on configuration"""
        model_params = self.config['model_parameters']
        data_params = self.config['data_parameters']
        
        model_type = model_params.get('model_type', 'enhanced')
        
        if model_type == 'enhanced':
            model = EnhancedS2VT(
                vocab_size=len(self.vocabulary),
                video_feature_dim=model_params['video_feature_dim'],
                hidden_dim=model_params['hidden_dim'],
                embedding_dim=model_params['embedding_dim'],
                num_layers=model_params['num_layers'],
                dropout=model_params['dropout'],
                max_seq_length=data_params['max_caption_length'],
                attention_dim=model_params['attention_dim']
            )
        else:
            # Baseline attention model
            model = AttentionS2VT(
                vocab_size=len(self.vocabulary),
                video_feature_dim=model_params['video_feature_dim'],
                hidden_dim=model_params['hidden_dim'],
                embedding_dim=model_params['embedding_dim'],
                num_layers=model_params['num_layers'],
                dropout=model_params['dropout'],
                max_seq_length=data_params['max_caption_length'],
                attention_dim=model_params['attention_dim']
            )
        
        return model.to(self.device)
    
    def setup_training(self):
        """Setup training components"""
        
        # Loss function
        if hasattr(self.model, '__class__') and 'Enhanced' in self.model.__class__.__name__:
            self.criterion = EnhancedS2VTLoss(
                vocab_size=len(self.vocabulary),
                pad_token_id=self.vocabulary.word2idx.get('<PAD>', 0),
                label_smoothing=0.1
            )
        else:
            from models.attention_s2vt import S2VTLoss
            self.criterion = S2VTLoss(
                vocab_size=len(self.vocabulary),
                pad_token_id=self.vocabulary.word2idx.get('<PAD>', 0)
            )
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['training_parameters']['learning_rate'],
            weight_decay=self.config['training_parameters'].get('weight_decay', 0)
        )
        
        # Beam search decoder
        vocab_dict = self.vocabulary.word2idx
        
        if hasattr(self.model, '__class__') and 'Enhanced' in self.model.__class__.__name__:
            self.beam_search = CoverageBeamSearch(
                model=self.model,
                vocabulary=vocab_dict,
                beam_size=self.config['beam_search']['beam_size'],
                length_penalty=self.config['beam_search']['length_penalty'],
                coverage_penalty=0.2
            )
        else:
            self.beam_search = BeamSearchDecoder(
                model=self.model,
                vocabulary=vocab_dict,
                beam_size=self.config['beam_search']['beam_size'],
                length_penalty=self.config['beam_search']['length_penalty']
            )
    
    def get_sampling_probability(self, epoch: int) -> float:
        """Calculate scheduled sampling probability"""
        if not self.config['advanced_features']['use_scheduled_sampling']:
            return 0.0
        
        schedule_epochs = self.config['advanced_features']['sampling_schedule_epochs']
        initial_prob = self.config['advanced_features']['initial_sampling_prob']
        final_prob = self.config['advanced_features']['final_sampling_prob']
        
        if epoch < schedule_epochs:
            progress = epoch / schedule_epochs
            return initial_prob + progress * (final_prob - initial_prob)
        else:
            return final_prob
    
    def train_epoch(self, epoch: int) -> float:
        """Train one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        sampling_prob = self.get_sampling_probability(epoch)
        
        for batch_idx, batch_data in enumerate(self.train_loader):
            video_features = batch_data['video_features'].to(self.device)
            captions = batch_data['captions'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Prepare input and target
            input_captions = captions[:, :-1]
            target_captions = captions[:, 1:]
            
            # Forward pass
            if hasattr(self.model, '__class__') and 'Enhanced' in self.model.__class__.__name__:
                outputs = self.model(video_features, input_captions, sampling_prob)
                loss = self.criterion(outputs['logits'], target_captions)
            else:
                outputs = self.model(video_features, input_captions, sampling_prob)
                loss = self.criterion(outputs, target_captions)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                                         self.config['training_parameters']['grad_clip'])
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 20 == 0:
                logging.info(f'Epoch {epoch}, Batch {batch_idx}/{num_batches}, '
                           f'Loss: {loss.item():.4f}, Sampling: {sampling_prob:.3f}')
        
        return total_loss / num_batches
    
    def evaluate(self, num_samples: int = 100) -> Dict[str, float]:
        """Evaluate model"""
        self.model.eval()
        
        predictions = []
        references = []
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(self.test_loader):
                if batch_idx * self.config['training_parameters']['batch_size'] >= num_samples:
                    break
                
                video_features = batch_data['video_features'].to(self.device)
                captions = batch_data['captions'].to(self.device)
                batch_size = video_features.size(0)
                
                for i in range(batch_size):
                    single_video = video_features[i:i+1]
                    
                    # Generate prediction
                    if hasattr(self.beam_search, 'decode'):
                        pred_tokens, _ = self.beam_search.decode(single_video)
                    else:
                        pred_tokens, _, _ = self.beam_search.decode(single_video)
                    
                    # Convert to words
                    pred_caption = self.tokens_to_words(pred_tokens.squeeze().cpu().numpy())
                    predictions.append(pred_caption)
                    
                    # Reference caption
                    ref_caption = self.tokens_to_words(captions[i].cpu().numpy())
                    references.append([ref_caption])
        
        # Calculate BLEU scores
        return calculate_bleu_score(predictions, references)
    
    def tokens_to_words(self, tokens: np.ndarray) -> str:
        """Convert tokens to words"""
        words = []
        for token in tokens:
            if token == self.vocabulary.word2idx.get('<EOS>', 2):
                break
            if token not in [self.vocabulary.word2idx.get('<PAD>', 0), 
                           self.vocabulary.word2idx.get('<BOS>', 1)]:
                word = self.vocabulary.idx2word.get(token, '<UNK>')
                words.append(word)
        return ' '.join(words)
    
    def save_checkpoint(self, epoch: int, bleu_score: float, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'bleu_score': bleu_score,
            'vocabulary': self.vocabulary,
            'config': self.config
        }
        
        # Save regular checkpoint
        if epoch % self.config['logging']['save_every'] == 0:
            checkpoint_path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch}.pth')
            torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.save_dir, 'best_model_enhanced.pth')
            torch.save(checkpoint, best_path)
            logging.info(f'New best model saved with BLEU-1: {bleu_score:.4f}')
    
    def train(self):
        """Main training loop"""
        logging.info("Starting training...")
        logging.info(f"Target BLEU-1: {self.config['evaluation']['target_metrics']['BLEU-1']}")
        
        start_time = time.time()
        
        for epoch in range(1, self.config['training_parameters']['num_epochs'] + 1):
            self.current_epoch = epoch
            
            # Train epoch
            train_loss = self.train_epoch(epoch)
            
            # Evaluate
            if epoch % self.config['evaluation']['eval_every'] == 0:
                metrics = self.evaluate(self.config['evaluation']['eval_samples'])
                bleu1 = metrics['BLEU-1']
                
                logging.info(f"Epoch {epoch} - Loss: {train_loss:.4f}, BLEU-1: {bleu1:.4f}")
                
                # Save checkpoint
                is_best = bleu1 > self.best_bleu
                if is_best:
                    self.best_bleu = bleu1
                
                self.save_checkpoint(epoch, bleu1, is_best)
                
                # Check if target reached
                target_bleu = self.config['evaluation']['target_metrics']['BLEU-1']
                if bleu1 >= target_bleu:
                    logging.info(f"🎉 Target BLEU-1 reached: {bleu1:.4f} >= {target_bleu}")
        
        total_time = time.time() - start_time
        logging.info(f"Training completed in {total_time/60:.1f} minutes")
        
        return {'BLEU-1': self.best_bleu}


def main():
    parser = argparse.ArgumentParser(description='HW2 Seq2Seq Model Training')
    parser.add_argument('--config', type=str, default='config_enhanced.json',
                       help='Configuration file path')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = Seq2SeqModelTrainer(args.config)
    
    # Resume if specified
    if args.resume:
        checkpoint = torch.load(args.resume)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.current_epoch = checkpoint['epoch']
        trainer.best_bleu = checkpoint['bleu_score']
        logging.info(f"Resumed from epoch {trainer.current_epoch}")
    
    # Start training
    final_metrics = trainer.train()
    
    print(f"\n🎯 FINAL RESULTS:")
    print(f"Best BLEU-1: {final_metrics['BLEU-1']:.4f}")


if __name__ == '__main__':
    main()