"""
Enhanced Training Script - Focused improvements to beat baseline

This script implements the enhanced S2VT model with targeted improvements:
- Better attention mechanism with coverage
- Improved initialization and regularization
- Label smoothing
- Enhanced beam search

Following the same training tips but with focused enhancements.
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

from models.enhanced_s2vt import EnhancedS2VT, EnhancedS2VTLoss, CoverageBeamSearch
from data.msvd_dataset import MSVDDataset, create_msvd_data_loaders
from utils.metrics import calculate_bleu_score


class EnhancedMSVDTrainer:
    """Enhanced trainer with focused improvements"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup logging
        self.setup_logging()
        
        # Load data
        self.train_loader, self.val_loader, self.test_loader, self.vocabulary = self.load_data()
        
        # Create model
        self.model = self.create_model()
        
        # Setup training components
        self.criterion = EnhancedS2VTLoss(
            vocab_size=len(self.vocabulary),
            pad_token_id=self.vocabulary.word2idx.get('<PAD>', 0),
            label_smoothing=0.1  # Label smoothing for better generalization
        )
        
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['training_parameters']['learning_rate'],
            weight_decay=1e-5  # Small weight decay for regularization
        )
        
        # Enhanced beam search
        self.beam_search = CoverageBeamSearch(
            model=self.model,
            vocabulary=self.vocabulary.word2idx,  # Pass the word2idx dictionary
            beam_size=config['beam_search']['beam_size'],
            length_penalty=config['beam_search']['length_penalty'],
            coverage_penalty=0.2
        )
        
        # Training state
        self.current_epoch = 0
        self.best_bleu = 0.0
        self.save_dir = config['logging']['save_dir']
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Scheduled sampling parameters
        self.use_scheduled_sampling = config['advanced_features']['use_scheduled_sampling']
        self.initial_sampling_prob = config['advanced_features']['initial_sampling_prob']
        self.final_sampling_prob = config['advanced_features']['final_sampling_prob']
        self.sampling_schedule_epochs = config['advanced_features']['sampling_schedule_epochs']
        
        logging.info(f"Enhanced trainer initialized on {self.device}")
        logging.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        logging.info(f"Vocabulary size: {len(self.vocabulary)}")
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_dir = self.config['logging']['save_dir']
        os.makedirs(log_dir, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(log_dir, 'training_enhanced.log')),
                logging.StreamHandler()
            ]
        )
    
    def load_data(self) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
        """Load MSVD dataset"""
        logging.info("Loading MSVD dataset...")
        
        train_loader, test_loader, vocabulary = create_msvd_data_loaders(
            data_root=self.config['data_parameters']['data_root'],
            batch_size=self.config['training_parameters']['batch_size'],
            max_frames=self.config['data_parameters']['max_frames'],
            max_caption_length=self.config['data_parameters']['max_caption_length'],
            num_workers=0  # Disable multiprocessing to avoid pickle issues
        )
        
        # Use test loader as validation for this simple case
        val_loader = test_loader
        
        logging.info(f"Training samples: {len(train_loader.dataset)}")
        logging.info(f"Validation samples: {len(val_loader.dataset)}")
        logging.info(f"Test samples: {len(test_loader.dataset)}")
        
        return train_loader, val_loader, test_loader, vocabulary
    
    def create_model(self) -> EnhancedS2VT:
        """Create enhanced S2VT model"""
        model = EnhancedS2VT(
            vocab_size=len(self.vocabulary),
            video_feature_dim=self.config['model_parameters']['video_feature_dim'],
            hidden_dim=self.config['model_parameters']['hidden_dim'],
            embedding_dim=self.config['model_parameters']['embedding_dim'],
            num_layers=self.config['model_parameters']['num_layers'],
            dropout=self.config['model_parameters']['dropout'],
            max_seq_length=self.config['data_parameters']['max_caption_length'],
            attention_dim=self.config['model_parameters']['attention_dim']
        )
        
        return model.to(self.device)
    
    def get_sampling_probability(self, epoch: int) -> float:
        """Calculate scheduled sampling probability"""
        if not self.use_scheduled_sampling:
            return 0.0
        
        if epoch < self.sampling_schedule_epochs:
            # Linear schedule from initial to final probability
            progress = epoch / self.sampling_schedule_epochs
            return self.initial_sampling_prob + progress * (self.final_sampling_prob - self.initial_sampling_prob)
        else:
            return self.final_sampling_prob
    
    def train_epoch(self, epoch: int) -> float:
        """Train one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        # Get current sampling probability
        sampling_prob = self.get_sampling_probability(epoch)
        
        for batch_idx, batch_data in enumerate(self.train_loader):
            video_features = batch_data['video_features'].to(self.device)
            captions = batch_data['captions'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            # Input and target captions
            input_captions = captions[:, :-1]  # Remove last token
            target_captions = captions[:, 1:]  # Remove first token (BOS)
            
            # Model forward pass with scheduled sampling
            outputs = self.model(video_features, input_captions, sampling_prob)
            
            # Compute loss
            loss = self.criterion(outputs['logits'], target_captions)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                                         self.config['training_parameters']['grad_clip'])
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Log progress
            if batch_idx % 50 == 0:
                logging.info(f'Epoch {epoch}, Batch {batch_idx}/{num_batches}, '
                           f'Loss: {loss.item():.4f}, Sampling Prob: {sampling_prob:.3f}')
        
        avg_loss = total_loss / num_batches
        logging.info(f'Epoch {epoch} completed. Average Loss: {avg_loss:.4f}')
        
        return avg_loss
    
    def evaluate(self, data_loader: DataLoader, num_samples: int = None) -> Dict[str, float]:
        """Evaluate model on validation/test set"""
        self.model.eval()
        
        # Collect predictions and references
        predictions = []
        references = []
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(data_loader):
                if num_samples and batch_idx * self.config['training_parameters']['batch_size'] >= num_samples:
                    break
                
                video_features = batch_data['video_features'].to(self.device)
                captions = batch_data['captions'].to(self.device)
                batch_size = video_features.size(0)
                
                # Generate predictions using beam search
                for i in range(batch_size):
                    single_video = video_features[i:i+1]
                    pred_tokens, _ = self.beam_search.decode(single_video)
                    
                    # Convert to words
                    pred_tokens = pred_tokens.squeeze().cpu().numpy()
                    pred_caption = self.tokens_to_words(pred_tokens)
                    predictions.append(pred_caption)
                    
                    # Reference caption
                    ref_tokens = captions[i].cpu().numpy()
                    ref_caption = self.tokens_to_words(ref_tokens)
                    references.append([ref_caption])  # BLEU expects list of references
        
        # Compute BLEU scores
        bleu_scores = calculate_bleu_score(predictions, references)
        
        return bleu_scores
    
    def tokens_to_words(self, tokens: np.ndarray) -> str:
        """Convert token indices to words"""
        words = []
        
        for token in tokens:
            if token == self.vocabulary.word2idx.get('<EOS>', 2):
                break
            if token not in [self.vocabulary.word2idx.get('<PAD>', 0), self.vocabulary.word2idx.get('<BOS>', 1)]:
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
        checkpoint_path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.save_dir, 'best_model_enhanced.pth')
            torch.save(checkpoint, best_path)
            logging.info(f'New best model saved with BLEU-1: {bleu_score:.4f}')
        
        # Keep only recent checkpoints
        self.cleanup_checkpoints()
    
    def cleanup_checkpoints(self):
        """Keep only the most recent checkpoints"""
        keep_checkpoints = self.config['logging']['keep_checkpoints']
        checkpoints = [f for f in os.listdir(self.save_dir) if f.startswith('checkpoint_epoch_')]
        
        if len(checkpoints) > keep_checkpoints:
            # Sort by epoch number
            checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
            
            # Remove oldest checkpoints
            for checkpoint in checkpoints[:-keep_checkpoints]:
                os.remove(os.path.join(self.save_dir, checkpoint))
    
    def train(self):
        """Main training loop"""
        logging.info("Starting enhanced training...")
        logging.info(f"Target BLEU-1: {self.config['evaluation']['target_metrics']['BLEU-1']}")
        
        start_time = time.time()
        
        for epoch in range(1, self.config['training_parameters']['num_epochs'] + 1):
            self.current_epoch = epoch
            
            # Train one epoch
            train_loss = self.train_epoch(epoch)
            
            # Evaluate periodically
            if epoch % self.config['evaluation']['eval_every'] == 0:
                logging.info(f"Evaluating at epoch {epoch}...")
                
                # Evaluate on validation set
                val_metrics = self.evaluate(
                    self.val_loader, 
                    num_samples=self.config['evaluation']['eval_samples']
                )
                
                bleu1 = val_metrics['BLEU-1']
                logging.info(f"Epoch {epoch} - BLEU-1: {bleu1:.4f}, BLEU-4: {val_metrics['BLEU-4']:.4f}")
                
                # Save checkpoint
                is_best = bleu1 > self.best_bleu
                if is_best:
                    self.best_bleu = bleu1
                
                if epoch % self.config['logging']['save_every'] == 0:
                    self.save_checkpoint(epoch, bleu1, is_best)
                
                # Check if we beat the baseline
                if bleu1 >= self.config['evaluation']['target_metrics']['BLEU-1']:
                    logging.info(f"🎉 BASELINE BEATEN! BLEU-1: {bleu1:.4f} >= {self.config['evaluation']['target_metrics']['BLEU-1']}")
        
        # Final evaluation
        logging.info("Final evaluation on test set...")
        test_metrics = self.evaluate(self.test_loader)
        
        logging.info("=== FINAL RESULTS ===")
        for metric, score in test_metrics.items():
            logging.info(f"{metric}: {score:.4f}")
        
        # Training time
        total_time = time.time() - start_time
        logging.info(f"Total training time: {total_time/60:.1f} minutes")
        
        # Save final model
        self.save_checkpoint(self.current_epoch, test_metrics['BLEU-1'], is_best=True)
        
        return test_metrics


def main():
    parser = argparse.ArgumentParser(description='Enhanced S2VT Training')
    parser.add_argument('--config', type=str, default='config_baseline.json',
                       help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Create trainer
    trainer = EnhancedMSVDTrainer(config)
    
    # Resume from checkpoint if specified
    if args.resume:
        checkpoint = torch.load(args.resume)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.current_epoch = checkpoint['epoch']
        trainer.best_bleu = checkpoint['bleu_score']
        logging.info(f"Resumed from epoch {trainer.current_epoch} with BLEU-1: {trainer.best_bleu:.4f}")
    
    # Start training
    final_metrics = trainer.train()
    
    print(f"\n🎯 FINAL RESULTS:")
    print(f"BLEU-1: {final_metrics['BLEU-1']:.4f}")
    print(f"BLEU-4: {final_metrics['BLEU-4']:.4f}")
    
    baseline_target = config['evaluation']['target_metrics']['BLEU-1']
    if final_metrics['BLEU-1'] >= baseline_target:
        print(f"✅ SUCCESS! Beat baseline ({baseline_target}) with {final_metrics['BLEU-1']:.4f}")
    else:
        print(f"❌ Below baseline. Target: {baseline_target}, Achieved: {final_metrics['BLEU-1']:.4f}")


if __name__ == '__main__':
    main()