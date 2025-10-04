"""
HW2 Model Seq2Seq - Enhanced S2VT Training Implementation

This is the main training script for HW2 video caption generation targeting BLEU@1 = 0.6 baseline.
It implements Enhanced S2VT (Sequence to Sequence - Video to Text) with comprehensive improvements.

Key Features:
- Enhanced S2VT architecture with coverage attention (512-dim hidden states)
- Scheduled sampling with exponential schedule (0.0 → 0.15)
- Coverage beam search with length and coverage penalties
- Training tips BLEU@1 evaluation (exact formula from slides)
- Label smoothing regularization (factor=0.05)
- Advanced learning rate scheduling with early stopping
- Two-layer LSTM structure (num_layers=2) as per training tips

Training Tips Compliance:
✅ Two-layer LSTM encoder-decoder
✅ Attention mechanism with coverage
✅ Scheduled sampling for exposure bias
✅ Beam search decoding
✅ BLEU@1 evaluation using exact formula: BLEU@1 = BP × Precision

Usage:
    python model_seq2seq.py --config config_enhanced.json --train
    python model_seq2seq.py --train --data_path /path/to/MLDS_hw2_1_data
    python model_seq2seq.py --config config.json
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
    from models.enhanced_s2vt import EnhancedS2VT, EnhancedS2VTLoss, CoverageBeamSearch
    from data.msvd_dataset import create_msvd_data_loaders
    from utils.metrics import calculate_training_tips_bleu
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
        
        # LSTM layers (2-layer as per training tips)
        self.encoder_lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.2)
        self.decoder_lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.2)
        
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
        
        # Enhanced default configuration matching latest training
        self.config = {
            'model_parameters': {
                'video_feature_dim': 4096,
                'hidden_dim': 512,
                'embedding_dim': 512,
                'num_layers': 2,
                'dropout': 0.2,
                'attention_dim': 512
            },
            'training_parameters': {
                'batch_size': 32,
                'num_epochs': 200,
                'learning_rate': 0.0001,
                'weight_decay': 1e-4,
                'grad_clip': 5.0,
                'use_scheduler': True
            },
            'data_parameters': {
                'max_frames': 80,
                'max_caption_length': 20
            },
            'advanced_features': {
                'use_scheduled_sampling': True,
                'initial_sampling_prob': 0.0,
                'final_sampling_prob': 0.15,
                'sampling_schedule_epochs': 150,
                'label_smoothing': 0.05
            },
            'beam_search': {
                'beam_size': 5,
                'length_penalty': 1.0,
                'coverage_penalty': 0.2
            },
            'evaluation': {
                'target_metrics': {
                    'BLEU-1': 0.6
                }
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
                max_seq_length=self.config['data_parameters']['max_caption_length'],
                attention_dim=self.config['model_parameters']['attention_dim']
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
        
        # Setup enhanced training components
        if ENHANCED_AVAILABLE:
            criterion = EnhancedS2VTLoss(
                vocab_size=len(vocabulary),
                ignore_index=0,
                label_smoothing=self.config['advanced_features']['label_smoothing']
            )
            # Initialize beam search for evaluation
            beam_search = CoverageBeamSearch(
                model=model,
                vocabulary=vocabulary,
                beam_size=self.config['beam_search']['beam_size'],
                length_penalty=self.config['beam_search']['length_penalty'],
                coverage_penalty=self.config['beam_search']['coverage_penalty']
            )
        else:
            criterion = nn.CrossEntropyLoss(ignore_index=0)
            beam_search = None
        
        # Enhanced optimizer with weight decay
        optimizer = optim.Adam(
            model.parameters(), 
            lr=self.config['training_parameters']['learning_rate'],
            weight_decay=self.config['training_parameters'].get('weight_decay', 1e-4)
        )
        
        # Learning rate scheduler
        scheduler = None
        if self.config['training_parameters'].get('use_scheduler', False):
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', patience=10, factor=0.5, verbose=True
            )
        
        best_loss = float('inf')
        best_bleu = 0.0
        epochs_without_improvement = 0
        patience = 20  # Early stopping patience
        
        # Enhanced training loop with scheduled sampling and evaluation
        for epoch in range(1, self.config['training_parameters']['num_epochs'] + 1):
            model.train()
            total_loss = 0
            
            # Calculate scheduled sampling probability
            sampling_prob = self.calculate_sampling_prob(epoch)
            
            for batch_idx, batch_data in enumerate(train_loader):
                if isinstance(batch_data, dict):
                    video_features = batch_data['video_features'].to(self.device)
                    captions = batch_data['captions'].to(self.device)
                else:
                    video_features, captions = batch_data
                    video_features = video_features.to(self.device)
                    captions = captions.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass with scheduled sampling
                input_captions = captions[:, :-1]
                target_captions = captions[:, 1:]
                
                if ENHANCED_AVAILABLE:
                    outputs = model(video_features, input_captions, sampling_prob=sampling_prob)
                    if isinstance(outputs, dict):
                        logits = outputs['logits']
                    else:
                        logits = outputs
                    # Enhanced loss with label smoothing
                    loss = criterion(logits, target_captions)
                else:
                    logits = model(video_features, input_captions, sampling_prob=sampling_prob)
                    loss = criterion(logits.reshape(-1, logits.size(-1)), target_captions.reshape(-1))
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config['training_parameters']['grad_clip'])
                optimizer.step()
                
                total_loss += loss.item()
                
                if batch_idx % 10 == 0:
                    logging.info(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}, Sampling Prob: {sampling_prob:.3f}')
            
            avg_loss = total_loss / len(train_loader)
            logging.info(f'Epoch {epoch} completed. Average Loss: {avg_loss:.4f}')
            
            # Periodic evaluation with BLEU@1
            if epoch % 10 == 0 and ENHANCED_AVAILABLE and beam_search:
                logging.info(f"Evaluating at epoch {epoch}...")
                bleu_scores = self.evaluate_model(model, test_loader, beam_search, vocabulary)
                bleu1 = bleu_scores.get('BLEU-1', 0.0)
                precision = bleu_scores.get('precision', 0.0)
                bp = bleu_scores.get('brevity_penalty', 0.0)
                
                logging.info(f"Epoch {epoch} - BLEU@1: {bleu1:.4f} (target: >0.6) | Precision: {precision:.4f} | BP: {bp:.4f}")
                
                # Check for improvement
                is_best = bleu1 > best_bleu
                if is_best:
                    best_bleu = bleu1
                    epochs_without_improvement = 0
                    logging.info(f"*** New best BLEU@1: {bleu1:.4f} ***")
                    self.save_model(model, vocabulary, epoch, avg_loss, bleu1)
                else:
                    epochs_without_improvement += 1
                
                # Early stopping check
                if epochs_without_improvement >= patience:
                    logging.info(f"Early stopping triggered after {patience} epochs without improvement")
                    break
            else:
                # Save based on loss for non-evaluation epochs
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    self.save_model(model, vocabulary, epoch, avg_loss)
            
            # Learning rate scheduling
            if scheduler:
                scheduler.step(avg_loss)
        
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
    
    def calculate_sampling_prob(self, epoch):
        """Calculate scheduled sampling probability"""
        if not self.config['advanced_features']['use_scheduled_sampling']:
            return 0.0
        
        schedule_epochs = self.config['advanced_features']['sampling_schedule_epochs']
        final_prob = self.config['advanced_features']['final_sampling_prob']
        
        # Exponential schedule
        prob = final_prob * (1 - np.exp(-epoch / schedule_epochs))
        return min(prob, final_prob)
    
    def evaluate_model(self, model, test_loader, beam_search, vocabulary):
        """Evaluate model using training tips BLEU@1"""
        model.eval()
        predictions = []
        references = []
        
        with torch.no_grad():
            for batch_data in test_loader:
                if isinstance(batch_data, dict):
                    video_features = batch_data['video_features'].to(self.device)
                    captions = batch_data['captions']
                else:
                    video_features, captions = batch_data
                    video_features = video_features.to(self.device)
                
                # Generate predictions using beam search
                for i in range(video_features.size(0)):
                    single_video = video_features[i:i+1]
                    pred_tokens, _ = beam_search.decode(single_video)
                    
                    # Convert to words
                    pred_words = [vocabulary.idx2word.get(idx, '<UNK>') for idx in pred_tokens[0]]
                    pred_words = [w for w in pred_words if w not in ['<PAD>', '<BOS>', '<EOS>']]
                    
                    predictions.append(pred_words)
                    
                    # Reference caption
                    if isinstance(captions, torch.Tensor):
                        ref_tokens = captions[i].tolist()
                    else:
                        ref_tokens = captions[i]
                    ref_words = [vocabulary.idx2word.get(idx, '<UNK>') for idx in ref_tokens]
                    ref_words = [w for w in ref_words if w not in ['<PAD>', '<BOS>', '<EOS>']]
                    
                    references.append([ref_words])  # BLEU expects list of references
        
        # Calculate BLEU using training tips formula
        bleu_scores = calculate_training_tips_bleu(predictions, references)
        return bleu_scores
    
    def save_model(self, model, vocabulary, epoch, loss, bleu1=None):
        """Save model checkpoint with enhanced information"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'vocabulary': vocabulary,
            'config': self.config,
            'loss': loss,
            'bleu1': bleu1 if bleu1 is not None else 0.0
        }
        
        # Create model directory
        model_dir = 'your_seq2seq_model'
        os.makedirs(model_dir, exist_ok=True)
        
        # Save checkpoint
        torch.save(checkpoint, os.path.join(model_dir, 'best_model_enhanced.pth'))
        
        if bleu1 is not None:
            logging.info(f'Model saved at epoch {epoch} with loss {loss:.4f}, BLEU@1: {bleu1:.4f}')
        else:
            logging.info(f'Model saved at epoch {epoch} with loss {loss:.4f}')


def main():
    parser = argparse.ArgumentParser(description='HW2 Enhanced S2VT Model Training')
    parser.add_argument('--config', type=str, default='../config_enhanced.json', 
                       help='Config file path (default: ../config_enhanced.json)')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--data_path', type=str, default='E:/imgsynth/MLDS_hw2_1_data',
                       help='Path to training data (default: E:/imgsynth/MLDS_hw2_1_data)')
    
    args = parser.parse_args()
    
    # Create enhanced trainer
    trainer = HW2Trainer(args.config)
    
    if args.train or args.data_path:
        # Enhanced training mode
        print("=== HW2 Enhanced S2VT Training ===")
        print("Target: BLEU@1 > 0.6 using training tips evaluation")
        print("Features: Coverage attention, scheduled sampling, beam search")
        print("Architecture: Two-layer LSTM with 512-dim hidden states")
        print()
        
        data_path = args.data_path
        model, vocabulary = trainer.train_model(data_path)
        print("\n=== Training Completed! ===")
        print("Model saved to your_seq2seq_model/best_model_enhanced.pth")
        print("Ready for inference with hw2_seq2seq.sh script")
    else:
        # Demo mode
        print("HW2 Enhanced S2VT Model - Training Tips Compliant")
        print("=" * 50)
        print("Features:")
        print("  ✅ Two-layer LSTM structure (num_layers=2)")
        print("  ✅ Coverage attention mechanism")
        print("  ✅ Scheduled sampling (0.0 → 0.15)")
        print("  ✅ Enhanced beam search with coverage penalty")
        print("  ✅ Training tips BLEU@1 evaluation")
        print("  ✅ Label smoothing regularization")
        print()
        print("Usage:")
        print("  python model_seq2seq.py --train")
        print("  python model_seq2seq.py --config config_enhanced.json --train")
        print("  python model_seq2seq.py --train --data_path /path/to/MLDS_hw2_1_data")


if __name__ == '__main__':
    main()
