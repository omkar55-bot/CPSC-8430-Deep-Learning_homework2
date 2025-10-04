"""
Enhanced S2VT Model - Focused improvements to beat BLEU@1=0.6 baseline

This module implements targeted enhancements following the training tips:
1. Improved attention mechanism (from slides)
2. Better scheduled sampling strategy 
3. Enhanced beam search with coverage
4. Label smoothing for better generalization
5. Improved initialization and regularization

All changes are minimal and focused on the specific tips provided.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from typing import Optional, Tuple, Dict, List


class ImprovedGlobalAttention(nn.Module):
    """Enhanced global attention with coverage mechanism"""
    
    def __init__(self, hidden_dim: int, coverage_dim: int = 64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.coverage_dim = coverage_dim
        
        # Standard attention components
        self.W_h = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_s = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Parameter(torch.randn(hidden_dim))
        
        # Coverage mechanism (from training tips)
        self.W_c = nn.Linear(coverage_dim, hidden_dim, bias=False)
        self.coverage_conv = nn.Conv1d(1, coverage_dim, kernel_size=3, padding=1)
        
        # Improved initialization
        self.init_weights()
    
    def init_weights(self):
        """Better weight initialization following best practices"""
        nn.init.xavier_uniform_(self.W_h.weight)
        nn.init.xavier_uniform_(self.W_s.weight)
        nn.init.xavier_uniform_(self.W_c.weight)
        nn.init.normal_(self.v, 0, 0.1)
    
    def forward(self, decoder_hidden: torch.Tensor, encoder_outputs: torch.Tensor,
                coverage: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len, hidden_dim = encoder_outputs.shape
        
        # Project encoder outputs and decoder hidden state
        encoder_proj = self.W_h(encoder_outputs)  # [B, T, H]
        decoder_proj = self.W_s(decoder_hidden).unsqueeze(1)  # [B, 1, H]
        
        # Coverage mechanism
        if coverage is None:
            coverage = torch.zeros(batch_size, seq_len, device=encoder_outputs.device)
        elif coverage.dim() == 1:
            coverage = coverage.unsqueeze(0)  # [1, T] -> [1, T]
        
        # Apply coverage
        coverage_input = coverage.unsqueeze(1)  # [B, 1, T]
        coverage_features = self.coverage_conv(coverage_input).transpose(1, 2)  # [B, T, C]
        coverage_proj = self.W_c(coverage_features)  # [B, T, H]
        
        # Attention computation with coverage
        energy = torch.tanh(encoder_proj + decoder_proj + coverage_proj)  # [B, T, H]
        attention_scores = torch.matmul(energy, self.v)  # [B, T]
        
        # Softmax attention weights
        attention_weights = F.softmax(attention_scores, dim=1)  # [B, T]
        
        # Weighted context vector
        if attention_weights.dim() == 1:
            attention_weights = attention_weights.unsqueeze(0)  # [1, T]
        context_vector = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)  # [B, H]
        
        # Update coverage
        new_coverage = coverage + attention_weights
        
        return context_vector, attention_weights, new_coverage


class EnhancedS2VT(nn.Module):
    """Enhanced S2VT with focused improvements to beat baseline"""
    
    def __init__(
        self,
        vocab_size: int,
        video_feature_dim: int = 4096,
        hidden_dim: int = 256,
        embedding_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        max_seq_length: int = 20,
        attention_dim: int = 256
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.max_seq_length = max_seq_length
        
        # Video feature processing (following original S2VT)
        self.video_projection = nn.Linear(video_feature_dim, hidden_dim)
        
        # Encoder LSTM (Stage 1 - following S2VT architecture)
        self.encoder_lstm = nn.LSTM(
            hidden_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout
        )
        
        # Word embeddings with better initialization
        self.word_embedding = nn.Embedding(vocab_size, embedding_dim)
        nn.init.uniform_(self.word_embedding.weight, -0.1, 0.1)
        
        # Decoder LSTM (Stage 2)
        self.decoder_lstm = nn.LSTM(
            embedding_dim + hidden_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout
        )
        
        # Enhanced attention mechanism
        self.attention = ImprovedGlobalAttention(hidden_dim)
        
        # Output projection with layer normalization for stability
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, vocab_size)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Initialize model weights
        self.init_weights()
    
    def init_weights(self):
        """Initialize weights following best practices"""
        for name, param in self.named_parameters():
            if 'weight' in name and len(param.shape) >= 2:
                if 'lstm' in name:
                    # LSTM weights initialization
                    nn.init.orthogonal_(param)
                else:
                    # Linear layer weights
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
                if 'lstm' in name:
                    # Initialize forget gate bias to 1 (common practice)
                    n = param.size(0)
                    start, end = n // 4, n // 2
                    param.data[start:end].fill_(1.0)
    
    def forward(
        self, 
        video_features: torch.Tensor, 
        captions: Optional[torch.Tensor] = None,
        sampling_prob: float = 0.0
    ) -> Dict[str, torch.Tensor]:
        
        if captions is not None:
            return self._forward_training(video_features, captions, sampling_prob)
        else:
            return self._forward_inference(video_features)
    
    def _forward_training(
        self, 
        video_features: torch.Tensor, 
        captions: torch.Tensor,
        sampling_prob: float
    ) -> Dict[str, torch.Tensor]:
        batch_size, video_length, _ = video_features.shape
        caption_length = captions.size(1)
        
        # Stage 1: Encode video features
        video_proj = self.video_projection(video_features)
        encoder_outputs, (encoder_h, encoder_c) = self.encoder_lstm(video_proj)
        
        # Stage 2: Decode with attention
        decoder_hidden = (encoder_h, encoder_c)
        
        outputs = []
        attention_weights = []
        coverage = None
        
        for t in range(caption_length):
            # Determine input token (scheduled sampling)
            if t == 0:
                input_token = captions[:, t]
            else:
                if random.random() < sampling_prob:
                    # Use model's previous prediction
                    input_token = torch.argmax(outputs[-1], dim=-1)
                else:
                    # Use ground truth
                    input_token = captions[:, t]
            
            # Word embedding
            embedded = self.word_embedding(input_token.unsqueeze(1))
            embedded = self.dropout(embedded)
            
            # Attention mechanism
            decoder_state = decoder_hidden[0][-1]  # Use last layer hidden state
            context_vector, attn_weights, coverage = self.attention(
                decoder_state, encoder_outputs, coverage
            )
            attention_weights.append(attn_weights)
            
            # Concatenate embedding with context
            decoder_input = torch.cat([embedded, context_vector.unsqueeze(1)], dim=-1)
            
            # Decoder LSTM step
            decoder_output, decoder_hidden = self.decoder_lstm(decoder_input, decoder_hidden)
            
            # Output projection with normalization
            normalized_output = self.output_norm(decoder_output.squeeze(1))
            output = self.output_projection(normalized_output)
            outputs.append(output)
        
        # Stack outputs
        outputs = torch.stack(outputs, dim=1)
        
        return {
            'logits': outputs,
            'attention_weights': attention_weights
        }
    
    def _forward_inference(self, video_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size = video_features.size(0)
        
        # Stage 1: Encode video features
        video_proj = self.video_projection(video_features)
        encoder_outputs, (encoder_h, encoder_c) = self.encoder_lstm(video_proj)
        
        # Stage 2: Decode with attention
        decoder_hidden = (encoder_h, encoder_c)
        
        # Start with BOS token
        input_token = torch.ones(batch_size, dtype=torch.long, device=video_features.device)
        
        outputs = []
        attention_weights = []
        coverage = None
        
        for t in range(self.max_seq_length):
            # Word embedding
            embedded = self.word_embedding(input_token.unsqueeze(1))
            embedded = self.dropout(embedded)
            
            # Attention mechanism
            decoder_state = decoder_hidden[0][-1]
            context_vector, attn_weights, coverage = self.attention(
                decoder_state, encoder_outputs, coverage
            )
            attention_weights.append(attn_weights)
            
            # Concatenate embedding with context
            decoder_input = torch.cat([embedded, context_vector.unsqueeze(1)], dim=-1)
            
            # Decoder LSTM step
            decoder_output, decoder_hidden = self.decoder_lstm(decoder_input, decoder_hidden)
            
            # Output projection
            normalized_output = self.output_norm(decoder_output.squeeze(1))
            output = self.output_projection(normalized_output)
            outputs.append(output)
            
            # Next input token
            input_token = torch.argmax(output, dim=-1)
            
            # Check for EOS
            if (input_token == 2).all():  # EOS token
                break
        
        outputs = torch.stack(outputs, dim=1)
        
        return {
            'logits': outputs,
            'attention_weights': attention_weights
        }


class CoverageBeamSearch:
    """Enhanced beam search with coverage mechanism"""
    
    def __init__(
        self,
        model: EnhancedS2VT,
        vocabulary: Dict[str, int],
        beam_size: int = 5,
        length_penalty: float = 1.0,
        coverage_penalty: float = 0.2
    ):
        self.model = model
        self.vocabulary = vocabulary
        self.reverse_vocab = {v: k for k, v in vocabulary.items()}
        self.beam_size = beam_size
        self.length_penalty = length_penalty
        self.coverage_penalty = coverage_penalty
        
        self.bos_token = vocabulary.get('<BOS>', 1)
        self.eos_token = vocabulary.get('<EOS>', 2)
        self.pad_token = vocabulary.get('<PAD>', 0)
    
    def decode(self, video_features: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """Beam search with coverage penalty"""
        self.model.eval()
        
        with torch.no_grad():
            batch_size = video_features.size(0)
            
            # Encode video
            video_proj = self.model.video_projection(video_features)
            encoder_outputs, (encoder_h, encoder_c) = self.model.encoder_lstm(video_proj)
            
            # Initialize beams
            beams = [{
                'tokens': [self.bos_token],
                'score': 0.0,
                'hidden': (encoder_h, encoder_c),
                'coverage': None
            } for _ in range(self.beam_size)]
            
            completed_sequences = []
            
            for step in range(self.model.max_seq_length):
                all_candidates = []
                
                for beam in beams:
                    if beam['tokens'][-1] == self.eos_token:
                        completed_sequences.append(beam)
                        continue
                    
                    # Current token
                    current_token = torch.tensor([beam['tokens'][-1]], device=video_features.device)
                    embedded = self.model.word_embedding(current_token.unsqueeze(0))
                    
                    # Attention
                    decoder_state = beam['hidden'][0][-1]  # [1, H]
                    context_vector, attn_weights, new_coverage = self.model.attention(
                        decoder_state, encoder_outputs, beam['coverage']
                    )
                    # Ensure correct dimensions
                    if context_vector.dim() == 1:
                        context_vector = context_vector.unsqueeze(0)  # [1, H]
                    
                    # Decoder step
                    decoder_input = torch.cat([embedded, context_vector.unsqueeze(1)], dim=-1)
                    decoder_output, new_hidden = self.model.decoder_lstm(decoder_input, beam['hidden'])
                    
                    # Output projection
                    normalized_output = self.model.output_norm(decoder_output.squeeze(1))
                    logits = self.model.output_projection(normalized_output)
                    log_probs = F.log_softmax(logits, dim=-1).squeeze()
                    
                    # Coverage penalty
                    if beam['coverage'] is not None:
                        coverage_loss = self.coverage_penalty * torch.sum(torch.min(new_coverage, beam['coverage']))
                    else:
                        coverage_loss = torch.tensor(0.0, device=video_features.device)
                    
                    # Top-k candidates
                    top_k_scores, top_k_indices = torch.topk(log_probs, self.beam_size)
                    
                    for k in range(self.beam_size):
                        token_id = top_k_indices[k].item()
                        token_score = top_k_scores[k].item() - coverage_loss.item()
                        
                        candidate = {
                            'tokens': beam['tokens'] + [token_id],
                            'score': beam['score'] + token_score,
                            'hidden': new_hidden,
                            'coverage': new_coverage
                        }
                        all_candidates.append(candidate)
                
                # Select top beams
                all_candidates.sort(key=lambda x: x['score'], reverse=True)
                beams = all_candidates[:self.beam_size]
                
                if not beams:
                    break
            
            # Add remaining beams to completed
            completed_sequences.extend(beams)
            
            if not completed_sequences:
                return torch.tensor([[self.bos_token, self.eos_token]]), 0.0
            
            # Select best sequence with length penalty
            best_sequence = max(completed_sequences, key=lambda x: self._score_sequence(x))
            
            tokens = torch.tensor([best_sequence['tokens']], device=video_features.device)
            score = best_sequence['score']
            
            return tokens, score
    
    def _score_sequence(self, sequence: Dict) -> float:
        """Score sequence with length penalty"""
        length = len(sequence['tokens'])
        if self.length_penalty == 0:
            return sequence['score']
        return sequence['score'] / (length ** self.length_penalty)


class EnhancedS2VTLoss(nn.Module):
    """Enhanced loss with label smoothing"""
    
    def __init__(self, vocab_size: int, pad_token_id: int = 0, label_smoothing: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.label_smoothing = label_smoothing
        
        # Cross-entropy with label smoothing
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=pad_token_id,
            label_smoothing=label_smoothing
        )
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Reshape for loss computation
        logits = logits.reshape(-1, self.vocab_size)
        targets = targets.reshape(-1)
        
        return self.criterion(logits, targets)