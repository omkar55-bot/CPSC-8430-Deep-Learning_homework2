"""
Enhanced S2VT Model with Attention Mechanism
Implements attention on encoder hidden states as shown in training tips
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional

from .s2vt_model import S2VTModel


class AttentionS2VT(nn.Module):
    """
    S2VT Model with Attention Mechanism
    
    Implements attention on encoder hidden states to allow model to peek at 
    different sections of inputs at each decoding time step
    """
    
    def __init__(self, 
                 vocab_size,
                 max_frames,
                 video_feature_dim=4096,
                 hidden_dim=512,
                 embedding_dim=512,
                 num_layers=2,
                 dropout=0.5,
                 attention_dim=256):
        """
        Initialize Attention S2VT model
        
        Args:
            vocab_size: Size of vocabulary
            max_frames: Maximum number of video frames
            video_feature_dim: Dimension of video features
            hidden_dim: Hidden dimension of LSTM
            embedding_dim: Word embedding dimension
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            attention_dim: Attention mechanism dimension
        """
        super(AttentionS2VT, self).__init__()
        
        self.vocab_size = vocab_size
        self.max_frames = max_frames
        self.video_feature_dim = video_feature_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.attention_dim = attention_dim
        
        # Video feature projection
        self.video_feature_proj = nn.Linear(video_feature_dim, hidden_dim)
        
        # Word embedding layer
        self.word_embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Encoder LSTM
        self.encoder_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Decoder LSTM
        self.decoder_lstm = nn.LSTM(
            input_size=embedding_dim + hidden_dim,  # Word embedding + context vector
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Attention mechanism
        self.attention = GlobalAttention(hidden_dim, attention_dim)
        
        # Output projection layer
        self.output_proj = nn.Linear(hidden_dim, vocab_size)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        # Initialize LSTM weights
        for lstm in [self.encoder_lstm, self.decoder_lstm]:
            for name, param in lstm.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)
                    # Set forget gate bias to 1
                    n = param.size(0)
                    param.data[(n//4):(n//2)].fill_(1)
        
        # Initialize linear layers
        nn.init.xavier_uniform_(self.video_feature_proj.weight)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.video_feature_proj.bias, 0)
        nn.init.constant_(self.output_proj.bias, 0)
        
        # Initialize word embeddings
        nn.init.uniform_(self.word_embedding.weight, -0.1, 0.1)
    
    def init_hidden(self, batch_size, device):
        """Initialize hidden and cell states"""
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        return (h0, c0)
    
    def forward(self, video_features, captions=None, max_length=20, use_scheduled_sampling=False, sampling_prob=0.0):
        """
        Forward pass with attention mechanism
        
        Args:
            video_features: (batch_size, num_frames, feature_dim)
            captions: (batch_size, caption_length) - for training
            max_length: Maximum caption length for inference
            use_scheduled_sampling: Whether to use scheduled sampling
            sampling_prob: Probability of using model's prediction instead of ground truth
            
        Returns:
            outputs: (batch_size, sequence_length, vocab_size)
            attention_weights: (batch_size, sequence_length, num_frames)
        """
        batch_size = video_features.size(0)
        num_frames = video_features.size(1)
        device = video_features.device
        
        # Encode video features
        encoder_outputs, encoder_hiddens = self._encode_video(video_features)
        
        if self.training and captions is not None:
            return self._forward_train(encoder_outputs, encoder_hiddens, captions, 
                                     use_scheduled_sampling, sampling_prob)
        else:
            return self._forward_inference(encoder_outputs, encoder_hiddens, max_length)
    
    def _encode_video(self, video_features):
        """Encode video features through encoder LSTM"""
        batch_size = video_features.size(0)
        device = video_features.device
        
        # Project video features
        video_features = self.video_feature_proj(video_features)
        video_features = self.dropout(video_features)
        
        # Initialize encoder hidden state
        encoder_hidden = self.init_hidden(batch_size, device)
        
        # Pass through encoder LSTM
        encoder_outputs, encoder_final = self.encoder_lstm(video_features, encoder_hidden)
        
        return encoder_outputs, encoder_final
    
    def _forward_train(self, encoder_outputs, encoder_hiddens, captions, 
                      use_scheduled_sampling=False, sampling_prob=0.0):
        """Training forward pass with attention"""
        batch_size = encoder_outputs.size(0)
        caption_length = captions.size(1)
        device = encoder_outputs.device
        
        # Initialize decoder hidden state with encoder final state
        decoder_hidden = encoder_hiddens
        
        # Prepare outputs
        outputs = []
        attention_weights = []
        
        # Start with BOS token or first token
        if use_scheduled_sampling and sampling_prob > 0:
            # Use scheduled sampling
            current_input = captions[:, 0:1]  # BOS token
            
            for t in range(1, caption_length):
                # Embed current input
                embedded = self.word_embedding(current_input)
                embedded = self.dropout(embedded)
                
                # Apply attention
                context_vector, attn_weights = self.attention(
                    decoder_hidden[0][-1:].transpose(0, 1),  # Last layer hidden state
                    encoder_outputs
                )
                
                # Concatenate embedding and context
                decoder_input = torch.cat([embedded, context_vector], dim=2)
                
                # LSTM forward
                decoder_output, decoder_hidden = self.decoder_lstm(decoder_input, decoder_hidden)
                
                # Project to vocabulary
                vocab_output = self.output_proj(decoder_output)
                
                outputs.append(vocab_output)
                attention_weights.append(attn_weights)
                
                # Scheduled sampling: decide whether to use ground truth or prediction
                if torch.rand(1).item() < sampling_prob:
                    # Use model's prediction
                    current_input = vocab_output.argmax(dim=-1)
                else:
                    # Use ground truth
                    current_input = captions[:, t:t+1]
        
        else:
            # Teacher forcing (standard training)
            # Embed all captions except last token
            embedded = self.word_embedding(captions[:, :-1])
            embedded = self.dropout(embedded)
            
            for t in range(embedded.size(1)):
                # Get current embedding
                current_embedded = embedded[:, t:t+1, :]
                
                # Apply attention
                context_vector, attn_weights = self.attention(
                    decoder_hidden[0][-1:].transpose(0, 1),
                    encoder_outputs
                )
                
                # Concatenate embedding and context
                decoder_input = torch.cat([current_embedded, context_vector], dim=2)
                
                # LSTM forward
                decoder_output, decoder_hidden = self.decoder_lstm(decoder_input, decoder_hidden)
                
                # Project to vocabulary
                vocab_output = self.output_proj(decoder_output)
                
                outputs.append(vocab_output)
                attention_weights.append(attn_weights)
        
        # Stack outputs
        outputs = torch.cat(outputs, dim=1)
        attention_weights = torch.cat(attention_weights, dim=1)
        
        return outputs, attention_weights
    
    def _forward_inference(self, encoder_outputs, encoder_hiddens, max_length):
        """Inference forward pass with attention"""
        batch_size = encoder_outputs.size(0)
        device = encoder_outputs.device
        
        # Initialize decoder hidden state
        decoder_hidden = encoder_hiddens
        
        # Start with BOS token
        current_input = torch.zeros(batch_size, 1, dtype=torch.long).to(device)
        
        outputs = []
        attention_weights = []
        
        for t in range(max_length):
            # Embed current input
            embedded = self.word_embedding(current_input)
            
            # Apply attention
            context_vector, attn_weights = self.attention(
                decoder_hidden[0][-1:].transpose(0, 1),
                encoder_outputs
            )
            
            # Concatenate embedding and context
            decoder_input = torch.cat([embedded, context_vector], dim=2)
            
            # LSTM forward
            decoder_output, decoder_hidden = self.decoder_lstm(decoder_input, decoder_hidden)
            
            # Project to vocabulary
            vocab_output = self.output_proj(decoder_output)
            
            outputs.append(vocab_output)
            attention_weights.append(attn_weights)
            
            # Get next token (greedy)
            current_input = vocab_output.argmax(dim=-1)
            
            # Check for EOS token
            if (current_input == self.vocab_size - 1).all():
                break
        
        outputs = torch.cat(outputs, dim=1)
        attention_weights = torch.cat(attention_weights, dim=1)
        
        return outputs, attention_weights


class GlobalAttention(nn.Module):
    """
    Global Attention Mechanism
    Computes attention weights over all encoder hidden states
    """
    
    def __init__(self, hidden_dim, attention_dim):
        super(GlobalAttention, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim
        
        # Attention layers
        self.encoder_proj = nn.Linear(hidden_dim, attention_dim)
        self.decoder_proj = nn.Linear(hidden_dim, attention_dim)
        self.attention_score = nn.Linear(attention_dim, 1)
        
        # Context combination
        self.context_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, decoder_hidden, encoder_outputs):
        """
        Apply attention mechanism
        
        Args:
            decoder_hidden: (batch_size, 1, hidden_dim)
            encoder_outputs: (batch_size, num_frames, hidden_dim)
            
        Returns:
            context_vector: (batch_size, 1, hidden_dim)
            attention_weights: (batch_size, 1, num_frames)
        """
        batch_size = encoder_outputs.size(0)
        num_frames = encoder_outputs.size(1)
        
        # Project encoder outputs
        encoder_proj = self.encoder_proj(encoder_outputs)  # (batch, frames, attn_dim)
        
        # Project decoder hidden state
        decoder_proj = self.decoder_proj(decoder_hidden)  # (batch, 1, attn_dim)
        
        # Expand decoder projection to match encoder frames
        decoder_proj = decoder_proj.expand(-1, num_frames, -1)  # (batch, frames, attn_dim)
        
        # Compute attention scores
        attention_input = self.tanh(encoder_proj + decoder_proj)  # (batch, frames, attn_dim)
        attention_scores = self.attention_score(attention_input).squeeze(2)  # (batch, frames)
        
        # Apply softmax to get attention weights
        attention_weights = self.softmax(attention_scores).unsqueeze(1)  # (batch, 1, frames)
        
        # Compute context vector
        context_vector = torch.bmm(attention_weights, encoder_outputs)  # (batch, 1, hidden_dim)
        
        # Combine context with decoder hidden state
        combined = torch.cat([context_vector, decoder_hidden], dim=2)  # (batch, 1, hidden_dim*2)
        context_vector = self.tanh(self.context_proj(combined))  # (batch, 1, hidden_dim)
        
        return context_vector, attention_weights


class ScheduledSamplingS2VT(AttentionS2VT):
    """
    S2VT with Scheduled Sampling to solve exposure bias
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sampling_prob = 0.0
    
    def set_sampling_prob(self, prob):
        """Set the sampling probability for scheduled sampling"""
        self.sampling_prob = prob
    
    def forward(self, video_features, captions=None, max_length=20):
        """Forward pass with scheduled sampling during training"""
        if self.training and captions is not None:
            return super().forward(
                video_features, 
                captions, 
                max_length, 
                use_scheduled_sampling=True, 
                sampling_prob=self.sampling_prob
            )
        else:
            return super().forward(video_features, captions, max_length)


# Enhanced Beam Search Implementation
class BeamSearchDecoder:
    """
    Enhanced Beam Search Decoder for better caption generation
    """
    
    def __init__(self, model, vocabulary, beam_size=5, max_length=20, length_penalty=1.0):
        self.model = model
        self.vocabulary = vocabulary
        self.beam_size = beam_size
        self.max_length = max_length
        self.length_penalty = length_penalty
        
        self.bos_id = vocabulary.word2idx[vocabulary.BOS_TOKEN]
        self.eos_id = vocabulary.word2idx[vocabulary.EOS_TOKEN]
        self.pad_id = vocabulary.word2idx[vocabulary.PAD_TOKEN]
    
    def decode(self, video_features):
        """
        Perform beam search decoding
        
        Args:
            video_features: (1, num_frames, feature_dim)
            
        Returns:
            best_sequence: List of token IDs
            best_score: Score of the best sequence
        """
        batch_size = video_features.size(0)
        device = video_features.device
        
        # Encode video
        if hasattr(self.model, '_encode_video'):
            encoder_outputs, encoder_hiddens = self.model._encode_video(video_features)
        else:
            # Fallback for basic S2VT
            return self._basic_beam_search(video_features)
        
        # Initialize beams
        beams = [(
            [self.bos_id],  # sequence
            0.0,           # score
            encoder_hiddens,  # hidden state
            []             # attention weights
        )]
        
        completed_beams = []
        
        for step in range(self.max_length):
            candidates = []
            
            for sequence, score, hidden, attn_weights in beams:
                if sequence[-1] == self.eos_id:
                    completed_beams.append((sequence, score, attn_weights))
                    continue
                
                # Get current token
                current_token = torch.tensor([[sequence[-1]]], dtype=torch.long).to(device)
                
                # Embed token
                embedded = self.model.word_embedding(current_token)
                
                # Apply attention
                context_vector, current_attn = self.model.attention(
                    hidden[0][-1:].transpose(0, 1),
                    encoder_outputs
                )
                
                # Decoder step
                decoder_input = torch.cat([embedded, context_vector], dim=2)
                decoder_output, new_hidden = self.model.decoder_lstm(decoder_input, hidden)
                
                # Get vocabulary probabilities
                vocab_output = self.model.output_proj(decoder_output)
                log_probs = F.log_softmax(vocab_output.squeeze(0), dim=-1)
                
                # Get top k candidates
                top_log_probs, top_indices = torch.topk(log_probs[-1], self.beam_size)
                
                for i in range(self.beam_size):
                    new_sequence = sequence + [top_indices[i].item()]
                    new_score = score + top_log_probs[i].item()
                    new_attn_weights = attn_weights + [current_attn]
                    
                    candidates.append((new_sequence, new_score, new_hidden, new_attn_weights))
            
            # Keep top beam_size candidates
            candidates.sort(key=lambda x: x[1] / (len(x[0]) ** self.length_penalty), reverse=True)
            beams = candidates[:self.beam_size]
            
            # Early stopping if all beams completed
            if len(completed_beams) >= self.beam_size:
                break
        
        # Add remaining beams to completed
        completed_beams.extend(beams)
        
        # Select best beam
        if completed_beams:
            best_beam = max(completed_beams, key=lambda x: x[1] / (len(x[0]) ** self.length_penalty))
            return best_beam[0], best_beam[1], best_beam[2]
        else:
            return [self.bos_id, self.eos_id], 0.0, []
    
    def _basic_beam_search(self, video_features):
        """Fallback beam search for basic S2VT model"""
        outputs = self.model(video_features, max_length=self.max_length)
        predicted_ids = outputs.argmax(dim=-1).squeeze(0)
        return predicted_ids.cpu().numpy().tolist(), 0.0, []