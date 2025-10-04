"""
S2VT (Sequence to Sequence - Video to Text) Model Implementation
Based on the paper: http://www.cs.utexas.edu/users/ml/papers/venugopalan.iccv15.pdf

This model implements a two-layer LSTM architecture for video caption generation:
- Encoder LSTM: Processes video features
- Decoder LSTM: Generates text captions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class S2VTModel(nn.Module):
    """
    S2VT Model for Video Caption Generation
    
    Architecture:
    - Two-layer LSTM structure
    - First LSTM (encoder) processes video features
    - Second LSTM (decoder) generates text output
    """
    
    def __init__(self, 
                 vocab_size,
                 max_frames,
                 video_feature_dim=4096,  # CNN feature dimension
                 hidden_dim=512,
                 embedding_dim=512,
                 num_layers=2,
                 dropout=0.5):
        """
        Initialize S2VT model
        
        Args:
            vocab_size: Size of vocabulary
            max_frames: Maximum number of video frames
            video_feature_dim: Dimension of video features (from CNN)
            hidden_dim: Hidden dimension of LSTM
            embedding_dim: Word embedding dimension
            num_layers: Number of LSTM layers (should be 2 for S2VT)
            dropout: Dropout rate
        """
        super(S2VTModel, self).__init__()
        
        self.vocab_size = vocab_size
        self.max_frames = max_frames
        self.video_feature_dim = video_feature_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        
        # Video feature projection to match LSTM input dimension
        self.video_feature_proj = nn.Linear(video_feature_dim, hidden_dim)
        
        # Word embedding layer
        self.word_embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Two-layer LSTM as described in S2VT paper
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Output projection layer
        self.output_proj = nn.Linear(hidden_dim, vocab_size)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        # Initialize LSTM weights
        for name, param in self.lstm.named_parameters():
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
    
    def forward(self, video_features, captions=None, max_length=20):
        """
        Forward pass of S2VT model
        
        Args:
            video_features: (batch_size, num_frames, feature_dim)
            captions: (batch_size, caption_length) - for training
            max_length: Maximum caption length for inference
            
        Returns:
            outputs: (batch_size, sequence_length, vocab_size)
        """
        batch_size = video_features.size(0)
        num_frames = video_features.size(1)
        device = video_features.device
        
        # Initialize hidden states
        hidden = self.init_hidden(batch_size, device)
        
        # Project video features
        video_features = self.video_feature_proj(video_features)
        video_features = self.dropout(video_features)
        
        if self.training and captions is not None:
            return self._forward_train(video_features, captions, hidden)
        else:
            return self._forward_inference(video_features, hidden, max_length)
    
    def _forward_train(self, video_features, captions, hidden):
        """Training forward pass"""
        batch_size = video_features.size(0)
        num_frames = video_features.size(1)
        caption_length = captions.size(1)
        device = video_features.device
        
        # Phase 1: Encoding - Process video features
        # Feed video features with padding tokens for text positions
        pad_tokens = torch.zeros(batch_size, caption_length, self.hidden_dim).to(device)
        
        # Concatenate video features and padding for text
        encoder_input = torch.cat([video_features, pad_tokens], dim=1)
        
        # Run through LSTM for encoding phase
        encoder_output, hidden = self.lstm(encoder_input, hidden)
        
        # Phase 2: Decoding - Generate text
        # Prepare decoder input (shift captions by one position)
        # Add BOS token at the beginning
        decoder_input_tokens = torch.cat([
            torch.zeros(batch_size, 1, dtype=torch.long).to(device),  # BOS token
            captions[:, :-1]
        ], dim=1)
        
        # Embed decoder input tokens
        decoder_embedded = self.word_embedding(decoder_input_tokens)
        decoder_embedded = self.dropout(decoder_embedded)
        
        # Project embeddings to hidden dimension
        decoder_input = torch.zeros(batch_size, caption_length, self.hidden_dim).to(device)
        decoder_input = decoder_embedded @ self.word_embedding.weight.T @ torch.eye(self.embedding_dim, self.hidden_dim).to(device)
        
        # Actually, let's use a simpler approach - project embeddings properly
        if self.embedding_dim != self.hidden_dim:
            embed_proj = nn.Linear(self.embedding_dim, self.hidden_dim).to(device)
            decoder_input = embed_proj(decoder_embedded)
        else:
            decoder_input = decoder_embedded
        
        # Concatenate padding for video frames and decoder input
        video_pad = torch.zeros(batch_size, num_frames, self.hidden_dim).to(device)
        decoder_full_input = torch.cat([video_pad, decoder_input], dim=1)
        
        # Run decoder LSTM
        decoder_output, _ = self.lstm(decoder_full_input, hidden)
        
        # Extract only the text generation part
        text_output = decoder_output[:, num_frames:, :]
        
        # Project to vocabulary size
        outputs = self.output_proj(text_output)
        
        return outputs
    
    def _forward_inference(self, video_features, hidden, max_length):
        """Inference forward pass with beam search capability"""
        batch_size = video_features.size(0)
        num_frames = video_features.size(1)
        device = video_features.device
        
        # Phase 1: Encoding
        pad_tokens = torch.zeros(batch_size, max_length, self.hidden_dim).to(device)
        encoder_input = torch.cat([video_features, pad_tokens], dim=1)
        encoder_output, hidden = self.lstm(encoder_input, hidden)
        
        # Phase 2: Decoding - Generate one token at a time
        outputs = []
        current_token = torch.zeros(batch_size, 1, dtype=torch.long).to(device)  # BOS
        
        for t in range(max_length):
            # Embed current token
            embedded = self.word_embedding(current_token)
            
            # Project to hidden dimension if needed
            if self.embedding_dim != self.hidden_dim:
                # Create a simple projection
                embedded = embedded.view(batch_size, 1, self.embedding_dim)
                decoder_input = torch.zeros(batch_size, 1, self.hidden_dim).to(device)
                # Simple linear transformation
                decoder_input = embedded @ self.word_embedding.weight[:self.hidden_dim, :].T
            else:
                decoder_input = embedded
            
            # Prepare full input (padding for video frames + current token)
            video_pad = torch.zeros(batch_size, num_frames, self.hidden_dim).to(device)
            full_input = torch.cat([video_pad, decoder_input], dim=1)
            
            # LSTM forward
            lstm_out, hidden = self.lstm(full_input, hidden)
            
            # Get output for text generation
            text_out = lstm_out[:, num_frames:, :]
            
            # Project to vocabulary
            vocab_out = self.output_proj(text_out)
            
            outputs.append(vocab_out)
            
            # Get next token (greedy)
            current_token = vocab_out.argmax(dim=-1)
            
            # Check for EOS token (assuming EOS token id is vocab_size - 1)
            if (current_token == self.vocab_size - 1).all():
                break
        
        return torch.cat(outputs, dim=1)


class S2VTLoss(nn.Module):
    """Custom loss function for S2VT model"""
    
    def __init__(self, vocab_size, pad_token_id=0):
        super(S2VTLoss, self).__init__()
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: (batch_size, seq_len, vocab_size)
            targets: (batch_size, seq_len)
        """
        # Reshape for CrossEntropyLoss
        predictions = predictions.view(-1, self.vocab_size)
        targets = targets.view(-1)
        
        return self.criterion(predictions, targets)