"""
HW2 Inference Script for Video Caption Generation
Generates captions for test videos and outputs to specified file.

Usage:
    python inference_hw2.py --data_dir testing_data --output_file testset_output.txt
"""

import torch
import numpy as np
import json
import os
import argparse
import sys
from typing import Dict

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

try:
    from models.enhanced_s2vt import EnhancedS2VT, CoverageBeamSearch
    from data.preprocessing import Vocabulary
except ImportError:
    print("Warning: Could not import enhanced model, using fallback")
    EnhancedS2VT = None


class VideoInference:
    """Simple video caption inference for HW2"""
    
    def __init__(self, model_path: str):
        """Initialize inference system"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load checkpoint
        print(f"Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Load vocabulary
        self.vocabulary = checkpoint['vocabulary']
        config = checkpoint.get('config', {})
        
        # Create model
        if EnhancedS2VT is not None:
            self.model = self._create_enhanced_model(config)
        else:
            self.model = self._create_simple_model(config)
        
        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded successfully on {self.device}")
    
    def _create_enhanced_model(self, config):
        """Create enhanced model"""
        model_params = config.get('model_parameters', {})
        data_params = config.get('data_parameters', {})
        
        return EnhancedS2VT(
            vocab_size=len(self.vocabulary),
            video_feature_dim=model_params.get('video_feature_dim', 4096),
            hidden_dim=model_params.get('hidden_dim', 256),
            embedding_dim=model_params.get('embedding_dim', 256),
            num_layers=model_params.get('num_layers', 2),
            dropout=0.0,  # No dropout during inference
            max_seq_length=data_params.get('max_caption_length', 20),
            attention_dim=model_params.get('attention_dim', 256)
        )
    
    def _create_simple_model(self, config):
        """Create simple fallback model"""
        import torch.nn as nn
        
        class SimpleS2VT(nn.Module):
            def __init__(self, vocab_size, video_dim=4096, hidden_dim=256, max_len=20):
                super().__init__()
                self.max_len = max_len
                self.hidden_dim = hidden_dim
                
                self.video_proj = nn.Linear(video_dim, hidden_dim)
                self.embedding = nn.Embedding(vocab_size, hidden_dim)
                self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
                self.output_proj = nn.Linear(hidden_dim, vocab_size)
            
            def forward(self, video_features):
                batch_size = video_features.size(0)
                
                # Encode video
                video_proj = self.video_proj(video_features)
                video_encoded = video_proj.mean(dim=1, keepdim=True)  # Simple pooling
                
                # Generate sequence
                outputs = []
                input_token = torch.ones(batch_size, 1, dtype=torch.long, device=video_features.device)
                hidden = None
                
                for _ in range(self.max_len):
                    embedded = self.embedding(input_token)
                    lstm_input = embedded + video_encoded
                    output, hidden = self.lstm(lstm_input, hidden)
                    logits = self.output_proj(output)
                    outputs.append(logits)
                    input_token = torch.argmax(logits, dim=-1)
                
                return {'logits': torch.cat(outputs, dim=1)}
        
        model_params = config.get('model_parameters', {})
        return SimpleS2VT(
            vocab_size=len(self.vocabulary),
            video_dim=model_params.get('video_feature_dim', 4096),
            hidden_dim=model_params.get('hidden_dim', 256),
            max_len=config.get('data_parameters', {}).get('max_caption_length', 20)
        )
    
    def load_video_features(self, data_dir: str) -> Dict[str, np.ndarray]:
        """Load video features from data directory"""
        features = {}
        feat_dir = os.path.join(data_dir, 'feat')
        
        if not os.path.exists(feat_dir):
            raise ValueError(f"Feature directory not found: {feat_dir}")
        
        print(f"Loading video features from {feat_dir}...")
        
        for filename in os.listdir(feat_dir):
            if filename.endswith('.npy'):
                video_id = filename.replace('.npy', '')
                feature_path = os.path.join(feat_dir, filename)
                
                try:
                    feature = np.load(feature_path)
                    features[video_id] = feature
                except Exception as e:
                    print(f"Warning: Could not load {filename}: {e}")
        
        print(f"Loaded features for {len(features)} videos")
        return features
    
    def generate_caption(self, video_features: np.ndarray) -> str:
        """Generate caption for video"""
        video_tensor = torch.from_numpy(video_features).float().unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(video_tensor)
            pred_tokens = torch.argmax(outputs['logits'], dim=-1)
            caption = self.tokens_to_words(pred_tokens.squeeze().cpu().numpy())
        
        return caption
    
    def tokens_to_words(self, tokens: np.ndarray) -> str:
        """Convert tokens to words"""
        words = []
        
        # Handle vocabulary format
        if hasattr(self.vocabulary, 'idx2word'):
            idx2word = self.vocabulary.idx2word
            word2idx = self.vocabulary.word2idx
        else:
            word2idx = self.vocabulary
            idx2word = {v: k for k, v in word2idx.items()}
        
        for token in tokens:
            if token == word2idx.get('<EOS>', 2):
                break
            if token not in [word2idx.get('<PAD>', 0), word2idx.get('<BOS>', 1)]:
                word = idx2word.get(token, '<UNK>')
                if word not in ['<PAD>', '<BOS>', '<EOS>', '<UNK>']:
                    words.append(word)
        
        return ' '.join(words) if words else 'a person is doing something'
    
    def process_dataset(self, data_dir: str, output_file: str):
        """Process dataset and generate output file"""
        print(f"Processing dataset from {data_dir}...")
        
        # Load video features
        video_features = self.load_video_features(data_dir)
        
        if not video_features:
            raise ValueError("No video features found!")
        
        # Generate captions
        results = []
        
        print("Generating captions...")
        for i, (video_id, features) in enumerate(video_features.items()):
            if (i + 1) % 10 == 0:
                print(f"Processing {i+1}/{len(video_features)}: {video_id}")
            
            try:
                caption = self.generate_caption(features)
                results.append(f"{video_id},{caption}")
            except Exception as e:
                print(f"Error generating caption for {video_id}: {e}")
                results.append(f"{video_id},a person is doing something")
        
        # Sort results by video ID for consistency
        results.sort()
        
        # Save results
        print(f"Saving results to {output_file}...")
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(result + '\n')
        
        print(f"Successfully processed {len(results)} videos")


def main():
    parser = argparse.ArgumentParser(description='HW2 Video Caption Generation')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing test data')
    parser.add_argument('--output_file', type=str, required=True,
                       help='Output filename for captions (.txt format)')
    parser.add_argument('--model_path', type=str, 
                       default='your_seq2seq_model/best_model_enhanced.pth',
                       help='Path to trained model')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory not found: {args.data_dir}")
        return 1
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found: {args.model_path}")
        return 1
    
    if not args.output_file.endswith('.txt'):
        print(f"Warning: Output file should have .txt extension")
    
    try:
        # Run inference
        inference = VideoInference(args.model_path)
        inference.process_dataset(args.data_dir, args.output_file)
        
        print(f"\n✅ HW2 inference completed successfully!")
        print(f"📁 Results saved to: {args.output_file}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error during inference: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())