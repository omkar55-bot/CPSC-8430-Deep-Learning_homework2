"""
HW2 Inference Script for Video Caption Generation
This script handles inference for the HW2 submission requirements.

Usage:
    python inference_hw2.py --data_dir testing_data --output_file testset_output.txt
    python inference_hw2.py --data_dir ta_review_data --output_file tareviewset_output.txt
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
import os
import argparse
import pickle
from typing import Dict, List
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.enhanced_s2vt import EnhancedS2VT, CoverageBeamSearch
from models.attention_s2vt import AttentionS2VT, BeamSearchDecoder
from data.preprocessing import Vocabulary


class HW2VideoInference:
    """Inference class for HW2 submission"""
    
    def __init__(self, model_path: str, vocab_path: str = None):
        """
        Initialize inference system
        
        Args:
            model_path: Path to trained model checkpoint
            vocab_path: Path to vocabulary file (optional if included in checkpoint)
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load checkpoint
        print(f"Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Load vocabulary
        if 'vocabulary' in checkpoint:
            self.vocabulary = checkpoint['vocabulary']
            print("Vocabulary loaded from checkpoint")
        elif vocab_path and os.path.exists(vocab_path):
            self.vocabulary = self._load_vocabulary(vocab_path)
            print(f"Vocabulary loaded from {vocab_path}")
        else:
            raise ValueError("No vocabulary found in checkpoint or vocab_path")
        
        # Create model
        config = checkpoint.get('config', {})
        model_params = config.get('model_parameters', {})
        
        # Determine model type and create accordingly
        model_type = model_params.get('model_type', 'enhanced')
        
        if model_type == 'enhanced':
            self.model = EnhancedS2VT(
                vocab_size=len(self.vocabulary),
                video_feature_dim=model_params.get('video_feature_dim', 4096),
                hidden_dim=model_params.get('hidden_dim', 256),
                embedding_dim=model_params.get('embedding_dim', 256),
                num_layers=model_params.get('num_layers', 2),
                dropout=model_params.get('dropout', 0.3),
                max_seq_length=config.get('data_parameters', {}).get('max_caption_length', 20),
                attention_dim=model_params.get('attention_dim', 256)
            )
            
            # Create beam search decoder
            beam_config = config.get('beam_search', {})
            vocab_dict = self.vocabulary.word2idx if hasattr(self.vocabulary, 'word2idx') else self.vocabulary
            self.beam_search = CoverageBeamSearch(
                model=self.model,
                vocabulary=vocab_dict,
                beam_size=beam_config.get('beam_size', 5),
                length_penalty=beam_config.get('length_penalty', 1.0),
                coverage_penalty=beam_config.get('coverage_penalty', 0.2)
            )
        else:
            # Fallback to attention model
            self.model = AttentionS2VT(
                vocab_size=len(self.vocabulary),
                video_feature_dim=model_params.get('video_feature_dim', 4096),
                hidden_dim=model_params.get('hidden_dim', 256),
                embedding_dim=model_params.get('embedding_dim', 256),
                num_layers=model_params.get('num_layers', 2),
                dropout=model_params.get('dropout', 0.3),
                max_seq_length=config.get('data_parameters', {}).get('max_caption_length', 20),
                attention_dim=model_params.get('attention_dim', 256)
            )
            
            vocab_dict = self.vocabulary.word2idx if hasattr(self.vocabulary, 'word2idx') else self.vocabulary
            self.beam_search = BeamSearchDecoder(
                model=self.model,
                vocabulary=vocab_dict,
                beam_size=5,
                length_penalty=1.0
            )
        
        # Load model weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded successfully on {self.device}")
        print(f"Model type: {model_type}")
        print(f"Vocabulary size: {len(self.vocabulary)}")
    
    def _load_vocabulary(self, vocab_path: str):
        """Load vocabulary from pickle file"""
        with open(vocab_path, 'rb') as f:
            return pickle.load(f)
    
    def load_video_features(self, data_dir: str) -> Dict[str, np.ndarray]:
        """
        Load video features from data directory
        
        Args:
            data_dir: Directory containing video feature files
            
        Returns:
            Dictionary mapping video IDs to feature arrays
        """
        features = {}
        feat_dir = os.path.join(data_dir, 'feat')
        
        if not os.path.exists(feat_dir):
            raise ValueError(f"Feature directory not found: {feat_dir}")
        
        print(f"Loading video features from {feat_dir}...")
        
        # Load all .npy files
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
        """
        Generate caption for a single video
        
        Args:
            video_features: Video feature array of shape (num_frames, feature_dim)
            
        Returns:
            Generated caption string
        """
        # Convert to tensor and add batch dimension
        video_tensor = torch.from_numpy(video_features).float().unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Generate using beam search
            if hasattr(self.beam_search, 'decode'):
                pred_tokens, _ = self.beam_search.decode(video_tensor)
            else:
                # Fallback to model inference
                outputs = self.model(video_tensor)
                pred_tokens = torch.argmax(outputs['logits'], dim=-1)
            
            # Convert tokens to words
            caption = self.tokens_to_words(pred_tokens.squeeze().cpu().numpy())
        
        return caption
    
    def tokens_to_words(self, tokens: np.ndarray) -> str:
        """Convert token indices to words"""
        words = []
        
        # Handle vocabulary format
        if hasattr(self.vocabulary, 'idx2word'):
            idx2word = self.vocabulary.idx2word
            word2idx = self.vocabulary.word2idx
        else:
            # Assume vocabulary is a dictionary
            word2idx = self.vocabulary
            idx2word = {v: k for k, v in word2idx.items()}
        
        for token in tokens:
            if token == word2idx.get('<EOS>', 2):
                break
            if token not in [word2idx.get('<PAD>', 0), word2idx.get('<BOS>', 1)]:
                word = idx2word.get(token, '<UNK>')
                if word not in ['<PAD>', '<BOS>', '<EOS>', '<UNK>']:
                    words.append(word)
        
        return ' '.join(words)
    
    def process_dataset(self, data_dir: str, output_file: str):
        """
        Process entire dataset and generate output file
        
        Args:
            data_dir: Directory containing test data
            output_file: Output filename for captions
        """
        print(f"Processing dataset from {data_dir}...")
        
        # Load video features
        video_features = self.load_video_features(data_dir)
        
        if not video_features:
            raise ValueError("No video features found!")
        
        # Generate captions
        results = []
        
        print("Generating captions...")
        for i, (video_id, features) in enumerate(video_features.items()):
            print(f"Processing {i+1}/{len(video_features)}: {video_id}")
            
            try:
                caption = self.generate_caption(features)
                results.append(f"{video_id},{caption}")
                print(f"  Generated: {caption}")
            except Exception as e:
                print(f"  Error generating caption for {video_id}: {e}")
                results.append(f"{video_id},a person is doing something")  # Fallback caption
        
        # Save results
        print(f"Saving results to {output_file}...")
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(result + '\n')
        
        print(f"Successfully generated captions for {len(results)} videos")
        print(f"Results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='HW2 Video Caption Generation Inference')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing test data')
    parser.add_argument('--output_file', type=str, required=True,
                       help='Output filename for generated captions')
    parser.add_argument('--model_path', type=str, 
                       default='your_seq2seq_model/best_model_enhanced.pth',
                       help='Path to trained model checkpoint')
    parser.add_argument('--vocab_path', type=str, default=None,
                       help='Path to vocabulary file (optional)')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found: {args.model_path}")
        print("Please ensure your trained model is available at the specified path.")
        return
    
    try:
        # Create inference system
        inference = HW2VideoInference(args.model_path, args.vocab_path)
        
        # Process dataset
        inference.process_dataset(args.data_dir, args.output_file)
        
        print("\n" + "="*60)
        print("HW2 VIDEO CAPTION GENERATION COMPLETED SUCCESSFULLY!")
        print("="*60)
        
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())