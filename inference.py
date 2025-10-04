"""
Inference script for S2VT Video Caption Generation
"""

import torch
import numpy as np
import argparse
import os
import json
from PIL import Image
import matplotlib.pyplot as plt

# Import our modules
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.s2vt_model import S2VTModel
from data.preprocessing import Vocabulary
from utils.evaluation import generate_caption, inference_demo, evaluate_model


class S2VTInference:
    """Inference class for S2VT model"""
    
    def __init__(self, model_path, vocab_path, device=None):
        """
        Initialize inference class
        
        Args:
            model_path: Path to trained model checkpoint
            vocab_path: Path to vocabulary file
            device: Device to run inference on
        """
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load vocabulary
        self.vocabulary = Vocabulary()
        self.vocabulary.load(vocab_path)
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()
        
        print(f"Model loaded on {self.device}")
        print(f"Vocabulary size: {len(self.vocabulary)}")
    
    def _load_model(self, model_path):
        """Load trained model from checkpoint"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Get model configuration
        config = checkpoint['config']
        
        # Initialize model
        model = S2VTModel(
            vocab_size=len(self.vocabulary),
            max_frames=config['max_frames'],
            video_feature_dim=config['video_feature_dim'],
            hidden_dim=config['hidden_dim'],
            embedding_dim=config['embedding_dim'],
            num_layers=config['num_layers'],
            dropout=config['dropout']
        ).to(self.device)
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model
    
    def generate_caption_for_video(self, video_features, method='greedy', max_length=20):
        """
        Generate caption for a single video
        
        Args:
            video_features: Video features array (num_frames, feature_dim)
            method: Generation method ('greedy' or 'beam_search')
            max_length: Maximum caption length
            
        Returns:
            Generated caption string
        """
        # Convert to tensor and add batch dimension
        if isinstance(video_features, np.ndarray):
            video_features = torch.tensor(video_features, dtype=torch.float32)
        
        if len(video_features.shape) == 2:
            video_features = video_features.unsqueeze(0)  # Add batch dimension
        
        # Generate caption
        caption = generate_caption(
            self.model, 
            video_features, 
            self.vocabulary, 
            max_length=max_length, 
            method=method
        )
        
        return caption
    
    def generate_multiple_captions(self, video_features_list, method='greedy', max_length=20):
        """
        Generate captions for multiple videos
        
        Args:
            video_features_list: List of video features arrays
            method: Generation method
            max_length: Maximum caption length
            
        Returns:
            List of generated captions
        """
        captions = []
        
        for i, video_features in enumerate(video_features_list):
            print(f"Processing video {i+1}/{len(video_features_list)}")
            caption = self.generate_caption_for_video(video_features, method, max_length)
            captions.append(caption)
        
        return captions
    
    def run_demo(self, num_samples=5):
        """
        Run demo with sample video features
        
        Args:
            num_samples: Number of sample videos to generate captions for
        """
        print("Running S2VT inference demo...")
        
        # Generate sample video features
        sample_features = np.random.randn(num_samples, 80, 4096).astype(np.float32)
        
        print(f"Generated {num_samples} sample videos with features shape: {sample_features.shape}")
        
        # Generate captions
        for i in range(num_samples):
            print(f"\nVideo {i+1}:")
            
            # Greedy decoding
            greedy_caption = self.generate_caption_for_video(
                sample_features[i], method='greedy'
            )
            
            # Beam search decoding
            beam_caption = self.generate_caption_for_video(
                sample_features[i], method='beam_search'
            )
            
            print(f"  Greedy: {greedy_caption}")
            print(f"  Beam Search: {beam_caption}")
    
    def evaluate_on_data(self, video_features, reference_captions, max_length=20):
        """
        Evaluate model on provided data
        
        Args:
            video_features: Array of video features (num_videos, num_frames, feature_dim)
            reference_captions: List of reference captions
            max_length: Maximum caption length
            
        Returns:
            Dictionary with evaluation metrics
        """
        print("Generating captions for evaluation...")
        
        predictions = []
        references = []
        
        for i, features in enumerate(video_features):
            # Generate caption
            predicted_caption = self.generate_caption_for_video(features, max_length=max_length)
            predictions.append(predicted_caption)
            references.append([reference_captions[i]])  # Wrap in list for multiple references
        
        # Calculate evaluation metrics
        from utils.metrics import evaluate_captions
        scores = evaluate_captions(predictions, references)
        
        # Print results
        print("\nEvaluation Results:")
        for metric, score in scores.items():
            print(f"{metric}: {score:.4f}")
        
        # Show some examples
        print("\nExample Predictions:")
        for i in range(min(5, len(predictions))):
            print(f"Reference: {reference_captions[i]}")
            print(f"Predicted: {predictions[i]}")
            print("-" * 50)
        
        return scores
    
    def save_results(self, captions, output_path):
        """
        Save generated captions to file
        
        Args:
            captions: List of generated captions
            output_path: Output file path
        """
        results = {
            'captions': captions,
            'model_info': {
                'vocab_size': len(self.vocabulary),
                'device': str(self.device)
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {output_path}")


def load_video_features_from_file(file_path):
    """
    Load video features from file
    This is a placeholder - implement based on your data format
    
    Args:
        file_path: Path to video features file
        
    Returns:
        Video features array
    """
    # Example implementation for different file formats
    if file_path.endswith('.npy'):
        return np.load(file_path)
    elif file_path.endswith('.npz'):
        data = np.load(file_path)
        return data['features']  # Assuming features are stored under 'features' key
    else:
        raise ValueError(f"Unsupported file format: {file_path}")


def main():
    """Main inference function"""
    parser = argparse.ArgumentParser(description='S2VT Video Caption Generation Inference')
    
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--vocab_path', type=str, required=True,
                       help='Path to vocabulary file')
    parser.add_argument('--video_features', type=str, default=None,
                       help='Path to video features file')
    parser.add_argument('--reference_captions', type=str, default=None,
                       help='Path to reference captions file (for evaluation)')
    parser.add_argument('--output_path', type=str, default='results.json',
                       help='Output path for generated captions')
    parser.add_argument('--method', type=str, default='greedy', 
                       choices=['greedy', 'beam_search'],
                       help='Caption generation method')
    parser.add_argument('--max_length', type=int, default=20,
                       help='Maximum caption length')
    parser.add_argument('--demo', action='store_true',
                       help='Run demo with sample data')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to run inference on (cpu/cuda)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize inference class
    inference = S2VTInference(args.model_path, args.vocab_path, device)
    
    if args.demo:
        # Run demo
        inference.run_demo()
    
    elif args.video_features:
        # Load video features
        print(f"Loading video features from {args.video_features}")
        video_features = load_video_features_from_file(args.video_features)
        
        print(f"Loaded video features shape: {video_features.shape}")
        
        # Generate captions
        captions = inference.generate_multiple_captions(
            video_features, 
            method=args.method, 
            max_length=args.max_length
        )
        
        # Save results
        inference.save_results(captions, args.output_path)
        
        # Evaluate if reference captions provided
        if args.reference_captions:
            with open(args.reference_captions, 'r') as f:
                if args.reference_captions.endswith('.json'):
                    ref_data = json.load(f)
                    reference_captions = ref_data['captions'] if 'captions' in ref_data else ref_data
                else:
                    reference_captions = [line.strip() for line in f.readlines()]
            
            scores = inference.evaluate_on_data(
                video_features, 
                reference_captions, 
                max_length=args.max_length
            )
    
    else:
        print("Please provide --video_features or use --demo flag")


if __name__ == "__main__":
    main()