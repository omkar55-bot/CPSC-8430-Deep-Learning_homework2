"""
Inference script for S2VT Video Caption Generation with MSVD Dataset
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
from data.msvd_dataset import MSVDDataset
from utils.evaluation import generate_caption, evaluate_model
from utils.metrics import evaluate_captions


class MSVDInference:
    """Inference class for S2VT model trained on MSVD dataset"""
    
    def __init__(self, model_path, vocab_path=None, device=None):
        """
        Initialize inference class
        
        Args:
            model_path: Path to trained model checkpoint
            vocab_path: Path to vocabulary file (if None, will try to load from checkpoint dir)
            device: Device to run inference on
        """
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model and vocabulary
        self.model, self.vocabulary, self.config = self._load_model_and_vocab(model_path, vocab_path)
        self.model.eval()
        
        print(f"MSVD Inference initialized on {self.device}")
        print(f"Vocabulary size: {len(self.vocabulary)}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def _load_model_and_vocab(self, model_path, vocab_path):
        """Load trained model and vocabulary"""
        print(f"Loading model from {model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        config = checkpoint['config']
        
        # Load vocabulary
        if vocab_path is None:
            # Try to find vocabulary in the same directory as model
            model_dir = os.path.dirname(model_path)
            vocab_path = os.path.join(model_dir, 'vocabulary.pkl')
        
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")
        
        vocabulary = Vocabulary()
        vocabulary.load(vocab_path)
        
        # Initialize model
        model = S2VTModel(
            vocab_size=len(vocabulary),
            max_frames=config['max_frames'],
            video_feature_dim=config['video_feature_dim'],
            hidden_dim=config['hidden_dim'],
            embedding_dim=config['embedding_dim'],
            num_layers=config['num_layers'],
            dropout=config['dropout']
        ).to(self.device)
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"Model loaded from epoch {checkpoint['epoch']}")
        print(f"Best validation loss: {checkpoint.get('best_score', 'N/A')}")
        
        return model, vocabulary, config
    
    def generate_caption_for_video_file(self, video_feat_path, method='greedy', max_length=20):
        """
        Generate caption for a video feature file
        
        Args:
            video_feat_path: Path to .npy video feature file
            method: Generation method ('greedy' or 'beam_search')
            max_length: Maximum caption length
            
        Returns:
            Generated caption string
        """
        # Load video features
        video_features = np.load(video_feat_path)
        
        # Convert to tensor and add batch dimension
        video_features = torch.tensor(video_features, dtype=torch.float32)
        if len(video_features.shape) == 2:
            video_features = video_features.unsqueeze(0)  # Add batch dimension
        
        # Pad/truncate to max_frames
        max_frames = self.config['max_frames']
        num_frames = video_features.shape[1]
        
        if num_frames < max_frames:
            # Pad with zeros
            padding = torch.zeros(1, max_frames - num_frames, video_features.shape[2])
            video_features = torch.cat([video_features, padding], dim=1)
        else:
            # Truncate (take evenly spaced frames)
            indices = torch.linspace(0, num_frames - 1, max_frames).long()
            video_features = video_features[:, indices, :]
        
        # Generate caption
        caption = generate_caption(
            self.model, 
            video_features, 
            self.vocabulary, 
            max_length=max_length, 
            method=method
        )
        
        return caption
    
    def evaluate_on_msvd_test_set(self, data_root, num_samples=None):
        """
        Evaluate model on MSVD test set
        
        Args:
            data_root: Root directory of MLDS_hw2_1_data
            num_samples: Number of samples to evaluate (None for all)
            
        Returns:
            Dictionary with evaluation metrics
        """
        print("Evaluating on MSVD test set...")
        
        # Create test dataset
        test_dataset = MSVDDataset(
            data_root=data_root,
            split='test',
            vocabulary=self.vocabulary,
            max_caption_length=self.config['max_caption_length'],
            max_frames=self.config['max_frames']
        )
        
        predictions = []
        references = []
        video_ids = []
        
        # Limit samples if specified
        if num_samples:
            indices = np.random.choice(len(test_dataset), min(num_samples, len(test_dataset)), replace=False)
        else:
            indices = range(len(test_dataset))
            num_samples = len(test_dataset)
        
        print(f"Evaluating on {len(indices)} samples...")
        
        with torch.no_grad():
            for i, idx in enumerate(indices):
                if i % 50 == 0:
                    print(f"Processing {i+1}/{len(indices)}...")
                
                sample = test_dataset[idx]
                video_features = sample['video_features'].unsqueeze(0).to(self.device)  # Add batch dim
                
                # Generate caption
                outputs = self.model(video_features, max_length=self.config['max_caption_length'])
                predicted_ids = outputs.argmax(dim=-1).squeeze(0)
                
                # Decode caption
                predicted_caption = self.vocabulary.decode_caption(
                    predicted_ids.cpu().numpy().tolist()
                )
                
                predictions.append(predicted_caption)
                references.append(sample['caption_text'])  # All reference captions
                video_ids.append(sample['video_id'])
        
        # Calculate evaluation metrics
        print("Calculating evaluation metrics...")
        scores = evaluate_captions(predictions, references)
        
        # Print results
        print("\nMSVD Test Set Evaluation Results:")
        print("=" * 40)
        for metric, score in scores.items():
            print(f"{metric:12}: {score:.4f}")
        
        # Show some examples
        print(f"\nExample Predictions:")
        print("=" * 60)
        for i in range(min(10, len(predictions))):
            print(f"Video ID: {video_ids[i]}")
            print(f"Reference: {references[i][0] if isinstance(references[i], list) else references[i]}")
            print(f"Predicted: {predictions[i]}")
            print("-" * 60)
        
        # Save detailed results
        results = {
            'scores': scores,
            'predictions': predictions,
            'references': references,
            'video_ids': video_ids,
            'num_samples': len(predictions)
        }
        
        return results
    
    def generate_captions_for_directory(self, feat_dir, output_file, method='greedy'):
        """
        Generate captions for all videos in a directory
        
        Args:
            feat_dir: Directory containing .npy feature files
            output_file: Output JSON file path
            method: Generation method
        """
        print(f"Generating captions for videos in {feat_dir}")
        
        # Get all .npy files
        feat_files = [f for f in os.listdir(feat_dir) if f.endswith('.npy')]
        feat_files.sort()
        
        print(f"Found {len(feat_files)} video feature files")
        
        results = []
        
        for i, feat_file in enumerate(feat_files):
            if i % 100 == 0:
                print(f"Processing {i+1}/{len(feat_files)}...")
            
            feat_path = os.path.join(feat_dir, feat_file)
            video_id = feat_file.replace('.npy', '')
            
            # Generate caption
            try:
                caption = self.generate_caption_for_video_file(feat_path, method=method)
                results.append({
                    'video_id': video_id,
                    'caption': caption
                })
            except Exception as e:
                print(f"Error processing {feat_file}: {e}")
                results.append({
                    'video_id': video_id,
                    'caption': "Error generating caption"
                })
        
        # Save results
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {output_file}")
        return results
    
    def demo_random_videos(self, data_root, num_videos=5):
        """
        Demo with random videos from MSVD dataset
        
        Args:
            data_root: Root directory of MLDS_hw2_1_data
            num_videos: Number of videos to demo
        """
        print(f"Running demo with {num_videos} random videos...")
        
        # Create test dataset
        test_dataset = MSVDDataset(
            data_root=data_root,
            split='test',
            vocabulary=self.vocabulary,
            max_caption_length=self.config['max_caption_length'],
            max_frames=self.config['max_frames']
        )
        
        # Select random videos
        indices = np.random.choice(len(test_dataset), num_videos, replace=False)
        
        for i, idx in enumerate(indices):
            sample = test_dataset[idx]
            video_features = sample['video_features'].unsqueeze(0).to(self.device)
            
            print(f"\nVideo {i+1}: {sample['video_id']}")
            print("-" * 50)
            
            # Generate with different methods
            greedy_caption = self.generate_caption_for_video_file(
                test_dataset.data[idx]['feat_file'], method='greedy'
            )
            
            beam_caption = self.generate_caption_for_video_file(
                test_dataset.data[idx]['feat_file'], method='beam_search'
            )
            
            print(f"Reference captions:")
            for j, ref in enumerate(sample['caption_text'][:3]):  # Show first 3 references
                print(f"  {j+1}. {ref}")
            
            print(f"\nGenerated captions:")
            print(f"  Greedy:      {greedy_caption}")
            print(f"  Beam Search: {beam_caption}")


def main():
    """Main inference function"""
    parser = argparse.ArgumentParser(description='S2VT MSVD Inference')
    
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--vocab_path', type=str, default=None,
                       help='Path to vocabulary file')
    parser.add_argument('--data_root', type=str, default='E:/imgsynth/MLDS_hw2_1_data',
                       help='Root directory of MLDS_hw2_1_data')
    parser.add_argument('--video_feat', type=str, default=None,
                       help='Path to single video feature file')
    parser.add_argument('--feat_dir', type=str, default=None,
                       help='Directory containing video feature files')
    parser.add_argument('--output_file', type=str, default='msvd_results.json',
                       help='Output file for results')
    parser.add_argument('--method', type=str, default='greedy', 
                       choices=['greedy', 'beam_search'],
                       help='Caption generation method')
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate on MSVD test set')
    parser.add_argument('--demo', action='store_true',
                       help='Run demo with random videos')
    parser.add_argument('--num_samples', type=int, default=None,
                       help='Number of samples for evaluation/demo')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to run inference on (cpu/cuda)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize inference
    inference = MSVDInference(args.model_path, args.vocab_path, device)
    
    if args.demo:
        # Run demo
        inference.demo_random_videos(args.data_root, args.num_samples or 5)
    
    elif args.evaluate:
        # Evaluate on test set
        results = inference.evaluate_on_msvd_test_set(args.data_root, args.num_samples)
        
        # Save evaluation results
        eval_output = args.output_file.replace('.json', '_evaluation.json')
        with open(eval_output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Evaluation results saved to {eval_output}")
    
    elif args.video_feat:
        # Generate caption for single video
        caption = inference.generate_caption_for_video_file(args.video_feat, args.method)
        print(f"Video: {args.video_feat}")
        print(f"Caption: {caption}")
    
    elif args.feat_dir:
        # Generate captions for directory
        results = inference.generate_captions_for_directory(
            args.feat_dir, args.output_file, args.method
        )
        print(f"Generated {len(results)} captions")
    
    else:
        print("Please specify one of: --demo, --evaluate, --video_feat, or --feat_dir")


if __name__ == "__main__":
    main()