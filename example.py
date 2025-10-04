"""
Example script demonstrating S2VT video caption generation
This script shows how to use the S2VT model for video caption generation
"""

import torch
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.s2vt_model import S2VTModel, S2VTLoss
from data.preprocessing import Vocabulary, VideoCaptionDataset, create_data_loader, load_sample_data
from utils.metrics import evaluate_captions
from utils.evaluation import generate_caption


def create_sample_data():
    """Create sample data for demonstration"""
    print("Creating sample data...")
    
    # Sample video features (5 videos, 60 frames each, 4096 features per frame)
    num_videos = 5
    num_frames = 60
    feature_dim = 4096
    
    video_features = np.random.randn(num_videos, num_frames, feature_dim).astype(np.float32)
    
    # Sample captions
    captions = [
        "a man is walking down the street",
        "a woman is cooking in the kitchen", 
        "children are playing in the park",
        "a dog is running in the yard",
        "people are dancing at a party"
    ]
    
    print(f"Created {num_videos} sample videos with shape: {video_features.shape}")
    print(f"Sample captions: {captions}")
    
    return video_features, captions


def demonstrate_preprocessing():
    """Demonstrate data preprocessing"""
    print("\n" + "="*50)
    print("DEMONSTRATING DATA PREPROCESSING")
    print("="*50)
    
    # Create sample data
    video_features, captions = create_sample_data()
    
    # Build vocabulary
    print("\nBuilding vocabulary...")
    vocab = Vocabulary()
    vocab.build_vocabulary(captions, min_count=1)
    
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Special tokens: {vocab.PAD_TOKEN}, {vocab.BOS_TOKEN}, {vocab.EOS_TOKEN}, {vocab.UNK_TOKEN}")
    
    # Demonstrate encoding/decoding
    sample_caption = captions[0]
    encoded = vocab.encode_caption(sample_caption, max_length=15)
    decoded = vocab.decode_caption(encoded)
    
    print(f"\nOriginal caption: {sample_caption}")
    print(f"Encoded indices: {encoded}")
    print(f"Decoded caption: {decoded}")
    
    # Create dataset
    print("\nCreating dataset...")
    dataset = VideoCaptionDataset(video_features, captions, vocab, max_caption_length=15, max_frames=80)
    
    # Create data loader
    data_loader = create_data_loader(dataset, batch_size=2, shuffle=False)
    
    # Show sample batch
    for batch in data_loader:
        print(f"Batch video features shape: {batch['video_features'].shape}")
        print(f"Batch captions shape: {batch['captions'].shape}")
        print(f"Sample caption text: {batch['caption_texts'][0]}")
        break
    
    return video_features, captions, vocab, dataset


def demonstrate_model():
    """Demonstrate model creation and forward pass"""
    print("\n" + "="*50)
    print("DEMONSTRATING MODEL")
    print("="*50)
    
    # Get sample data
    video_features, captions, vocab, dataset = demonstrate_preprocessing()
    
    # Create model
    print("\nCreating S2VT model...")
    model = S2VTModel(
        vocab_size=len(vocab),
        max_frames=80,
        video_feature_dim=4096,
        hidden_dim=256,  # Smaller for demo
        embedding_dim=256,
        num_layers=2,
        dropout=0.1
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create data loader
    data_loader = create_data_loader(dataset, batch_size=2)
    
    # Demonstrate forward pass
    print("\nDemonstrating forward pass...")
    model.train()  # Training mode
    
    for batch in data_loader:
        video_feat = batch['video_features']
        captions_batch = batch['captions']
        
        print(f"Input video features shape: {video_feat.shape}")
        print(f"Input captions shape: {captions_batch.shape}")
        
        # Forward pass
        outputs = model(video_feat, captions_batch)
        print(f"Output shape: {outputs.shape}")
        
        # Calculate loss
        criterion = S2VTLoss(len(vocab))
        targets = captions_batch[:, 1:]  # Remove BOS token
        outputs = outputs[:, :targets.size(1), :]  # Match sequence length
        loss = criterion(outputs, targets)
        
        print(f"Loss: {loss.item():.4f}")
        break
    
    return model, vocab


def demonstrate_inference():
    """Demonstrate model inference"""
    print("\n" + "="*50)
    print("DEMONSTRATING INFERENCE")
    print("="*50)
    
    # Get trained model (in practice, you would load from checkpoint)
    model, vocab = demonstrate_model()
    model.eval()  # Inference mode
    
    # Create sample video for inference
    sample_video = torch.randn(1, 60, 4096)  # (batch_size=1, frames=60, features=4096)
    
    print(f"Sample video features shape: {sample_video.shape}")
    
    # Generate caption with greedy decoding
    print("\nGenerating caption with greedy decoding...")
    greedy_caption = generate_caption(model, sample_video, vocab, method='greedy')
    print(f"Greedy caption: {greedy_caption}")
    
    # Generate caption with beam search
    print("\nGenerating caption with beam search...")
    beam_caption = generate_caption(model, sample_video, vocab, method='beam_search')
    print(f"Beam search caption: {beam_caption}")


def demonstrate_evaluation():
    """Demonstrate evaluation metrics"""
    print("\n" + "="*50)
    print("DEMONSTRATING EVALUATION")
    print("="*50)
    
    # Sample predictions and references
    predictions = [
        "a man is walking down the street",
        "a woman is cooking food",
        "children are playing outside"
    ]
    
    references = [
        ["a person is walking on the road", "a man walks down the street"],
        ["a woman cooks in the kitchen", "someone is cooking food"],
        ["kids play in the park", "children are having fun outside"]
    ]
    
    print("Sample predictions:")
    for i, pred in enumerate(predictions):
        print(f"  {i+1}. {pred}")
    
    print("\nSample references:")
    for i, refs in enumerate(references):
        print(f"  {i+1}. {refs}")
    
    # Calculate evaluation metrics
    print("\nCalculating evaluation metrics...")
    scores = evaluate_captions(predictions, references)
    
    print("\nEvaluation Results:")
    for metric, score in scores.items():
        print(f"  {metric}: {score:.4f}")


def main():
    """Main demonstration function"""
    print("S2VT VIDEO CAPTION GENERATION DEMONSTRATION")
    print("="*60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        # Run demonstrations
        demonstrate_preprocessing()
        demonstrate_model()
        demonstrate_inference() 
        demonstrate_evaluation()
        
        print("\n" + "="*60)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        print("\nNext steps:")
        print("1. Prepare your video features and captions")
        print("2. Train the model using: python train.py")
        print("3. Run inference using: python inference.py --demo")
        print("4. Evaluate on your test data")
        
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        print("Please check your installation and try again.")


if __name__ == "__main__":
    main()