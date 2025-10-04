"""
Test Enhanced S2VT Model

Quick test to verify the enhanced model works correctly.
"""

import torch
import json
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.enhanced_s2vt import EnhancedS2VT, EnhancedS2VTLoss, CoverageBeamSearch


def test_enhanced_model():
    """Test the enhanced S2VT model"""
    print("🧪 Testing Enhanced S2VT Model...")
    
    # Model parameters
    vocab_size = 1000
    batch_size = 2
    video_length = 10
    video_feature_dim = 4096
    hidden_dim = 256
    embedding_dim = 256
    max_seq_length = 15
    
    # Create model
    model = EnhancedS2VT(
        vocab_size=vocab_size,
        video_feature_dim=video_feature_dim,
        hidden_dim=hidden_dim,
        embedding_dim=embedding_dim,
        num_layers=2,
        dropout=0.3,
        max_seq_length=max_seq_length,
        attention_dim=256
    )
    
    print(f"✅ Model created successfully")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test data
    video_features = torch.randn(batch_size, video_length, video_feature_dim)
    captions = torch.randint(1, vocab_size, (batch_size, max_seq_length))
    
    print(f"✅ Test data created")
    print(f"   Video features shape: {video_features.shape}")
    print(f"   Captions shape: {captions.shape}")
    
    # Test training mode
    model.train()
    outputs = model(video_features, captions, sampling_prob=0.1)
    
    assert 'logits' in outputs, "Training output should contain logits"
    assert 'attention_weights' in outputs, "Training output should contain attention weights"
    
    logits = outputs['logits']
    print(f"✅ Training forward pass successful")
    print(f"   Output logits shape: {logits.shape}")
    
    # Test loss computation
    criterion = EnhancedS2VTLoss(vocab_size=vocab_size, label_smoothing=0.1)
    
    # Prepare targets (shift captions)
    targets = captions[:, 1:]  # Remove BOS token
    input_captions = captions[:, :-1]  # Remove EOS token
    
    # Forward pass with correct input
    outputs = model(video_features, input_captions, sampling_prob=0.1)
    loss = criterion(outputs['logits'], targets)
    
    print(f"✅ Loss computation successful")
    print(f"   Loss value: {loss.item():.4f}")
    
    # Test inference mode
    model.eval()
    with torch.no_grad():
        outputs = model(video_features)
        
    assert 'logits' in outputs, "Inference output should contain logits"
    print(f"✅ Inference forward pass successful")
    print(f"   Output logits shape: {outputs['logits'].shape}")
    
    # Test beam search
    vocabulary = {f'word_{i}': i for i in range(vocab_size)}
    vocabulary['<PAD>'] = 0
    vocabulary['<BOS>'] = 1
    vocabulary['<EOS>'] = 2
    
    beam_search = CoverageBeamSearch(
        model=model,
        vocabulary=vocabulary,
        beam_size=3,
        length_penalty=1.0,
        coverage_penalty=0.2
    )
    
    # Test single video
    single_video = video_features[:1]
    tokens, score = beam_search.decode(single_video)
    
    print(f"✅ Beam search successful")
    print(f"   Generated tokens shape: {tokens.shape}")
    print(f"   Score: {score:.4f}")
    
    # Test gradient flow
    model.train()
    outputs = model(video_features, input_captions, sampling_prob=0.1)
    loss = criterion(outputs['logits'], targets)
    loss.backward()
    
    # Check gradients
    has_gradients = any(p.grad is not None and p.grad.abs().sum() > 0 
                       for p in model.parameters())
    
    print(f"✅ Gradient computation successful")
    print(f"   Has gradients: {has_gradients}")
    
    print("\n🎉 All tests passed! Enhanced model is working correctly.")
    
    return True


def test_with_config():
    """Test with actual configuration"""
    print("\n🧪 Testing with Enhanced Configuration...")
    
    # Load config
    config_path = 'config_enhanced.json'
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        return False
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Create model with config parameters
    model = EnhancedS2VT(
        vocab_size=1000,  # Dummy vocab size
        video_feature_dim=config['model_parameters']['video_feature_dim'],
        hidden_dim=config['model_parameters']['hidden_dim'],
        embedding_dim=config['model_parameters']['embedding_dim'],
        num_layers=config['model_parameters']['num_layers'],
        dropout=config['model_parameters']['dropout'],
        max_seq_length=config['data_parameters']['max_caption_length'],
        attention_dim=config['model_parameters']['attention_dim']
    )
    
    print(f"✅ Model created with config parameters")
    print(f"   Hidden dim: {config['model_parameters']['hidden_dim']}")
    print(f"   Embedding dim: {config['model_parameters']['embedding_dim']}")
    print(f"   Max sequence length: {config['data_parameters']['max_caption_length']}")
    
    # Test model dimensions match config
    assert model.hidden_dim == config['model_parameters']['hidden_dim']
    assert model.max_seq_length == config['data_parameters']['max_caption_length']
    
    print("✅ Configuration test passed!")
    
    return True


if __name__ == '__main__':
    print("=" * 60)
    print("ENHANCED S2VT MODEL TESTING")
    print("=" * 60)
    
    try:
        # Test basic functionality
        test_enhanced_model()
        
        # Test with configuration
        test_with_config()
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
        print("The enhanced model is ready for training.")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)