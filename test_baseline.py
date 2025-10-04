"""
Test script for baseline S2VT implementation with advanced training tips
"""

import torch
import numpy as np
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.attention_s2vt import AttentionS2VT, ScheduledSamplingS2VT, BeamSearchDecoder
from data.msvd_dataset import create_msvd_data_loaders
from data.preprocessing import Vocabulary


def test_attention_model():
    """Test attention-based S2VT model"""
    print("Testing Attention S2VT Model")
    print("=" * 40)
    
    # Create dummy vocabulary
    vocab = Vocabulary()
    vocab.word2idx = {'<PAD>': 0, '<BOS>': 1, '<EOS>': 2, '<UNK>': 3}
    for i, word in enumerate(['a', 'man', 'is', 'walking', 'woman', 'cooking'], 4):
        vocab.word2idx[word] = i
    vocab.idx2word = {v: k for k, v in vocab.word2idx.items()}
    
    try:
        # Create model
        model = AttentionS2VT(
            vocab_size=len(vocab),
            max_frames=60,
            video_feature_dim=4096,
            hidden_dim=256,
            embedding_dim=256,
            num_layers=2,
            dropout=0.1,
            attention_dim=128
        )
        
        print(f"✅ Attention model created with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Test forward pass
        batch_size = 2
        video_features = torch.randn(batch_size, 60, 4096)
        captions = torch.tensor([[1, 4, 5, 6, 2], [1, 7, 8, 4, 2]])  # [BOS, words..., EOS]
        
        model.train()
        outputs, attention_weights = model(video_features, captions)
        
        print(f"✅ Training forward pass successful:")
        print(f"   Output shape: {outputs.shape}")
        print(f"   Attention weights shape: {attention_weights.shape}")
        
        # Test inference
        model.eval()
        with torch.no_grad():
            inference_outputs, inference_attention = model(video_features[:1], max_length=10)
        
        print(f"✅ Inference forward pass successful:")
        print(f"   Output shape: {inference_outputs.shape}")
        print(f"   Attention weights shape: {inference_attention.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Attention model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_scheduled_sampling():
    """Test scheduled sampling model"""
    print("\nTesting Scheduled Sampling S2VT Model")
    print("=" * 40)
    
    # Create dummy vocabulary
    vocab = Vocabulary()
    vocab.word2idx = {'<PAD>': 0, '<BOS>': 1, '<EOS>': 2, '<UNK>': 3}
    for i, word in enumerate(['a', 'man', 'is', 'walking', 'woman', 'cooking'], 4):
        vocab.word2idx[word] = i
    vocab.idx2word = {v: k for k, v in vocab.word2idx.items()}
    
    try:
        # Create model
        model = ScheduledSamplingS2VT(
            vocab_size=len(vocab),
            max_frames=60,
            video_feature_dim=4096,
            hidden_dim=256,
            embedding_dim=256,
            num_layers=2,
            dropout=0.1
        )
        
        print(f"✅ Scheduled sampling model created")
        
        # Test with different sampling probabilities
        batch_size = 2
        video_features = torch.randn(batch_size, 60, 4096)
        captions = torch.tensor([[1, 4, 5, 6, 2], [1, 7, 8, 4, 2]])
        
        for sampling_prob in [0.0, 0.25, 0.5]:
            model.set_sampling_prob(sampling_prob)
            model.train()
            
            outputs, attention_weights = model(video_features, captions)
            print(f"✅ Sampling prob {sampling_prob}: Output shape {outputs.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Scheduled sampling test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_beam_search():
    """Test beam search decoder"""
    print("\nTesting Beam Search Decoder")
    print("=" * 40)
    
    # Create dummy vocabulary
    vocab = Vocabulary()
    vocab.word2idx = {'<PAD>': 0, '<BOS>': 1, '<EOS>': 2, '<UNK>': 3}
    for i, word in enumerate(['a', 'man', 'is', 'walking', 'woman', 'cooking'], 4):
        vocab.word2idx[word] = i
    vocab.idx2word = {v: k for k, v in vocab.word2idx.items()}
    
    try:
        # Create model
        model = AttentionS2VT(
            vocab_size=len(vocab),
            max_frames=60,
            video_feature_dim=4096,
            hidden_dim=128,  # Smaller for faster testing
            embedding_dim=128,
            num_layers=2,
            dropout=0.0
        )
        
        # Create beam search decoder
        beam_decoder = BeamSearchDecoder(
            model=model,
            vocabulary=vocab,
            beam_size=3,
            max_length=10
        )
        
        print(f"✅ Beam search decoder created")
        
        # Test decoding
        model.eval()
        video_features = torch.randn(1, 60, 4096)
        
        with torch.no_grad():
            predicted_ids, score, attention_weights = beam_decoder.decode(video_features)
        
        print(f"✅ Beam search decoding successful:")
        print(f"   Predicted sequence: {predicted_ids}")
        print(f"   Score: {score:.4f}")
        print(f"   Sequence length: {len(predicted_ids)}")
        
        # Decode to text
        decoded_text = vocab.decode_caption(predicted_ids)
        print(f"   Decoded text: '{decoded_text}'")
        
        return True
        
    except Exception as e:
        print(f"❌ Beam search test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_baseline_configuration():
    """Test baseline configuration loading"""
    print("\nTesting Baseline Configuration")
    print("=" * 40)
    
    try:
        from train_baseline import get_baseline_config
        
        config = get_baseline_config()
        
        print("✅ Baseline configuration loaded:")
        print(f"   Model type: {config['model_type']}")
        print(f"   Hidden dim: {config['hidden_dim']} (baseline: 256)")
        print(f"   Learning rate: {config['learning_rate']} (baseline: 0.001)")
        print(f"   Epochs: {config['num_epochs']} (baseline: 200)")
        print(f"   Vocab min count: {config['vocab_min_count']} (baseline: 3)")
        print(f"   Scheduled sampling: {config['use_scheduled_sampling']}")
        
        # Verify baseline parameters match training tips
        assert config['hidden_dim'] == 256, f"Hidden dim should be 256, got {config['hidden_dim']}"
        assert config['learning_rate'] == 0.001, f"LR should be 0.001, got {config['learning_rate']}"
        assert config['num_epochs'] == 200, f"Epochs should be 200, got {config['num_epochs']}"
        assert config['vocab_min_count'] == 3, f"Min count should be 3, got {config['vocab_min_count']}"
        
        print("✅ All baseline parameters match training tips requirements")
        
        return True
        
    except Exception as e:
        print(f"❌ Baseline configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_msvd_integration():
    """Test MSVD dataset integration with new models"""
    print("\nTesting MSVD Integration with Advanced Models")
    print("=" * 40)
    
    data_root = "E:/imgsynth/MLDS_hw2_1_data"
    
    if not os.path.exists(data_root):
        print(f"⚠️  Dataset not found at {data_root}")
        print("Skipping MSVD integration test")
        return True
    
    try:
        # Create data loaders
        train_loader, test_loader, vocabulary = create_msvd_data_loaders(
            data_root=data_root,
            batch_size=4,
            max_caption_length=15,
            max_frames=60,
            num_workers=0
        )
        
        print(f"✅ MSVD data loaders created")
        print(f"   Vocabulary size: {len(vocabulary)}")
        
        # Test with attention model
        model = AttentionS2VT(
            vocab_size=len(vocabulary),
            max_frames=60,
            video_feature_dim=4096,
            hidden_dim=128,  # Small for testing
            embedding_dim=128,
            num_layers=2,
            dropout=0.1
        )
        
        # Test training step
        model.train()
        for batch in train_loader:
            video_features = batch['video_features']
            captions = batch['captions']
            
            outputs, attention_weights = model(video_features, captions)
            
            print(f"✅ MSVD training step successful:")
            print(f"   Batch size: {video_features.shape[0]}")
            print(f"   Output shape: {outputs.shape}")
            print(f"   Attention shape: {attention_weights.shape}")
            
            break
        
        return True
        
    except Exception as e:
        print(f"❌ MSVD integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all baseline implementation tests"""
    print("BASELINE S2VT IMPLEMENTATION TESTS")
    print("=" * 60)
    
    tests = [
        ("Attention Model", test_attention_model),
        ("Scheduled Sampling", test_scheduled_sampling),
        ("Beam Search", test_beam_search),
        ("Baseline Configuration", test_baseline_configuration),
        ("MSVD Integration", test_msvd_integration)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
    
    print(f"\n{'='*60}")
    print(f"TEST SUMMARY: {passed}/{total} tests passed")
    print(f"{'='*60}")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Ready for baseline training!")
        print("\nNext steps:")
        print("1. Train with baseline config: python train_baseline.py --config config_baseline.json")
        print("2. Target: BLEU@1 = 0.6 (Captions Avg.) in 200 epochs")
        print("3. Expected training time: ~72 minutes on GTX 960")
        print("4. Evaluate with: python evaluate_baseline.py --model_path <model.pth>")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        print("Make sure all dependencies are installed and the dataset is available.")


if __name__ == "__main__":
    main()