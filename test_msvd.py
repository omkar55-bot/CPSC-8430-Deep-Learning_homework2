"""
Test script to verify MSVD dataset loading and S2VT model integration
"""

import torch
import numpy as np
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.msvd_dataset import MSVDDataset, create_msvd_data_loaders, analyze_msvd_dataset
from models.s2vt_model import S2VTModel, S2VTLoss


def test_msvd_dataset_loading():
    """Test MSVD dataset loading"""
    print("Testing MSVD Dataset Loading")
    print("=" * 40)
    
    data_root = "E:/imgsynth/MLDS_hw2_1_data"
    
    if not os.path.exists(data_root):
        print(f"❌ Dataset not found at {data_root}")
        print("Please extract MLDS_hw2_1_data.tar.gz first")
        return False
    
    try:
        # Test dataset analysis
        print("1. Analyzing dataset...")
        train_dataset, test_dataset = analyze_msvd_dataset(data_root)
        print("✅ Dataset analysis completed")
        
        # Test data loaders
        print("\n2. Creating data loaders...")
        train_loader, test_loader, vocabulary = create_msvd_data_loaders(
            data_root=data_root,
            batch_size=4,
            max_caption_length=15,
            max_frames=60,
            num_workers=0  # Use 0 for testing to avoid multiprocessing issues
        )
        print("✅ Data loaders created successfully")
        
        # Test batch loading
        print("\n3. Testing batch loading...")
        for batch in train_loader:
            print(f"✅ Train batch loaded:")
            print(f"   Video features shape: {batch['video_features'].shape}")
            print(f"   Captions shape: {batch['captions'].shape}")
            print(f"   Sample video ID: {batch['video_ids'][0]}")
            print(f"   Sample caption: {batch['caption_texts'][0]}")
            break
        
        for batch in test_loader:
            print(f"\n✅ Test batch loaded:")
            print(f"   Video features shape: {batch['video_features'].shape}")
            print(f"   Captions shape: {batch['captions'].shape}")
            print(f"   Sample video ID: {batch['video_ids'][0]}")
            print(f"   Sample caption: {batch['caption_texts'][0]}")
            break
        
        return True, vocabulary
        
    except Exception as e:
        print(f"❌ Error during dataset testing: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_model_with_msvd_data(vocabulary):
    """Test S2VT model with MSVD data"""
    print("\n\nTesting S2VT Model with MSVD Data")
    print("=" * 40)
    
    try:
        # Create model
        print("1. Creating S2VT model...")
        model = S2VTModel(
            vocab_size=len(vocabulary),
            max_frames=60,
            video_feature_dim=4096,
            hidden_dim=256,  # Smaller for testing
            embedding_dim=256,
            num_layers=2,
            dropout=0.1
        )
        
        print(f"✅ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Test forward pass
        print("\n2. Testing forward pass...")
        batch_size = 2
        num_frames = 60
        feature_dim = 4096
        caption_length = 15
        
        # Create dummy data
        video_features = torch.randn(batch_size, num_frames, feature_dim)
        captions = torch.randint(1, len(vocabulary)-1, (batch_size, caption_length))
        
        # Ensure proper token structure (BOS at start)
        captions[:, 0] = vocabulary.word2idx[vocabulary.BOS_TOKEN]
        
        # Forward pass
        model.train()
        outputs = model(video_features, captions)
        
        print(f"✅ Forward pass successful:")
        print(f"   Input video shape: {video_features.shape}")
        print(f"   Input captions shape: {captions.shape}")
        print(f"   Output shape: {outputs.shape}")
        
        # Test loss calculation
        print("\n3. Testing loss calculation...")
        criterion = S2VTLoss(len(vocabulary))
        targets = captions[:, 1:]  # Remove BOS token
        outputs_matched = outputs[:, :targets.size(1), :]
        loss = criterion(outputs_matched, targets)
        
        print(f"✅ Loss calculation successful: {loss.item():.4f}")
        
        # Test inference mode
        print("\n4. Testing inference mode...")
        model.eval()
        with torch.no_grad():
            single_video = video_features[:1]  # Take first video
            inference_outputs = model(single_video, max_length=15)
            predicted_ids = inference_outputs.argmax(dim=-1).squeeze(0)
            
            # Decode caption
            predicted_caption = vocabulary.decode_caption(predicted_ids.numpy().tolist())
            
        print(f"✅ Inference successful:")
        print(f"   Generated caption: '{predicted_caption}'")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during model testing: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_integration():
    """Test full integration with real MSVD data"""
    print("\n\nTesting Full Integration")
    print("=" * 40)
    
    data_root = "E:/imgsynth/MLDS_hw2_1_data"
    
    try:
        # Load real data
        print("1. Loading real MSVD data...")
        train_loader, test_loader, vocabulary = create_msvd_data_loaders(
            data_root=data_root,
            batch_size=2,
            max_caption_length=20,
            max_frames=80,
            num_workers=0
        )
        
        # Create model
        print("2. Creating model for real data...")
        model = S2VTModel(
            vocab_size=len(vocabulary),
            max_frames=80,
            video_feature_dim=4096,
            hidden_dim=256,
            embedding_dim=256,
            num_layers=2,
            dropout=0.1
        )
        
        criterion = S2VTLoss(len(vocabulary))
        
        # Test training step
        print("3. Testing training step with real data...")
        model.train()
        
        for batch in train_loader:
            video_features = batch['video_features']
            captions = batch['captions']
            
            # Forward pass
            outputs = model(video_features, captions)
            
            # Loss calculation
            targets = captions[:, 1:]
            outputs_matched = outputs[:, :targets.size(1), :]
            loss = criterion(outputs_matched, targets)
            
            print(f"✅ Training step successful:")
            print(f"   Batch size: {video_features.shape[0]}")
            print(f"   Loss: {loss.item():.4f}")
            print(f"   Video IDs: {batch['video_ids']}")
            print(f"   Sample captions: {batch['caption_texts'][0]}")
            
            break
        
        # Test inference step
        print("\n4. Testing inference step with real data...")
        model.eval()
        
        with torch.no_grad():
            for batch in test_loader:
                video_features = batch['video_features']
                
                # Generate captions
                outputs = model(video_features, max_length=20)
                predicted_ids = outputs.argmax(dim=-1)
                
                # Decode captions
                for i in range(video_features.shape[0]):
                    pred_caption = vocabulary.decode_caption(predicted_ids[i].numpy().tolist())
                    ref_captions = batch['caption_texts'][i]
                    
                    print(f"✅ Inference for video {batch['video_ids'][i]}:")
                    print(f"   Generated: '{pred_caption}'")
                    if isinstance(ref_captions, list):
                        print(f"   Reference: '{ref_captions[0]}'")
                    else:
                        print(f"   Reference: '{ref_captions}'")
                    print()
                
                break
        
        print("✅ Full integration test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error during integration testing: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("MSVD Dataset and S2VT Model Integration Test")
    print("=" * 60)
    
    # Test 1: Dataset loading
    success, vocabulary = test_msvd_dataset_loading()
    if not success:
        print("❌ Dataset loading test failed. Stopping.")
        return
    
    # Test 2: Model with dummy data
    success = test_model_with_msvd_data(vocabulary)
    if not success:
        print("❌ Model testing failed. Stopping.")
        return
    
    # Test 3: Full integration
    success = test_full_integration()
    if not success:
        print("❌ Integration test failed.")
        return
    
    print("\n" + "=" * 60)
    print("🎉 ALL TESTS PASSED! 🎉")
    print("=" * 60)
    print("\nYou can now:")
    print("1. Train the model: python train_msvd.py")
    print("2. Run inference: python inference_msvd.py --demo --model_path <model.pth>")
    print("3. Evaluate on test set: python inference_msvd.py --evaluate --model_path <model.pth>")


if __name__ == "__main__":
    main()