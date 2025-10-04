"""
Setup script for S2VT Video Caption Generation project
Run this script to verify installation and setup
"""

import subprocess
import sys
import os
import torch
import numpy as np


def check_python_version():
    """Check Python version"""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print("⚠️  Warning: Python 3.7+ is recommended")
        return False
    else:
        print("✅ Python version is compatible")
        return True


def check_dependencies():
    """Check if required dependencies are installed"""
    print("\nChecking dependencies...")
    
    dependencies = {
        'torch': torch,
        'numpy': np,
    }
    
    try:
        import torchvision
        dependencies['torchvision'] = torchvision
    except ImportError:
        print("⚠️  torchvision not found - install with: pip install torchvision")
    
    try:
        import tqdm
        dependencies['tqdm'] = tqdm
    except ImportError:
        print("⚠️  tqdm not found - install with: pip install tqdm")
    
    try:
        import matplotlib
        dependencies['matplotlib'] = matplotlib
    except ImportError:
        print("⚠️  matplotlib not found - install with: pip install matplotlib")
    
    for name, module in dependencies.items():
        print(f"✅ {name}: {getattr(module, '__version__', 'version unknown')}")
    
    return True


def check_pytorch_setup():
    """Check PyTorch setup and CUDA availability"""
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CPU-only mode (training will be slower)")
    
    # Test basic tensor operations
    try:
        x = torch.randn(2, 3)
        y = torch.randn(3, 2)
        z = torch.mm(x, y)
        print("✅ Basic tensor operations working")
    except Exception as e:
        print(f"❌ Tensor operations failed: {e}")
        return False
    
    return True


def create_directories():
    """Create necessary directories"""
    print("\nCreating directories...")
    
    directories = [
        'data',
        'checkpoints',
        'results',
        'logs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created/verified directory: {directory}")


def test_model_creation():
    """Test model creation with small parameters"""
    print("\nTesting model creation...")
    
    try:
        # Import model
        from models.s2vt_model import S2VTModel
        
        # Create small model for testing
        model = S2VTModel(
            vocab_size=1000,
            max_frames=10,
            video_feature_dim=512,
            hidden_dim=64,
            embedding_dim=64,
            num_layers=2,
            dropout=0.1
        )
        
        print(f"✅ Model created successfully with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Test forward pass
        batch_size = 2
        num_frames = 10
        feature_dim = 512
        
        video_features = torch.randn(batch_size, num_frames, feature_dim)
        captions = torch.randint(0, 1000, (batch_size, 15))
        
        outputs = model(video_features, captions)
        print(f"✅ Forward pass successful, output shape: {outputs.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model creation/testing failed: {e}")
        return False


def test_data_preprocessing():
    """Test data preprocessing functionality"""
    print("\nTesting data preprocessing...")
    
    try:
        from data.preprocessing import Vocabulary, VideoCaptionDataset
        
        # Test vocabulary
        vocab = Vocabulary()
        sample_captions = [
            "a man is walking",
            "a woman is cooking",
            "children are playing"
        ]
        
        vocab.build_vocabulary(sample_captions, min_count=1)
        print(f"✅ Vocabulary created with {len(vocab)} words")
        
        # Test encoding/decoding
        encoded = vocab.encode_caption(sample_captions[0])
        decoded = vocab.decode_caption(encoded)
        print(f"✅ Encoding/decoding working: '{sample_captions[0]}' -> '{decoded}'")
        
        return True
        
    except Exception as e:
        print(f"❌ Data preprocessing test failed: {e}")
        return False


def install_requirements():
    """Install requirements if needed"""
    print("\nChecking if requirements installation is needed...")
    
    requirements_file = "requirements.txt"
    if os.path.exists(requirements_file):
        try:
            # Check if we need to install anything
            result = subprocess.run([sys.executable, "-m", "pip", "check"], 
                                  capture_output=True, text=True)
            
            if result.returncode != 0:
                print("Some packages may be missing. Install with:")
                print(f"pip install -r {requirements_file}")
            else:
                print("✅ All requirements appear to be satisfied")
                
        except Exception as e:
            print(f"Could not check requirements: {e}")
    else:
        print(f"⚠️  {requirements_file} not found")


def run_example():
    """Ask user if they want to run the example"""
    print("\n" + "="*50)
    print("SETUP COMPLETE!")
    print("="*50)
    
    response = input("\nWould you like to run the example script? (y/n): ").lower().strip()
    
    if response in ['y', 'yes']:
        print("\nRunning example script...")
        try:
            import example
            example.main()
        except Exception as e:
            print(f"Example script failed: {e}")
            print("You can run it manually with: python example.py")
    else:
        print("\nYou can run the example later with: python example.py")


def main():
    """Main setup function"""
    print("S2VT VIDEO CAPTION GENERATION - SETUP")
    print("="*50)
    
    success = True
    
    # Run all checks
    success &= check_python_version()
    success &= check_dependencies() 
    success &= check_pytorch_setup()
    
    if success:
        create_directories()
        success &= test_model_creation()
        success &= test_data_preprocessing()
        install_requirements()
        
        if success:
            run_example()
        else:
            print("\n❌ Some tests failed. Please check the errors above.")
    else:
        print("\n❌ Setup failed. Please install missing dependencies.")
        print("\nInstall PyTorch: https://pytorch.org/get-started/locally/")
        print("Install other dependencies: pip install -r requirements.txt")


if __name__ == "__main__":
    main()