# HW2 Submission Setup Script
# This script helps prepare your submission according to the requirements

import os
import shutil

def create_hw2_submission():
    """Create HW2 submission structure"""
    
    print("🎯 Setting up HW2 submission structure...")
    
    # Create main hw2 directory
    hw2_dir = "hw2"
    if os.path.exists(hw2_dir):
        print(f"Directory {hw2_dir} already exists")
    else:
        os.makedirs(hw2_dir)
        print(f"Created directory: {hw2_dir}")
    
    # Create hw2_1 subdirectory
    hw2_1_dir = os.path.join(hw2_dir, "hw2_1")
    if os.path.exists(hw2_1_dir):
        print(f"Directory {hw2_1_dir} already exists")
    else:
        os.makedirs(hw2_1_dir)
        print(f"Created directory: {hw2_1_dir}")
    
    # Copy required files
    files_to_copy = [
        ("hw2_seq2seq.sh", "hw2_seq2seq.sh"),
        ("model_seq2seq.py", "model_seq2seq.py"),  
        ("inference_hw2.py", "inference_hw2.py"),
        ("models/enhanced_s2vt.py", "models/enhanced_s2vt.py"),
        ("models/attention_s2vt.py", "models/attention_s2vt.py"),
        ("data/preprocessing.py", "data/preprocessing.py"),
        ("data/msvd_dataset.py", "data/msvd_dataset.py"),
        ("utils/metrics.py", "utils/metrics.py"),
        ("config_enhanced.json", "config_enhanced.json"),
        ("README.md", "README.md")
    ]
    
    for source, dest in files_to_copy:
        source_path = source
        dest_path = os.path.join(hw2_1_dir, dest)
        
        # Create subdirectories if needed
        dest_dir = os.path.dirname(dest_path)
        if dest_dir and not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        
        if os.path.exists(source_path):
            shutil.copy2(source_path, dest_path)
            print(f"Copied: {source} -> {dest_path}")
        else:
            print(f"Warning: Source file not found: {source}")
    
    # Create model placeholder directory
    model_dir = os.path.join(hw2_1_dir, "your_seq2seq_model")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"Created model directory: {model_dir}")
    
    # Create placeholder files
    placeholder_files = [
        (os.path.join(model_dir, "README.txt"), 
         "Place your trained model files here:\n- best_model_enhanced.pth\n- vocabulary.pkl\n"),
        (os.path.join(hw2_1_dir, "requirements.txt"),
         "torch>=1.8.0\nnumpy>=1.19.0\ntorchvision>=0.9.0\n")
    ]
    
    for filepath, content in placeholder_files:
        if not os.path.exists(filepath):
            with open(filepath, 'w') as f:
                f.write(content)
            print(f"Created: {filepath}")
    
    print("\n✅ HW2 submission structure created successfully!")
    print(f"\nYour submission structure:")
    print(f"hw2/")
    print(f"└── hw2_1/")
    print(f"    ├── hw2_seq2seq.sh          # Main execution script")
    print(f"    ├── model_seq2seq.py        # Training code")
    print(f"    ├── inference_hw2.py        # Inference code")
    print(f"    ├── your_seq2seq_model/     # Place your trained model here")
    print(f"    ├── models/                 # Model implementations")
    print(f"    ├── data/                   # Data processing")
    print(f"    ├── utils/                  # Utilities")
    print(f"    └── README.md               # Documentation")
    
    print(f"\n📝 Next steps:")
    print(f"1. Copy your trained model to: {model_dir}/")
    print(f"2. Update the model path in hw2_seq2seq.sh")
    print(f"3. Test the script with: ./hw2_seq2seq.sh testing_data testset_output.txt")
    print(f"4. Upload hw2/ directory to GitHub")

if __name__ == "__main__":
    create_hw2_submission()