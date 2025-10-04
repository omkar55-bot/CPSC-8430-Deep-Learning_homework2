#!/bin/bash

# Test script for HW2 submission
# Run this script to test your submission before uploading

echo "🧪 Testing HW2 Submission..."

# Check if required files exist
echo "Checking required files..."

required_files=(
    "hw2_seq2seq.sh"
    "model_seq2seq.py" 
    "inference_hw2.py"
    "models/enhanced_s2vt.py"
    "data/preprocessing.py"
    "utils/metrics.py"
)

missing_files=0
for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file"
    else
        echo "❌ $file (missing)"
        missing_files=$((missing_files + 1))
    fi
done

if [ $missing_files -gt 0 ]; then
    echo "❌ $missing_files required files are missing!"
    exit 1
fi

echo "✅ All required files present"

# Check if model directory exists
if [ -d "your_seq2seq_model" ]; then
    echo "✅ Model directory exists"
    
    # Check for model files
    if [ -f "your_seq2seq_model/best_model_enhanced.pth" ]; then
        echo "✅ Model file found"
    else
        echo "⚠️  Model file not found (will be added after training)"
    fi
else
    echo "⚠️  Model directory not found (will be created)"
    mkdir -p your_seq2seq_model
fi

# Check script syntax
echo "Checking Python syntax..."
python -m py_compile model_seq2seq.py
if [ $? -eq 0 ]; then
    echo "✅ model_seq2seq.py syntax OK"
else
    echo "❌ model_seq2seq.py has syntax errors"
    exit 1
fi

python -m py_compile inference_hw2.py
if [ $? -eq 0 ]; then
    echo "✅ inference_hw2.py syntax OK" 
else
    echo "❌ inference_hw2.py has syntax errors"
    exit 1
fi

# Check shell script
if [ -f "hw2_seq2seq.sh" ]; then
    chmod +x hw2_seq2seq.sh
    echo "✅ hw2_seq2seq.sh is executable"
else
    echo "❌ hw2_seq2seq.sh not found"
    exit 1
fi

echo ""
echo "🎉 HW2 submission validation completed!"
echo ""
echo "📋 Submission checklist:"
echo "✅ Required files present"
echo "✅ Python syntax valid"
echo "✅ Shell script executable"
echo "⏳ Model training (in progress on other device)"
echo ""
echo "🚀 Once model training is complete:"
echo "1. Copy trained model to your_seq2seq_model/"
echo "2. Update model path in hw2_seq2seq.sh"
echo "3. Test with: ./hw2_seq2seq.sh testing_data testset_output.txt"
echo "4. Upload to GitHub in hw2/hw2_1/ directory"