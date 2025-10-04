#!/bin/bash

# HW2 Seq2Seq Video Caption Generation Script
# Usage: ./hw2_seq2seq.sh <data_directory> <output_filename>
# Example: ./hw2_seq2seq.sh testing_data testset_output.txt

# Check if correct number of arguments provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <data_directory> <output_filename>"
    echo "Example: $0 testing_data testset_output.txt"
    exit 1
fi

# Get arguments
DATA_DIR=$1
OUTPUT_FILE=$2

echo "=== HW2 Seq2Seq Video Caption Generation ==="
echo "Data directory: $DATA_DIR"
echo "Output file: $OUTPUT_FILE"

# Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory '$DATA_DIR' does not exist!"
    exit 1
fi

# Check if model exists
MODEL_PATH="your_seq2seq_model/best_model_enhanced.pth"
if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file not found at '$MODEL_PATH'"
    echo "Please ensure your trained model is placed in the your_seq2seq_model/ directory"
    exit 1
fi

# Run inference
echo "Running video caption generation..."
python inference_hw2.py \
    --data_dir "$DATA_DIR" \
    --output_file "$OUTPUT_FILE" \
    --model_path "$MODEL_PATH"

# Check if successful
if [ $? -eq 0 ]; then
    echo "✅ Video caption generation completed successfully!"
    echo "📁 Results saved to: $OUTPUT_FILE"
    
    # Show first few results
    if [ -f "$OUTPUT_FILE" ]; then
        echo ""
        echo "📋 Sample results (first 5 lines):"
        head -5 "$OUTPUT_FILE"
        echo ""
        echo "📊 Total videos processed: $(wc -l < "$OUTPUT_FILE")"
    fi
else
    echo "❌ Error: Video caption generation failed!"
    exit 1
fi