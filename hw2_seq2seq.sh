#!/bin/bash

# HW2 Seq2Seq Video Caption Generation Script
# Usage: ./hw2_seq2seq.sh <data_directory> <output_filename>
# Example: ./hw2_seq2seq.sh testing_data testset_output.txt

# Check if correct number of arguments provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <data_directory> <output_filename>"
    echo "Example: $0 testing_data testset_output.txt"
    echo "Example: $0 ta_review_data tareviewset_output.txt"
    exit 1
fi

# Get arguments
DATA_DIR=$1
OUTPUT_FILE=$2

echo "Starting HW2 Seq2Seq Video Caption Generation..."
echo "Data directory: $DATA_DIR"
echo "Output file: $OUTPUT_FILE"

# Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory '$DATA_DIR' does not exist!"
    exit 1
fi

# Run the Python inference script
python inference_hw2.py \
    --data_dir "$DATA_DIR" \
    --output_file "$OUTPUT_FILE" \
    --model_path "your_seq2seq_model/best_model_enhanced.pth" \
    --vocab_path "your_seq2seq_model/vocabulary.pkl"

# Check if the script executed successfully
if [ $? -eq 0 ]; then
    echo "Video caption generation completed successfully!"
    echo "Output saved to: $OUTPUT_FILE"
else
    echo "Error: Video caption generation failed!"
    exit 1
fi