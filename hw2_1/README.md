# HW2 Seq2Seq Video Caption Generation

This directory contains the HW2 submission for video caption generation using sequence-to-sequence models.

## Files

1. **hw2_seq2seq.sh** - Main execution script
2. **model_seq2seq.py** - Training code implementation  
3. **inference_hw2.py** - Inference script for generating captions
4. **your_seq2seq_model/** - Directory containing trained model

## Usage

### Running Inference (for evaluation)

```bash
./hw2_seq2seq.sh <data_directory> <output_filename>
```

**Examples:**
```bash
./hw2_seq2seq.sh testing_data testset_output.txt
./hw2_seq2seq.sh ta_review_data tareviewset_output.txt
```

**Parameters:**
- `$1`: Data directory containing video features
- `$2`: Output filename (.txt format)

### Training (if needed)

```bash
python model_seq2seq.py --train --data_path /path/to/training/data
```

## Model Architecture

- **S2VT (Sequence to Sequence - Video to Text)**
- Enhanced with attention mechanism
- Scheduled sampling for exposure bias reduction
- Beam search decoding
- BLEU evaluation metrics

## Output Format

The script generates a text file with format:
```
video_id_1,generated caption 1
video_id_2,generated caption 2
...
```

## Requirements

- PyTorch >= 1.8.0
- NumPy >= 1.19.0
- Python >= 3.7

## Model Performance

Target: BLEU@1 > 0.6 on MSVD dataset