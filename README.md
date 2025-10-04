# S2VT Video Caption Generation

This project implements the S2VT (Sequence to Sequence - Video to Text) model for automatic video caption generation, based on the paper:

**"Sequence to Sequence -- Video to Text"** by Subhashini Venugopalan et al.
[Paper Link](http://www.cs.utexas.edu/users/ml/papers/venugopalan.iccv15.pdf)

## Overview

The S2VT model addresses the challenge of generating natural language descriptions for video content. It uses a sequence-to-sequence architecture with two LSTM layers:

- **Encoder LSTM**: Processes video features extracted from CNN (e.g., VGG, ResNet)
- **Decoder LSTM**: Generates text captions word by word

### Key Features

- **Two-layer LSTM architecture** as described in the S2VT paper
- **Handles variable-length input/output** sequences
- **Multiple generation methods**: Greedy decoding and beam search
- **Comprehensive evaluation metrics**: BLEU, METEOR, CIDEr, ROUGE-L
- **Flexible data preprocessing** with vocabulary management
- **Special token handling**: PAD, BOS, EOS, UNK tokens

### Challenges Addressed

1. **Different attributes of video** (objects, actions, scenes)
2. **Variable length I/O** sequences
3. **Temporal understanding** of video content
4. **Language generation** quality

## Project Structure

```
video_caption_generation/
├── models/
│   ├── __init__.py
│   └── s2vt_model.py          # S2VT model implementation
├── data/
│   ├── __init__.py
│   └── preprocessing.py        # Data preprocessing utilities
├── utils/
│   ├── __init__.py
│   ├── metrics.py             # Evaluation metrics
│   └── evaluation.py          # Model evaluation utilities
├── train.py                   # Training script
├── inference.py               # Inference script
├── config.json               # Configuration file
├── requirements.txt          # Dependencies
└── README.md                # This file
```

## Installation

1. **Clone or create the project directory**
2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

3. **Verify PyTorch installation:**

```bash
python -c "import torch; print(torch.__version__)"
```

## Usage

### 1. Data Preparation

The model expects video features to be pre-extracted using CNN models. You need:

- **Video features**: Shape `(num_videos, num_frames, feature_dim)`
- **Captions**: List of corresponding text descriptions

Example data loading function is provided in `data/preprocessing.py`.

### 2. Training

#### Basic Training:

```bash
python train.py
```

#### Training with custom configuration:

```bash
python train.py --config config.json
```

#### Resume from checkpoint:

```bash
python train.py --resume checkpoints/checkpoint_epoch_10.pth
```

### 3. Inference

#### Run demo with sample data:

```bash
python inference.py --model_path checkpoints/best_model.pth --vocab_path checkpoints/vocabulary.pkl --demo
```

#### Generate captions for your videos:

```bash
python inference.py \
    --model_path checkpoints/best_model.pth \
    --vocab_path checkpoints/vocabulary.pkl \
    --video_features path/to/features.npy \
    --output_path results.json \
    --method beam_search
```

#### Evaluate on test data:

```bash
python inference.py \
    --model_path checkpoints/best_model.pth \
    --vocab_path checkpoints/vocabulary.pkl \
    --video_features test_features.npy \
    --reference_captions test_captions.json
```

## Model Architecture

### S2VT Model Details

The S2VT model follows a two-stage process:

#### Stage 1: Encoding
- Video frames are processed through a CNN (VGG16, ResNet, etc.) to extract features
- Features are passed through the first LSTM layer
- Text positions are padded during this stage

#### Stage 2: Decoding
- The second LSTM layer generates text tokens
- Video positions are padded during this stage
- Special tokens (BOS, EOS) guide the generation process

### Key Components

1. **Video Feature Projection**: Maps CNN features to LSTM input dimension
2. **Word Embedding**: Converts tokens to dense vectors
3. **Two-layer LSTM**: Core sequence-to-sequence processing
4. **Output Projection**: Maps LSTM output to vocabulary distribution

### Training Process

1. **Data Preprocessing**:
   - Build vocabulary from training captions
   - Encode captions with special tokens
   - Pad/truncate sequences to fixed length

2. **Loss Calculation**:
   - Cross-entropy loss on predicted vs. ground truth tokens
   - Ignore padding tokens in loss computation

3. **Optimization**:
   - Adam optimizer with learning rate scheduling
   - Gradient clipping for stability
   - Early stopping based on validation loss

## Configuration

The `config.json` file contains all hyperparameters:

```json
{
  "model_parameters": {
    "video_feature_dim": 4096,    # CNN feature dimension
    "hidden_dim": 512,            # LSTM hidden size
    "embedding_dim": 512,         # Word embedding size
    "num_layers": 2,              # Number of LSTM layers
    "dropout": 0.5                # Dropout rate
  },
  "training_parameters": {
    "batch_size": 32,
    "learning_rate": 0.0001,
    "num_epochs": 50,
    "grad_clip": 5.0
  }
}
```

## Evaluation Metrics

The model is evaluated using standard video captioning metrics:

- **BLEU-1, BLEU-2, BLEU-3, BLEU-4**: N-gram overlap with references
- **METEOR**: Alignment-based metric considering synonyms
- **CIDEr**: Consensus-based metric using TF-IDF weighting
- **ROUGE-L**: Longest common subsequence metric

## Advanced Features

### 🎯 Baseline Training

Achieve **BLEU@1 = 0.6** following the exact training tips:

```bash
# Train with baseline configuration
python train_baseline.py --config config_baseline.json

# Expected results:
# - BLEU@1: 0.60 (Captions Avg.)
# - Training epochs: 200
# - LSTM dimension: 256
# - Learning rate: 0.001
# - Training time: ~72 minutes on GTX 960
```

### 🚀 Enhanced Training (NEW!)

**Beat the baseline** with focused improvements:

```bash
# Train with enhanced model to beat baseline
python train_enhanced.py --config config_enhanced.json

# Key improvements:
# - Coverage attention mechanism
# - Better weight initialization
# - Label smoothing (0.1)
# - Enhanced beam search with coverage penalty
# - Improved regularization
# 
# Expected results: BLEU@1 > 0.6
```

### 🔍 Attention Mechanism

Visual attention allows the model to focus on different video regions:

```python
# Create attention-based model
model = AttentionS2VT(
    vocab_size=vocab_size,
    attention_dim=256,
    ...
)
```

### 📚 Scheduled Sampling

Solves exposure bias problem during training:

```python
# Enable scheduled sampling
model = ScheduledSamplingS2VT(...)
model.set_sampling_prob(0.25)  # 25% model prediction, 75% ground truth
```

### 🚀 Enhanced Beam Search

Improved beam search with length penalty and attention visualization:

```python
beam_decoder = BeamSearchDecoder(
    model=model,
    vocabulary=vocab,
    beam_size=5,
    length_penalty=1.0
)
predicted_ids, score, attention = beam_decoder.decode(video_features)
```

### 📊 Comprehensive Evaluation

Baseline-compliant evaluation following training tips:

```bash
# Evaluate with baseline metrics
python evaluate_baseline.py --model_path model.pth

# Generates:
# - BLEU@1 (Captions Avg.) - matches training tips calculation
# - Standard metrics (BLEU, METEOR, CIDEr, ROUGE-L)
# - Error analysis and visualizations
```

## Example Output

```
Video 1:
  Reference: a man is walking down the street
  Predicted: a person is walking on the road

Video 2:
  Reference: children are playing in the park
  Predicted: kids are playing outside

Evaluation Results:
BLEU-1: 0.7523
BLEU-4: 0.3421
METEOR: 0.2876
CIDEr: 0.8934
ROUGE-L: 0.5432
```

## Common Issues and Solutions

### 1. CUDA Out of Memory
- Reduce batch size in config
- Use gradient accumulation
- Process videos in smaller batches

### 2. Poor Caption Quality
- Increase training epochs
- Use beam search instead of greedy decoding
- Check data quality and preprocessing

### 3. Slow Training
- Use GPU acceleration
- Optimize data loading (increase num_workers)
- Use mixed precision training

## Extensions and Improvements

1. **Attention Mechanism**: Add visual attention for better focus
2. **Pretrained Embeddings**: Use GloVe or Word2Vec embeddings
3. **Advanced Architectures**: Transformer-based models
4. **Multi-modal Features**: Combine visual and audio features
5. **Hierarchical Models**: Handle longer videos with hierarchical processing

## Citation

If you use this implementation, please cite the original S2VT paper:

```bibtex
@inproceedings{venugopalan2015sequence,
  title={Sequence to sequence-video to text},
  author={Venugopalan, Subhashini and Rohrbach, Marcus and Donahue, Jeffrey and Mooney, Raymond and Darrell, Trevor and Saenko, Kate},
  booktitle={Proceedings of the IEEE international conference on computer vision},
  pages={4534--4542},
  year={2015}
}
```

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is for educational and research purposes.