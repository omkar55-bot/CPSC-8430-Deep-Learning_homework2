"""
Evaluation utilities for S2VT model inference and testing
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple
from tqdm import tqdm
from collections import Counter

from .metrics import evaluate_captions


def generate_caption(model, video_features, vocabulary, max_length=20, method='greedy'):
    """
    Generate caption for a single video
    
    Args:
        model: Trained S2VT model
        video_features: Video features tensor (1, num_frames, feature_dim)
        vocabulary: Vocabulary object
        max_length: Maximum caption length
        method: Generation method ('greedy' or 'beam_search')
        
    Returns:
        Generated caption string
    """
    model.eval()
    device = next(model.parameters()).device
    
    with torch.no_grad():
        video_features = video_features.to(device)
        
        if method == 'greedy':
            outputs = model(video_features, max_length=max_length)
            predicted_ids = outputs.argmax(dim=-1).squeeze(0)
        elif method == 'beam_search':
            predicted_ids = beam_search_decode(model, video_features, vocabulary, max_length)
        else:
            raise ValueError(f"Unknown generation method: {method}")
        
        # Convert to caption
        caption = vocabulary.decode_caption(predicted_ids.cpu().numpy().tolist())
    
    return caption


def beam_search_decode(model, video_features, vocabulary, max_length=20, beam_size=5):
    """
    Beam search decoding for caption generation
    
    Args:
        model: Trained S2VT model
        video_features: Video features tensor (1, num_frames, feature_dim)
        vocabulary: Vocabulary object
        max_length: Maximum caption length
        beam_size: Beam search width
        
    Returns:
        Best predicted sequence
    """
    device = next(model.parameters()).device
    batch_size = video_features.size(0)
    num_frames = video_features.size(1)
    
    # Initialize beam
    # Each beam element: (sequence, score, hidden_state)
    beams = [(
        [vocabulary.word2idx[vocabulary.BOS_TOKEN]], 
        0.0, 
        model.init_hidden(1, device)
    )]
    
    # Encode video features first
    hidden = model.init_hidden(batch_size, device)
    pad_tokens = torch.zeros(batch_size, max_length, model.hidden_dim).to(device)
    video_proj = model.video_feature_proj(video_features)
    encoder_input = torch.cat([video_proj, pad_tokens], dim=1)
    encoder_output, hidden = model.lstm(encoder_input, hidden)
    
    for step in range(max_length - 1):
        new_beams = []
        
        for sequence, score, beam_hidden in beams:
            if sequence[-1] == vocabulary.word2idx[vocabulary.EOS_TOKEN]:
                new_beams.append((sequence, score, beam_hidden))
                continue
            
            # Get current token
            current_token = torch.tensor([[sequence[-1]]], dtype=torch.long).to(device)
            
            # Embed token
            embedded = model.word_embedding(current_token)
            
            # Project to hidden dimension
            if model.embedding_dim != model.hidden_dim:
                # Simple projection
                decoder_input = embedded @ model.word_embedding.weight[:model.hidden_dim, :].T
            else:
                decoder_input = embedded
            
            # Prepare full input
            video_pad = torch.zeros(1, num_frames, model.hidden_dim).to(device)
            full_input = torch.cat([video_pad, decoder_input], dim=1)
            
            # LSTM forward
            lstm_out, new_hidden = model.lstm(full_input, beam_hidden)
            text_out = lstm_out[:, num_frames:, :]
            vocab_out = model.output_proj(text_out)
            
            # Get top k tokens
            log_probs = F.log_softmax(vocab_out.squeeze(0), dim=-1)
            top_log_probs, top_indices = torch.topk(log_probs[-1], beam_size)
            
            # Create new beam candidates
            for i in range(beam_size):
                new_sequence = sequence + [top_indices[i].item()]
                new_score = score + top_log_probs[i].item()
                new_beams.append((new_sequence, new_score, new_hidden))
        
        # Keep top beam_size beams
        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]
        
        # Early stopping if all beams end with EOS
        if all(beam[0][-1] == vocabulary.word2idx[vocabulary.EOS_TOKEN] for beam in beams):
            break
    
    # Return best sequence
    best_sequence = beams[0][0]
    return torch.tensor(best_sequence)


def evaluate_model(model, data_loader, vocabulary, device, max_length=20):
    """
    Evaluate model on validation/test set
    
    Args:
        model: Trained S2VT model
        data_loader: DataLoader for evaluation data
        vocabulary: Vocabulary object
        device: Device to run evaluation on
        max_length: Maximum caption length
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    
    predictions = []
    references = []
    
    print("Generating captions for evaluation...")
    
    with torch.no_grad():
        for batch in tqdm(data_loader):
            video_features = batch['video_features'].to(device)
            caption_texts = batch['caption_texts']
            
            # Generate captions for each video in batch
            for i in range(video_features.size(0)):
                single_video = video_features[i:i+1]  # Keep batch dimension
                
                # Generate caption
                outputs = model(single_video, max_length=max_length)
                predicted_ids = outputs.argmax(dim=-1).squeeze(0)
                
                # Decode caption
                predicted_caption = vocabulary.decode_caption(
                    predicted_ids.cpu().numpy().tolist()
                )
                
                predictions.append(predicted_caption)
                references.append([caption_texts[i]])  # Wrap in list for multiple references
    
    # Calculate evaluation metrics
    print("Calculating evaluation metrics...")
    scores = evaluate_captions(predictions, references)
    
    # Print some example predictions
    print("\nExample predictions:")
    for i in range(min(5, len(predictions))):
        print(f"Reference: {references[i][0]}")
        print(f"Predicted: {predictions[i]}")
        print("-" * 50)
    
    return scores


def inference_demo(model, video_features, vocabulary, device):
    """
    Demo function for model inference
    
    Args:
        model: Trained S2VT model
        video_features: Video features array (num_videos, num_frames, feature_dim)
        vocabulary: Vocabulary object
        device: Device to run inference on
    """
    model.eval()
    
    print("Running inference demo...")
    
    for i in range(min(5, len(video_features))):
        # Prepare single video
        video = torch.tensor(video_features[i:i+1], dtype=torch.float32)
        
        # Generate caption with different methods
        greedy_caption = generate_caption(
            model, video, vocabulary, method='greedy'
        )
        
        beam_caption = generate_caption(
            model, video, vocabulary, method='beam_search'
        )
        
        print(f"Video {i+1}:")
        print(f"  Greedy: {greedy_caption}")
        print(f"  Beam Search: {beam_caption}")
        print("-" * 50)


def analyze_attention_weights(model, video_features, vocabulary, device):
    """
    Analyze attention patterns in the model (if attention is implemented)
    This is a placeholder for future attention visualization
    """
    # TODO: Implement attention analysis when attention mechanism is added
    pass


def calculate_diversity_metrics(captions: List[str]) -> Dict[str, float]:
    """
    Calculate diversity metrics for generated captions
    
    Args:
        captions: List of generated captions
        
    Returns:
        Dictionary with diversity metrics
    """
    
    # Tokenize all captions
    all_tokens = []
    unique_captions = set()
    
    for caption in captions:
        tokens = caption.lower().split()
        all_tokens.extend(tokens)
        unique_captions.add(caption.lower())
    
    # Calculate metrics
    total_tokens = len(all_tokens)
    unique_tokens = len(set(all_tokens))
    total_captions = len(captions)
    
    metrics = {
        'unique_caption_ratio': len(unique_captions) / total_captions,
        'vocabulary_usage': unique_tokens / total_tokens if total_tokens > 0 else 0,
        'average_caption_length': np.mean([len(cap.split()) for cap in captions]),
        'caption_length_std': np.std([len(cap.split()) for cap in captions])
    }
    
    return metrics


def error_analysis(predictions: List[str], 
                  references: List[List[str]], 
                  vocabulary: object) -> Dict[str, any]:
    """
    Perform error analysis on predictions
    
    Args:
        predictions: List of predicted captions
        references: List of reference caption lists
        vocabulary: Vocabulary object
        
    Returns:
        Dictionary with error analysis results
    """
    analysis = {
        'length_errors': [],
        'vocabulary_errors': [],
        'common_mistakes': Counter(),
        'missed_words': Counter()
    }
    
    for pred, refs in zip(predictions, references):
        pred_tokens = pred.lower().split()
        
        # Analyze against best matching reference
        best_ref = ""
        best_overlap = 0
        
        for ref in refs:
            ref_tokens = ref.lower().split()
            overlap = len(set(pred_tokens) & set(ref_tokens))
            if overlap > best_overlap:
                best_overlap = overlap
                best_ref = ref
        
        ref_tokens = best_ref.lower().split()
        
        # Length analysis
        analysis['length_errors'].append(len(pred_tokens) - len(ref_tokens))
        
        # Vocabulary analysis
        pred_set = set(pred_tokens)
        ref_set = set(ref_tokens)
        
        # Words in prediction but not in reference
        extra_words = pred_set - ref_set
        for word in extra_words:
            analysis['common_mistakes'][word] += 1
        
        # Words in reference but not in prediction
        missed_words = ref_set - pred_set
        for word in missed_words:
            analysis['missed_words'][word] += 1
    
    # Calculate statistics
    analysis['avg_length_error'] = np.mean(analysis['length_errors'])
    analysis['length_error_std'] = np.std(analysis['length_errors'])
    analysis['most_common_mistakes'] = analysis['common_mistakes'].most_common(10)
    analysis['most_missed_words'] = analysis['missed_words'].most_common(10)
    
    return analysis