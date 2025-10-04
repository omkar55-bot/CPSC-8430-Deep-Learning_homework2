"""
Evaluation metrics for video caption generation
Includes BLEU, METEOR, CIDEr, and ROUGE scores
"""

import torch
import numpy as np
from collections import Counter
import re
from typing import List, Dict, Tuple
import math


def tokenize_caption(caption: str) -> List[str]:
    """Simple tokenization for captions"""
    # Convert to lowercase and remove punctuation
    caption = re.sub(r'[^\w\s]', '', caption.lower())
    return caption.split()


def calculate_bleu_score(predictions: List[str], 
                        references: List[List[str]], 
                        n_gram: int = 4) -> Dict[str, float]:
    """
    Calculate BLEU score for predicted captions
    
    Args:
        predictions: List of predicted captions
        references: List of reference caption lists (multiple references per prediction)
        n_gram: Maximum n-gram order to consider
        
    Returns:
        Dictionary with BLEU scores
    """
    def get_ngrams(tokens: List[str], n: int) -> Counter:
        """Get n-grams from token list"""
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngrams.append(tuple(tokens[i:i+n]))
        return Counter(ngrams)
    
    def modified_precision(pred_tokens: List[str], ref_tokens_list: List[List[str]], n: int) -> float:
        """Calculate modified precision for n-grams"""
        pred_ngrams = get_ngrams(pred_tokens, n)
        
        # Count maximum occurrences in any reference
        max_ref_counts = Counter()
        for ref_tokens in ref_tokens_list:
            ref_ngrams = get_ngrams(ref_tokens, n)
            for ngram in ref_ngrams:
                max_ref_counts[ngram] = max(max_ref_counts[ngram], ref_ngrams[ngram])
        
        # Calculate clipped counts
        clipped_counts = 0
        total_counts = 0
        
        for ngram, count in pred_ngrams.items():
            clipped_counts += min(count, max_ref_counts[ngram])
            total_counts += count
        
        return clipped_counts / total_counts if total_counts > 0 else 0.0
    
    def brevity_penalty(pred_length: int, ref_lengths: List[int]) -> float:
        """Calculate brevity penalty"""
        # Find closest reference length
        closest_ref_len = min(ref_lengths, key=lambda x: abs(x - pred_length))
        
        if pred_length > closest_ref_len:
            return 1.0
        else:
            return math.exp(1 - closest_ref_len / pred_length) if pred_length > 0 else 0.0
    
    # Calculate BLEU for each prediction
    bleu_scores = []
    precision_scores = {i: [] for i in range(1, n_gram + 1)}
    
    for pred, refs in zip(predictions, references):
        pred_tokens = tokenize_caption(pred)
        ref_tokens_list = [tokenize_caption(ref) for ref in refs]
        
        # Calculate modified precision for each n-gram
        precisions = []
        for n in range(1, n_gram + 1):
            precision = modified_precision(pred_tokens, ref_tokens_list, n)
            precisions.append(precision)
            precision_scores[n].append(precision)
        
        # Calculate brevity penalty
        pred_len = len(pred_tokens)
        ref_lengths = [len(ref_tokens) for ref_tokens in ref_tokens_list]
        bp = brevity_penalty(pred_len, ref_lengths)
        
        # Calculate BLEU score
        if all(p > 0 for p in precisions):
            log_precisions = [math.log(p) for p in precisions]
            bleu = bp * math.exp(sum(log_precisions) / len(log_precisions))
        else:
            bleu = 0.0
        
        bleu_scores.append(bleu)
    
    # Calculate average scores
    results = {
        'BLEU': np.mean(bleu_scores),
        'BLEU-1': np.mean(precision_scores[1]),
        'BLEU-2': np.mean(precision_scores[2]),
        'BLEU-3': np.mean(precision_scores[3]),
        'BLEU-4': np.mean(precision_scores[4])
    }
    
    return results


def calculate_meteor_score(predictions: List[str], references: List[List[str]]) -> float:
    """
    Simplified METEOR score calculation
    This is a basic implementation - for production, use official METEOR scorer
    """
    def get_unigram_matches(pred_tokens: List[str], ref_tokens: List[str]) -> Tuple[int, int, int]:
        """Get unigram matches between prediction and reference"""
        pred_set = set(pred_tokens)
        ref_set = set(ref_tokens)
        
        matches = len(pred_set & ref_set)
        pred_len = len(pred_tokens)
        ref_len = len(ref_tokens)
        
        return matches, pred_len, ref_len
    
    meteor_scores = []
    
    for pred, refs in zip(predictions, references):
        pred_tokens = tokenize_caption(pred)
        
        # Find best matching reference
        best_score = 0.0
        for ref in refs:
            ref_tokens = tokenize_caption(ref)
            
            matches, pred_len, ref_len = get_unigram_matches(pred_tokens, ref_tokens)
            
            if matches > 0:
                precision = matches / pred_len if pred_len > 0 else 0
                recall = matches / ref_len if ref_len > 0 else 0
                
                if precision + recall > 0:
                    f_score = (2 * precision * recall) / (precision + recall)
                    best_score = max(best_score, f_score)
        
        meteor_scores.append(best_score)
    
    return np.mean(meteor_scores)


def calculate_cider_score(predictions: List[str], references: List[List[str]]) -> float:
    """
    Simplified CIDEr score calculation
    This is a basic implementation - for production, use official CIDEr scorer
    """
    def get_tfidf_weights(all_captions: List[str]) -> Dict[str, float]:
        """Calculate TF-IDF weights for all captions"""
        # Tokenize all captions
        all_tokens = []
        for caption in all_captions:
            all_tokens.extend(tokenize_caption(caption))
        
        # Calculate document frequency
        doc_freq = Counter()
        for caption in all_captions:
            tokens = set(tokenize_caption(caption))
            for token in tokens:
                doc_freq[token] += 1
        
        # Calculate IDF weights
        num_docs = len(all_captions)
        idf_weights = {}
        for token, freq in doc_freq.items():
            idf_weights[token] = math.log(num_docs / freq)
        
        return idf_weights
    
    # Prepare all captions for TF-IDF calculation
    all_captions = predictions.copy()
    for refs in references:
        all_captions.extend(refs)
    
    idf_weights = get_tfidf_weights(all_captions)
    
    cider_scores = []
    
    for pred, refs in zip(predictions, references):
        pred_tokens = tokenize_caption(pred)
        
        # Calculate CIDEr for each reference
        ref_scores = []
        for ref in refs:
            ref_tokens = tokenize_caption(ref)
            
            # Calculate weighted n-gram similarity
            score = 0.0
            for n in range(1, 5):  # 1-gram to 4-gram
                pred_ngrams = Counter()
                ref_ngrams = Counter()
                
                # Get n-grams
                for i in range(len(pred_tokens) - n + 1):
                    ngram = ' '.join(pred_tokens[i:i+n])
                    pred_ngrams[ngram] += 1
                
                for i in range(len(ref_tokens) - n + 1):
                    ngram = ' '.join(ref_tokens[i:i+n])
                    ref_ngrams[ngram] += 1
                
                # Calculate cosine similarity with TF-IDF weighting
                dot_product = 0.0
                pred_norm = 0.0
                ref_norm = 0.0
                
                all_ngrams = set(pred_ngrams.keys()) | set(ref_ngrams.keys())
                
                for ngram in all_ngrams:
                    # Use average IDF of constituent words
                    words = ngram.split()
                    avg_idf = np.mean([idf_weights.get(word, 0) for word in words])
                    
                    pred_tf = pred_ngrams.get(ngram, 0)
                    ref_tf = ref_ngrams.get(ngram, 0)
                    
                    weighted_pred = pred_tf * avg_idf
                    weighted_ref = ref_tf * avg_idf
                    
                    dot_product += weighted_pred * weighted_ref
                    pred_norm += weighted_pred ** 2
                    ref_norm += weighted_ref ** 2
                
                if pred_norm > 0 and ref_norm > 0:
                    cosine_sim = dot_product / (math.sqrt(pred_norm) * math.sqrt(ref_norm))
                    score += cosine_sim
            
            ref_scores.append(score)
        
        # Average over references
        cider_scores.append(np.mean(ref_scores))
    
    return np.mean(cider_scores)


def calculate_rouge_l(predictions: List[str], references: List[List[str]]) -> float:
    """
    Calculate ROUGE-L score (Longest Common Subsequence)
    """
    def lcs_length(seq1: List[str], seq2: List[str]) -> int:
        """Calculate length of longest common subsequence"""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    rouge_scores = []
    
    for pred, refs in zip(predictions, references):
        pred_tokens = tokenize_caption(pred)
        
        best_score = 0.0
        for ref in refs:
            ref_tokens = tokenize_caption(ref)
            
            lcs_len = lcs_length(pred_tokens, ref_tokens)
            
            if len(pred_tokens) > 0 and len(ref_tokens) > 0:
                precision = lcs_len / len(pred_tokens)
                recall = lcs_len / len(ref_tokens)
                
                if precision + recall > 0:
                    f_score = (2 * precision * recall) / (precision + recall)
                    best_score = max(best_score, f_score)
        
        rouge_scores.append(best_score)
    
    return np.mean(rouge_scores)


def evaluate_captions(predictions: List[str], 
                     references: List[List[str]]) -> Dict[str, float]:
    """
    Comprehensive evaluation of generated captions
    
    Args:
        predictions: List of predicted captions
        references: List of reference caption lists
        
    Returns:
        Dictionary with all evaluation metrics
    """
    results = {}
    
    # BLEU scores
    bleu_results = calculate_bleu_score(predictions, references)
    results.update(bleu_results)
    
    # METEOR score
    results['METEOR'] = calculate_meteor_score(predictions, references)
    
    # CIDEr score
    results['CIDEr'] = calculate_cider_score(predictions, references)
    
    # ROUGE-L score
    results['ROUGE-L'] = calculate_rouge_l(predictions, references)
    
    return results


if __name__ == "__main__":
    # Test the metrics
    predictions = [
        "a man is walking down the street",
        "a woman is cooking in the kitchen",
        "children are playing in the park"
    ]
    
    references = [
        ["a person is walking on the road", "a man walks down the street"],
        ["a woman cooks food", "someone is cooking in a kitchen"],
        ["kids play in the park", "children are having fun at the park"]
    ]
    
    scores = evaluate_captions(predictions, references)
    
    print("Evaluation Results:")
    for metric, score in scores.items():
        print(f"{metric}: {score:.4f}")