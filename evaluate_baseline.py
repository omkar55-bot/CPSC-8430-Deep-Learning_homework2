"""
Comprehensive evaluation script following the baseline standards
Implements proper BLEU evaluation and baseline comparison
"""

import torch
import numpy as np
import json
import argparse
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# Import our modules
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.attention_s2vt import AttentionS2VT, BeamSearchDecoder
from models.s2vt_model import S2VTModel
from data.preprocessing import Vocabulary
from data.msvd_dataset import MSVDDataset
from utils.metrics import evaluate_captions, calculate_bleu_score


class BaselineEvaluator:
    """
    Evaluator following baseline standards for video caption generation
    """
    
    def __init__(self, model_path, vocab_path=None, device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model and vocabulary
        self.model, self.vocabulary, self.config = self._load_model_and_vocab(model_path, vocab_path)
        self.model.eval()
        
        # Initialize beam search decoder
        self.beam_decoder = BeamSearchDecoder(
            self.model,
            self.vocabulary,
            beam_size=5,
            max_length=20,
            length_penalty=1.0
        )
        
        print(f"Baseline Evaluator initialized on {self.device}")
        print(f"Model type: {type(self.model).__name__}")
        print(f"Vocabulary size: {len(self.vocabulary)}")
    
    def _load_model_and_vocab(self, model_path, vocab_path):
        """Load model and vocabulary"""
        print(f"Loading model from {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        config = checkpoint['config']
        
        # Load vocabulary
        if vocab_path is None:
            model_dir = os.path.dirname(model_path)
            vocab_path = os.path.join(model_dir, 'vocabulary.pkl')
        
        vocabulary = Vocabulary()
        vocabulary.load(vocab_path)
        
        # Initialize model based on type
        model_type = config.get('model_type', 'basic')
        
        if model_type == 'attention' or model_type == 'scheduled_sampling':
            model = AttentionS2VT(
                vocab_size=len(vocabulary),
                max_frames=config['max_frames'],
                video_feature_dim=config['video_feature_dim'],
                hidden_dim=config['hidden_dim'],
                embedding_dim=config['embedding_dim'],
                num_layers=config['num_layers'],
                dropout=config['dropout'],
                attention_dim=config.get('attention_dim', 256)
            ).to(self.device)
        else:
            model = S2VTModel(
                vocab_size=len(vocabulary),
                max_frames=config['max_frames'],
                video_feature_dim=config['video_feature_dim'],
                hidden_dim=config['hidden_dim'],
                embedding_dim=config['embedding_dim'],
                num_layers=config['num_layers'],
                dropout=config['dropout']
            ).to(self.device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model, vocabulary, config
    
    def evaluate_baseline_bleu(self, data_root, split='test', use_beam_search=True):
        """
        Evaluate using baseline BLEU calculation method
        Following the exact calculation shown in training tips
        """
        print(f"Evaluating baseline BLEU on {split} set...")
        
        # Create dataset
        dataset = MSVDDataset(
            data_root=data_root,
            split=split,
            vocabulary=self.vocabulary,
            max_caption_length=self.config['max_caption_length'],
            max_frames=self.config['max_frames']
        )
        
        predictions = []
        all_references = []
        video_ids = []
        
        print(f"Processing {len(dataset)} videos...")
        
        with torch.no_grad():
            for i in range(len(dataset)):
                if i % 100 == 0:
                    print(f"Processed {i}/{len(dataset)} videos...")
                
                sample = dataset[i]
                video_features = sample['video_features'].unsqueeze(0).to(self.device)
                
                # Generate caption
                if use_beam_search:
                    predicted_ids, score, attention = self.beam_decoder.decode(video_features)
                    predicted_caption = self.vocabulary.decode_caption(predicted_ids)
                else:
                    # Greedy decoding
                    if hasattr(self.model, '_forward_inference'):
                        outputs, _ = self.model(video_features, max_length=self.config['max_caption_length'])
                    else:
                        outputs = self.model(video_features, max_length=self.config['max_caption_length'])
                    
                    predicted_ids = outputs.argmax(dim=-1).squeeze(0)
                    predicted_caption = self.vocabulary.decode_caption(
                        predicted_ids.cpu().numpy().tolist()
                    )
                
                predictions.append(predicted_caption)
                all_references.append(sample['caption_text'])
                video_ids.append(sample['video_id'])
        
        # Calculate baseline BLEU (Captions Avg.)
        # This averages BLEU scores across all reference captions for each video
        bleu_scores = []
        
        for pred, refs in zip(predictions, all_references):
            if isinstance(refs, str):
                refs = [refs]
            
            # Calculate BLEU for this prediction against all references
            video_bleu_scores = []
            for ref in refs:
                # Calculate precision for each n-gram
                pred_tokens = pred.lower().split()
                ref_tokens = ref.lower().split()
                
                if len(pred_tokens) == 0:
                    video_bleu_scores.append(0.0)
                    continue
                
                # BLEU@1 calculation as shown in training tips
                # Precision = correct words / candidate length
                correct_words = sum(1 for token in pred_tokens if token in ref_tokens)
                precision = correct_words / len(pred_tokens)
                
                # Brevity penalty
                if len(pred_tokens) > len(ref_tokens):
                    bp = 1.0
                else:
                    bp = np.exp(1 - len(ref_tokens) / len(pred_tokens)) if len(pred_tokens) > 0 else 0.0
                
                bleu1 = bp * precision
                video_bleu_scores.append(bleu1)
            
            # Average BLEU across all references for this video
            avg_bleu = np.mean(video_bleu_scores)
            bleu_scores.append(avg_bleu)
        
        # Calculate overall metrics
        baseline_bleu1 = np.mean(bleu_scores)
        
        # Also calculate standard evaluation metrics
        standard_scores = evaluate_captions(predictions, all_references)
        
        # Prepare results
        results = {
            'baseline_bleu1_captions_avg': baseline_bleu1,
            'standard_metrics': standard_scores,
            'num_videos': len(predictions),
            'decoding_method': 'beam_search' if use_beam_search else 'greedy',
            'baseline_target': 0.6,
            'baseline_reached': baseline_bleu1 >= 0.6
        }
        
        return results, predictions, all_references, video_ids
    
    def detailed_error_analysis(self, predictions, references, video_ids):
        """Perform detailed error analysis"""
        print("Performing detailed error analysis...")
        
        analysis = {
            'length_distribution': [],
            'word_frequency_errors': defaultdict(int),
            'successful_predictions': [],
            'failed_predictions': [],
            'bleu_distribution': []
        }
        
        for pred, refs, vid_id in zip(predictions, references, video_ids):
            pred_tokens = pred.lower().split()
            
            # Length analysis
            analysis['length_distribution'].append(len(pred_tokens))
            
            # BLEU score for this prediction
            if isinstance(refs, str):
                refs = [refs]
            
            pred_bleu_scores = []
            for ref in refs:
                ref_tokens = ref.lower().split()
                if len(pred_tokens) > 0:
                    correct = sum(1 for token in pred_tokens if token in ref_tokens)
                    precision = correct / len(pred_tokens)
                    
                    if len(pred_tokens) > len(ref_tokens):
                        bp = 1.0
                    else:
                        bp = np.exp(1 - len(ref_tokens) / len(pred_tokens))
                    
                    bleu1 = bp * precision
                    pred_bleu_scores.append(bleu1)
            
            avg_bleu = np.mean(pred_bleu_scores) if pred_bleu_scores else 0.0
            analysis['bleu_distribution'].append(avg_bleu)
            
            # Categorize predictions
            if avg_bleu >= 0.5:
                analysis['successful_predictions'].append({
                    'video_id': vid_id,
                    'prediction': pred,
                    'reference': refs[0],
                    'bleu1': avg_bleu
                })
            else:
                analysis['failed_predictions'].append({
                    'video_id': vid_id,
                    'prediction': pred,
                    'reference': refs[0],
                    'bleu1': avg_bleu
                })
            
            # Word frequency errors
            best_ref = max(refs, key=lambda r: len(set(pred_tokens) & set(r.lower().split())))
            ref_tokens = best_ref.lower().split()
            
            for token in pred_tokens:
                if token not in ref_tokens:
                    analysis['word_frequency_errors'][token] += 1
        
        return analysis
    
    def compare_with_baseline(self, results):
        """Compare results with baseline standards"""
        print("\nBaseline Comparison:")
        print("=" * 50)
        
        baseline_bleu1 = results['baseline_bleu1_captions_avg']
        target_bleu1 = 0.6
        
        print(f"Target BLEU@1 (Captions Avg.): {target_bleu1:.3f}")
        print(f"Achieved BLEU@1 (Captions Avg.): {baseline_bleu1:.3f}")
        
        if baseline_bleu1 >= target_bleu1:
            print(f"✅ BASELINE REACHED! ({baseline_bleu1:.3f} >= {target_bleu1:.3f})")
            gap = baseline_bleu1 - target_bleu1
            print(f"📈 Exceeded baseline by: {gap:.3f}")
        else:
            print(f"❌ Baseline not reached ({baseline_bleu1:.3f} < {target_bleu1:.3f})")
            gap = target_bleu1 - baseline_bleu1
            print(f"📉 Gap to baseline: {gap:.3f}")
        
        # Additional metrics comparison
        print(f"\nStandard Metrics:")
        for metric, score in results['standard_metrics'].items():
            print(f"  {metric}: {score:.4f}")
        
        return baseline_bleu1 >= target_bleu1
    
    def generate_evaluation_report(self, results, analysis, output_dir):
        """Generate comprehensive evaluation report"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save detailed results
        with open(os.path.join(output_dir, 'evaluation_results.json'), 'w') as f:
            json.dump({
                'results': results,
                'analysis': {
                    'length_distribution': analysis['length_distribution'],
                    'bleu_distribution': analysis['bleu_distribution'],
                    'num_successful': len(analysis['successful_predictions']),
                    'num_failed': len(analysis['failed_predictions']),
                    'most_common_errors': dict(list(analysis['word_frequency_errors'].most_common(20)))
                },
                'successful_examples': analysis['successful_predictions'][:10],
                'failed_examples': analysis['failed_predictions'][:10]
            }, f, indent=2)
        
        # Create visualizations
        self._create_visualizations(results, analysis, output_dir)
        
        # Generate text report
        self._generate_text_report(results, analysis, output_dir)
    
    def _create_visualizations(self, results, analysis, output_dir):
        """Create evaluation visualizations"""
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        
        # BLEU score distribution
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.hist(analysis['bleu_distribution'], bins=30, alpha=0.7, color='skyblue')
        plt.axvline(x=0.6, color='red', linestyle='--', label='Baseline Target')
        plt.axvline(x=np.mean(analysis['bleu_distribution']), color='green', linestyle='-', label='Mean Score')
        plt.xlabel('BLEU-1 Score')
        plt.ylabel('Frequency')
        plt.title('BLEU-1 Score Distribution')
        plt.legend()
        
        # Caption length distribution
        plt.subplot(2, 2, 2)
        plt.hist(analysis['length_distribution'], bins=20, alpha=0.7, color='lightcoral')
        plt.xlabel('Caption Length (words)')
        plt.ylabel('Frequency')
        plt.title('Generated Caption Length Distribution')
        
        # Success vs Failure
        plt.subplot(2, 2, 3)
        success_count = len(analysis['successful_predictions'])
        fail_count = len(analysis['failed_predictions'])
        plt.pie([success_count, fail_count], labels=['Success (BLEU≥0.5)', 'Failure (BLEU<0.5)'], 
                autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'])
        plt.title('Prediction Success Rate')
        
        # Top error words
        plt.subplot(2, 2, 4)
        top_errors = dict(list(analysis['word_frequency_errors'].most_common(10)))
        if top_errors:
            plt.barh(list(top_errors.keys()), list(top_errors.values()))
            plt.xlabel('Frequency')
            plt.title('Most Common Error Words')
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'evaluation_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_text_report(self, results, analysis, output_dir):
        """Generate text evaluation report"""
        with open(os.path.join(output_dir, 'evaluation_report.txt'), 'w') as f:
            f.write("S2VT Video Caption Generation - Evaluation Report\n")
            f.write("=" * 60 + "\n\n")
            
            # Baseline comparison
            f.write("BASELINE COMPARISON\n")
            f.write("-" * 30 + "\n")
            f.write(f"Target BLEU@1 (Captions Avg.): 0.600\n")
            f.write(f"Achieved BLEU@1 (Captions Avg.): {results['baseline_bleu1_captions_avg']:.3f}\n")
            f.write(f"Baseline Reached: {'YES' if results['baseline_reached'] else 'NO'}\n\n")
            
            # Standard metrics
            f.write("STANDARD METRICS\n")
            f.write("-" * 30 + "\n")
            for metric, score in results['standard_metrics'].items():
                f.write(f"{metric:12}: {score:.4f}\n")
            f.write("\n")
            
            # Statistics
            f.write("STATISTICS\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total videos evaluated: {results['num_videos']}\n")
            f.write(f"Successful predictions (BLEU≥0.5): {len(analysis['successful_predictions'])}\n")
            f.write(f"Failed predictions (BLEU<0.5): {len(analysis['failed_predictions'])}\n")
            f.write(f"Success rate: {len(analysis['successful_predictions'])/results['num_videos']*100:.1f}%\n")
            f.write(f"Average caption length: {np.mean(analysis['length_distribution']):.1f} words\n\n")
            
            # Examples
            f.write("SUCCESSFUL EXAMPLES\n")
            f.write("-" * 30 + "\n")
            for i, example in enumerate(analysis['successful_predictions'][:5]):
                f.write(f"{i+1}. Video: {example['video_id']}\n")
                f.write(f"   Reference: {example['reference']}\n")
                f.write(f"   Prediction: {example['prediction']}\n")
                f.write(f"   BLEU-1: {example['bleu1']:.3f}\n\n")
            
            f.write("FAILED EXAMPLES\n")
            f.write("-" * 30 + "\n")
            for i, example in enumerate(analysis['failed_predictions'][:5]):
                f.write(f"{i+1}. Video: {example['video_id']}\n")
                f.write(f"   Reference: {example['reference']}\n")
                f.write(f"   Prediction: {example['prediction']}\n")
                f.write(f"   BLEU-1: {example['bleu1']:.3f}\n\n")


def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description='Baseline S2VT Evaluation')
    
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--vocab_path', type=str, default=None,
                       help='Path to vocabulary file')
    parser.add_argument('--data_root', type=str, default='E:/imgsynth/MLDS_hw2_1_data',
                       help='Root directory of MLDS_hw2_1_data')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'test'],
                       help='Dataset split to evaluate on')
    parser.add_argument('--use_beam_search', action='store_true', default=True,
                       help='Use beam search for decoding')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to run evaluation on')
    
    args = parser.parse_args()
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize evaluator
    evaluator = BaselineEvaluator(args.model_path, args.vocab_path, device)
    
    # Run evaluation
    results, predictions, references, video_ids = evaluator.evaluate_baseline_bleu(
        args.data_root, args.split, args.use_beam_search
    )
    
    # Perform error analysis
    analysis = evaluator.detailed_error_analysis(predictions, references, video_ids)
    
    # Compare with baseline
    baseline_reached = evaluator.compare_with_baseline(results)
    
    # Generate comprehensive report
    evaluator.generate_evaluation_report(results, analysis, args.output_dir)
    
    print(f"\nEvaluation completed!")
    print(f"Results saved to: {args.output_dir}")
    print(f"Baseline BLEU@1: {results['baseline_bleu1_captions_avg']:.3f}")
    print(f"Baseline reached: {'YES' if baseline_reached else 'NO'}")


if __name__ == "__main__":
    main()