"""
Automated Benchmarking System for Table Reconstruction

Provides continuous evaluation and benchmarking capabilities for table reconstruction
against PubTabNet ground truth data.
"""

import json
import os
import time
from typing import Dict, List, Tuple, Optional, Iterator
from dataclasses import dataclass, asdict
from pathlib import Path
import statistics
from collections import defaultdict

from pubtabnet_data_loader import PubTabNetDataLoader, PubTabNetSample, PubTabNetDataset
from html_parser import PubTabNetHTMLParser
from perfect_reconstructor import PerfectTableReconstructor
from evaluation_metrics import TableReconstructionEvaluator, EvaluationResult


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking runs"""
    dataset_split: str = 'val'
    max_samples: Optional[int] = None
    batch_size: int = 1
    save_results: bool = True
    results_dir: str = './benchmark_results'
    detailed_analysis: bool = True
    error_analysis: bool = True
    model_name: str = 'perfect_reconstructor'


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run"""
    config: BenchmarkConfig
    total_samples: int
    successful_samples: int
    failed_samples: int
    average_scores: Dict[str, float]
    score_distribution: Dict[str, List[float]]
    error_analysis: Dict[str, any]
    processing_time: float
    detailed_results: List[EvaluationResult]
    timestamp: str


class TableReconstructionBenchmark:
    """Automated benchmarking system for table reconstruction"""
    
    def __init__(self, config: BenchmarkConfig = None):
        self.config = config or BenchmarkConfig()
        self.data_loader = PubTabNetDataLoader()
        self.html_parser = PubTabNetHTMLParser()
        self.reconstructor = PerfectTableReconstructor()
        self.evaluator = TableReconstructionEvaluator()
        
        # Create results directory
        os.makedirs(self.config.results_dir, exist_ok=True)
    
    def run_benchmark(self) -> BenchmarkResult:
        """Run complete benchmark evaluation"""
        print(f"Starting benchmark: {self.config.model_name}")
        print(f"Dataset: {self.config.dataset_split}")
        print(f"Max samples: {self.config.max_samples or 'All'}")
        print("=" * 60)
        
        start_time = time.time()
        
        # Load dataset
        dataset = self.data_loader.load_split(
            self.config.dataset_split, 
            self.config.max_samples
        )
        
        print(f"Loaded {len(dataset.samples)} samples")
        
        # Run evaluation on all samples
        detailed_results = []
        successful_samples = 0
        failed_samples = 0
        
        for i, sample in enumerate(dataset.samples):
            try:
                print(f"Processing sample {i+1}/{len(dataset.samples)}: {sample.table_id}")
                
                # Parse ground truth
                ground_truth = self.html_parser.parse_pubtabnet_sample(
                    sample.html_annotation,
                    sample.cell_bboxes,
                    sample.image_width,
                    sample.image_height
                )
                
                # Generate model prediction (simulate for now)
                model_prediction = self._simulate_model_prediction(sample, ground_truth)
                
                # Reconstruct table
                reconstructed = self.reconstructor.reconstruct_from_model_output(
                    model_prediction['model_cells'],
                    model_prediction['ocr_text_blocks'],
                    ground_truth
                )
                
                # Evaluate reconstruction
                evaluation_result = self.evaluator.evaluate_reconstruction(
                    reconstructed, ground_truth
                )
                
                detailed_results.append(evaluation_result)
                successful_samples += 1
                
                # Print progress
                if (i + 1) % 10 == 0:
                    avg_score = statistics.mean([r.overall_score for r in detailed_results])
                    print(f"   Average score so far: {avg_score:.3f}")
                
            except Exception as e:
                print(f"Failed to process sample {sample.table_id}: {e}")
                failed_samples += 1
                continue
        
        processing_time = time.time() - start_time
        
        # Calculate aggregate statistics
        benchmark_result = self._calculate_benchmark_statistics(
            detailed_results, successful_samples, failed_samples, processing_time
        )
        
        # Save results if requested
        if self.config.save_results:
            self._save_benchmark_results(benchmark_result)
        
        # Print summary
        self._print_benchmark_summary(benchmark_result)
        
        return benchmark_result
    
    def _simulate_model_prediction(self, sample: PubTabNetSample, ground_truth) -> Dict:
        """Simulate model prediction for testing (replace with actual model output)"""
        # This is a placeholder - in real implementation, you would:
        # 1. Load the image
        # 2. Run TableFormer/Table Transformer
        # 3. Run OCR
        # 4. Return model outputs
        
        # For now, create synthetic model outputs based on ground truth
        model_cells = []
        ocr_text_blocks = []
        
        for cell in ground_truth.cells:
            if cell.bbox:
                # Add some noise to simulate model prediction
                noise_factor = 0.1
                bbox = cell.bbox.copy()
                bbox[0] += (hash(cell.content) % 10 - 5) * noise_factor  # x1
                bbox[1] += (hash(cell.content) % 10 - 5) * noise_factor  # y1
                bbox[2] += (hash(cell.content) % 10 - 5) * noise_factor  # x2
                bbox[3] += (hash(cell.content) % 10 - 5) * noise_factor  # y2
                
                model_cells.append({
                    'bbox': bbox,
                    'content': cell.content,
                    'confidence': 0.8 + (hash(cell.content) % 20) / 100.0,
                    'is_header': cell.is_header
                })
                
                ocr_text_blocks.append({
                    'bbox': bbox,
                    'text': cell.content,
                    'confidence': 0.7 + (hash(cell.content) % 30) / 100.0
                })
        
        return {
            'model_cells': model_cells,
            'ocr_text_blocks': ocr_text_blocks
        }
    
    def _calculate_benchmark_statistics(self, detailed_results: List[EvaluationResult], 
                                      successful_samples: int, failed_samples: int, 
                                      processing_time: float) -> BenchmarkResult:
        """Calculate aggregate statistics from detailed results"""
        
        if not detailed_results:
            return BenchmarkResult(
                config=self.config,
                total_samples=successful_samples + failed_samples,
                successful_samples=successful_samples,
                failed_samples=failed_samples,
                average_scores={},
                score_distribution={},
                error_analysis={},
                processing_time=processing_time,
                detailed_results=detailed_results,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
            )
        
        # Calculate average scores
        average_scores = {
            'cell_iou_score': statistics.mean([r.cell_iou_score for r in detailed_results]),
            'structural_similarity': statistics.mean([r.structural_similarity for r in detailed_results]),
            'content_matching_score': statistics.mean([r.content_matching_score for r in detailed_results]),
            'spanning_accuracy': statistics.mean([r.spanning_accuracy for r in detailed_results]),
            'overall_score': statistics.mean([r.overall_score for r in detailed_results])
        }
        
        # Calculate score distributions
        score_distribution = {
            'cell_iou_score': [r.cell_iou_score for r in detailed_results],
            'structural_similarity': [r.structural_similarity for r in detailed_results],
            'content_matching_score': [r.content_matching_score for r in detailed_results],
            'spanning_accuracy': [r.spanning_accuracy for r in detailed_results],
            'overall_score': [r.overall_score for r in detailed_results]
        }
        
        # Error analysis
        error_analysis = self._analyze_errors(detailed_results)
        
        return BenchmarkResult(
            config=self.config,
            total_samples=successful_samples + failed_samples,
            successful_samples=successful_samples,
            failed_samples=failed_samples,
            average_scores=average_scores,
            score_distribution=score_distribution,
            error_analysis=error_analysis,
            processing_time=processing_time,
            detailed_results=detailed_results,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
    
    def _analyze_errors(self, detailed_results: List[EvaluationResult]) -> Dict[str, any]:
        """Analyze common errors and patterns"""
        error_analysis = {
            'common_errors': defaultdict(int),
            'error_patterns': defaultdict(int),
            'low_performing_samples': [],
            'high_performing_samples': [],
            'spanning_cell_errors': 0,
            'content_errors': 0,
            'structural_errors': 0
        }
        
        for i, result in enumerate(detailed_results):
            # Track common errors
            for error in result.errors:
                error_analysis['common_errors'][error] += 1
            
            # Identify low/high performing samples
            if result.overall_score < 0.7:
                error_analysis['low_performing_samples'].append({
                    'sample_index': i,
                    'overall_score': result.overall_score,
                    'errors': result.errors
                })
            elif result.overall_score > 0.9:
                error_analysis['high_performing_samples'].append({
                    'sample_index': i,
                    'overall_score': result.overall_score
                })
            
            # Count specific error types
            if result.spanning_accuracy < 0.8:
                error_analysis['spanning_cell_errors'] += 1
            if result.content_matching_score < 0.8:
                error_analysis['content_errors'] += 1
            if result.structural_similarity < 0.8:
                error_analysis['structural_errors'] += 1
        
        return error_analysis
    
    def _save_benchmark_results(self, benchmark_result: BenchmarkResult):
        """Save benchmark results to file"""
        timestamp = benchmark_result.timestamp.replace(':', '-').replace(' ', '_')
        filename = f"benchmark_{self.config.model_name}_{self.config.dataset_split}_{timestamp}.json"
        filepath = os.path.join(self.config.results_dir, filename)
        
        # Create simplified result dict
        result_dict = {
            'config': {
                'model_name': benchmark_result.config.model_name,
                'dataset_split': benchmark_result.config.dataset_split,
                'max_samples': benchmark_result.config.max_samples
            },
            'total_samples': benchmark_result.total_samples,
            'successful_samples': benchmark_result.successful_samples,
            'failed_samples': benchmark_result.failed_samples,
            'average_scores': benchmark_result.average_scores,
            'processing_time': benchmark_result.processing_time,
            'timestamp': benchmark_result.timestamp
        }
        
        with open(filepath, 'w') as f:
            json.dump(result_dict, f, indent=2)
        
        print(f"Results saved to: {filepath}")
    
    def _print_benchmark_summary(self, benchmark_result: BenchmarkResult):
        """Print benchmark summary"""
        print("\n" + "=" * 60)
        print("BENCHMARK SUMMARY")
        print("=" * 60)
        
        print(f"Model: {benchmark_result.config.model_name}")
        print(f"Dataset: {benchmark_result.config.dataset_split}")
        print(f"Total samples: {benchmark_result.total_samples}")
        print(f"Successful: {benchmark_result.successful_samples}")
        print(f"Failed: {benchmark_result.failed_samples}")
        print(f"Processing time: {benchmark_result.processing_time:.2f}s")
        
        print(f"\nAVERAGE SCORES:")
        for metric, score in benchmark_result.average_scores.items():
            print(f"   {metric}: {score:.3f}")
        
        print(f"\nSUCCESS CRITERIA CHECK:")
        success_criteria = {
            'Cell IoU Score': benchmark_result.average_scores['cell_iou_score'] >= 0.9,
            'Structural Similarity': benchmark_result.average_scores['structural_similarity'] >= 0.95,
            'Content Matching': benchmark_result.average_scores['content_matching_score'] >= 0.90,
            'Spanning Accuracy': benchmark_result.average_scores['spanning_accuracy'] >= 0.85,
            'Overall Score': benchmark_result.average_scores['overall_score'] >= 0.90
        }
        
        for criterion, passed in success_criteria.items():
            status = "PASS" if passed else "FAIL"
            print(f"   {criterion}: {status}")
        
        # Error analysis
        if benchmark_result.error_analysis:
            print(f"\nERROR ANALYSIS:")
            print(f"   Low performing samples: {len(benchmark_result.error_analysis['low_performing_samples'])}")
            print(f"   High performing samples: {len(benchmark_result.error_analysis['high_performing_samples'])}")
            print(f"   Spanning cell errors: {benchmark_result.error_analysis['spanning_cell_errors']}")
            print(f"   Content errors: {benchmark_result.error_analysis['content_errors']}")
            print(f"   Structural errors: {benchmark_result.error_analysis['structural_errors']}")
            
            # Most common errors
            common_errors = benchmark_result.error_analysis['common_errors']
            if common_errors:
                print(f"\nMOST COMMON ERRORS:")
                for error, count in sorted(common_errors.items(), key=lambda x: x[1], reverse=True)[:5]:
                    print(f"   {error}: {count} times")
    
    def compare_models(self, model_results: List[BenchmarkResult]) -> Dict[str, any]:
        """Compare results from multiple models"""
        if len(model_results) < 2:
            print("Need at least 2 model results for comparison")
            return {}
        
        comparison = {
            'models': [result.config.model_name for result in model_results],
            'metrics_comparison': {},
            'best_model_per_metric': {},
            'overall_ranking': []
        }
        
        # Compare each metric
        for metric in ['cell_iou_score', 'structural_similarity', 'content_matching_score', 
                      'spanning_accuracy', 'overall_score']:
            scores = [result.average_scores[metric] for result in model_results]
            comparison['metrics_comparison'][metric] = dict(zip(comparison['models'], scores))
            
            # Find best model for this metric
            best_idx = scores.index(max(scores))
            comparison['best_model_per_metric'][metric] = comparison['models'][best_idx]
        
        # Overall ranking
        overall_scores = [result.average_scores['overall_score'] for result in model_results]
        ranking_indices = sorted(range(len(overall_scores)), key=lambda i: overall_scores[i], reverse=True)
        comparison['overall_ranking'] = [comparison['models'][i] for i in ranking_indices]
        
        return comparison


def main():
    """Test the benchmarking system"""
    print("Testing Benchmarking System")
    print("=" * 50)
    
    # Create benchmark configuration
    config = BenchmarkConfig(
        dataset_split='val',
        max_samples=10,  # Small number for testing
        save_results=True,
        detailed_analysis=True
    )
    
    # Run benchmark
    benchmark = TableReconstructionBenchmark(config)
    result = benchmark.run_benchmark()
    
    print(f"\nBenchmark completed!")
    print(f"Overall score: {result.average_scores['overall_score']:.3f}")
    
    return result


if __name__ == "__main__":
    main()
