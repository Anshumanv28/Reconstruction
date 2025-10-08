"""
Simple test for benchmarking system without emojis
"""

import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from benchmarking_system import TableReconstructionBenchmark, BenchmarkConfig


def main():
    """Test the benchmarking system"""
    print("Testing Benchmarking System")
    print("=" * 50)
    
    # Create benchmark configuration
    config = BenchmarkConfig(
        dataset_split='val',
        max_samples=2,  # Very small for testing
        save_results=True,
        detailed_analysis=False  # Keep results small
    )
    
    # Run benchmark
    benchmark = TableReconstructionBenchmark(config)
    result = benchmark.run_benchmark()
    
    print(f"\nBenchmark completed!")
    print(f"Overall score: {result.average_scores['overall_score']:.3f}")
    print(f"Samples processed: {result.successful_samples}")
    print(f"Processing time: {result.processing_time:.2f}s")
    
    return result


if __name__ == "__main__":
    main()
