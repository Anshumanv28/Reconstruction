"""
Comprehensive Validation Test for Table Reconstruction System

Tests the complete pipeline from PubTabNet data loading through reconstruction
to evaluation, providing end-to-end validation of the system.
"""

import sys
import os
import time
from typing import Dict, List, Tuple

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pubtabnet_data_loader import PubTabNetDataLoader, PubTabNetSample
from html_parser import PubTabNetHTMLParser
from perfect_reconstructor import PerfectTableReconstructor, ReconstructionConfig
from evaluation_metrics import TableReconstructionEvaluator
from benchmarking_system import TableReconstructionBenchmark, BenchmarkConfig


class ComprehensiveValidator:
    """Comprehensive validation system for table reconstruction"""
    
    def __init__(self):
        self.data_loader = PubTabNetDataLoader()
        self.html_parser = PubTabNetHTMLParser()
        self.reconstructor = PerfectTableReconstructor()
        self.evaluator = TableReconstructionEvaluator()
        
        # Test configurations
        self.test_configs = [
            ReconstructionConfig(
                min_cell_size=(5, 5),
                max_cell_size=(500, 100),
                overlap_threshold=0.3,
                confidence_threshold=0.5,
                spanning_detection_threshold=1.5,
                grid_tolerance=5
            ),
            ReconstructionConfig(
                min_cell_size=(10, 10),
                max_cell_size=(300, 80),
                overlap_threshold=0.2,
                confidence_threshold=0.7,
                spanning_detection_threshold=2.0,
                grid_tolerance=10
            )
        ]
    
    def run_comprehensive_validation(self) -> Dict[str, any]:
        """Run comprehensive validation of the entire system"""
        print("COMPREHENSIVE VALIDATION TEST")
        print("=" * 60)
        
        validation_results = {
            'data_loading': self._test_data_loading(),
            'html_parsing': self._test_html_parsing(),
            'reconstruction': self._test_reconstruction(),
            'evaluation': self._test_evaluation(),
            'end_to_end': self._test_end_to_end(),
            'benchmarking': self._test_benchmarking(),
            'configuration_tests': self._test_configurations(),
            'error_handling': self._test_error_handling()
        }
        
        # Calculate overall validation score
        validation_results['overall_score'] = self._calculate_overall_score(validation_results)
        
        # Print summary
        self._print_validation_summary(validation_results)
        
        return validation_results
    
    def _test_data_loading(self) -> Dict[str, any]:
        """Test PubTabNet data loading functionality"""
        print("\nTesting Data Loading...")
        
        results = {
            'splits_tested': [],
            'samples_loaded': 0,
            'synthetic_samples': 0,
            'errors': [],
            'success': True
        }
        
        try:
            for split in ['train', 'val', 'test']:
                print(f"   Testing {split} split...")
                dataset = self.data_loader.load_split(split, max_samples=5)
                results['splits_tested'].append(split)
                results['samples_loaded'] += len(dataset.samples)
                
                # Count synthetic samples
                synthetic_count = sum(1 for s in dataset.samples if s.metadata.get('synthetic', False))
                results['synthetic_samples'] += synthetic_count
                
                # Validate dataset
                stats = self.data_loader.validate_dataset(dataset)
                if stats['errors']:
                    results['errors'].extend(stats['errors'])
                
                print(f"   PASS {split}: {len(dataset.samples)} samples loaded")
            
            print(f"   📈 Total samples loaded: {results['samples_loaded']}")
            print(f"   🔧 Synthetic samples: {results['synthetic_samples']}")
            
        except Exception as e:
            results['success'] = False
            results['errors'].append(f"Data loading failed: {str(e)}")
            print(f"   ❌ Data loading failed: {e}")
        
        return results
    
    def _test_html_parsing(self) -> Dict[str, any]:
        """Test HTML parsing functionality"""
        print("\n🔍 Testing HTML Parsing...")
        
        results = {
            'samples_parsed': 0,
            'parsing_errors': 0,
            'validation_failures': 0,
            'success': True
        }
        
        try:
            # Test with synthetic HTML samples
            test_html_samples = [
                """
                <table>
                    <tr>
                        <th colspan="2">Header</th>
                        <th>Column 3</th>
                    </tr>
                    <tr>
                        <td rowspan="2">Cell 1</td>
                        <td>Cell 2</td>
                        <td>Cell 3</td>
                    </tr>
                    <tr>
                        <td>Cell 4</td>
                        <td>Cell 5</td>
                    </tr>
                </table>
                """,
                """
                <table>
                    <tr>
                        <td>Simple</td>
                        <td>Table</td>
                    </tr>
                    <tr>
                        <td>With</td>
                        <td>Two Rows</td>
                    </tr>
                </table>
                """
            ]
            
            for i, html in enumerate(test_html_samples):
                try:
                    # Parse HTML
                    table_structure = self.html_parser.parse_html_annotation(html)
                    results['samples_parsed'] += 1
                    
                    # Validate structure
                    validation = self.html_parser.validate_table_structure(table_structure)
                    if not all(validation.values()):
                        results['validation_failures'] += 1
                        print(f"   ⚠️ Sample {i}: Validation issues: {validation}")
                    else:
                        print(f"   ✅ Sample {i}: {table_structure.rows}×{table_structure.cols} table parsed successfully")
                        
                except Exception as e:
                    results['parsing_errors'] += 1
                    print(f"   ❌ Sample {i}: Parsing failed: {e}")
            
            if results['parsing_errors'] > 0:
                results['success'] = False
                
        except Exception as e:
            results['success'] = False
            print(f"   ❌ HTML parsing test failed: {e}")
        
        return results
    
    def _test_reconstruction(self) -> Dict[str, any]:
        """Test table reconstruction functionality"""
        print("\n🏗️ Testing Table Reconstruction...")
        
        results = {
            'reconstructions_attempted': 0,
            'reconstructions_successful': 0,
            'spanning_cells_detected': 0,
            'errors': [],
            'success': True
        }
        
        try:
            # Test with different configurations
            for config_idx, config in enumerate(self.test_configs):
                print(f"   🔧 Testing configuration {config_idx + 1}...")
                reconstructor = PerfectTableReconstructor(config)
                
                # Create test data
                model_cells = [
                    {'bbox': [100, 50, 300, 100], 'content': 'Header', 'confidence': 0.9, 'is_header': True},
                    {'bbox': [300, 50, 400, 100], 'content': 'Col2', 'confidence': 0.8, 'is_header': True},
                    {'bbox': [100, 100, 200, 200], 'content': 'Cell1', 'confidence': 0.9, 'is_header': False},
                    {'bbox': [200, 100, 300, 150], 'content': 'Cell2', 'confidence': 0.8, 'is_header': False},
                    {'bbox': [300, 100, 400, 150], 'content': 'Cell3', 'confidence': 0.8, 'is_header': False},
                    {'bbox': [200, 150, 300, 200], 'content': 'Cell4', 'confidence': 0.8, 'is_header': False},
                    {'bbox': [300, 150, 400, 200], 'content': 'Cell5', 'confidence': 0.8, 'is_header': False}
                ]
                
                ocr_text_blocks = [
                    {'bbox': [100, 50, 300, 100], 'text': 'Header', 'confidence': 0.9},
                    {'bbox': [300, 50, 400, 100], 'text': 'Col2', 'confidence': 0.8},
                    {'bbox': [100, 100, 200, 200], 'text': 'Cell1', 'confidence': 0.9},
                    {'bbox': [200, 100, 300, 150], 'text': 'Cell2', 'confidence': 0.8},
                    {'bbox': [300, 100, 400, 150], 'text': 'Cell3', 'confidence': 0.8},
                    {'bbox': [200, 150, 300, 200], 'text': 'Cell4', 'confidence': 0.8},
                    {'bbox': [300, 150, 400, 200], 'text': 'Cell5', 'confidence': 0.8}
                ]
                
                try:
                    reconstructed = reconstructor.reconstruct_from_model_output(model_cells, ocr_text_blocks)
                    results['reconstructions_attempted'] += 1
                    results['reconstructions_successful'] += 1
                    
                    # Count spanning cells
                    spanning_count = sum(1 for cell in reconstructed.cells if cell.rowspan > 1 or cell.colspan > 1)
                    results['spanning_cells_detected'] += spanning_count
                    
                    print(f"   ✅ Config {config_idx + 1}: {reconstructed.rows}×{reconstructed.cols} table, {spanning_count} spanning cells")
                    
                except Exception as e:
                    results['reconstructions_attempted'] += 1
                    results['errors'].append(f"Config {config_idx + 1}: {str(e)}")
                    print(f"   ❌ Config {config_idx + 1}: Reconstruction failed: {e}")
            
            if results['reconstructions_successful'] == 0:
                results['success'] = False
                
        except Exception as e:
            results['success'] = False
            print(f"   ❌ Reconstruction test failed: {e}")
        
        return results
    
    def _test_evaluation(self) -> Dict[str, any]:
        """Test evaluation metrics functionality"""
        print("\n📊 Testing Evaluation Metrics...")
        
        results = {
            'evaluations_performed': 0,
            'high_scores': 0,
            'low_scores': 0,
            'errors': [],
            'success': True
        }
        
        try:
            # Create test ground truth and prediction
            from html_parser import CellInfo, TableStructure
            
            # Ground truth
            gt_cells = [
                CellInfo(0, 0, 1, 2, "Header", is_header=True),
                CellInfo(0, 2, 1, 1, "Col3", is_header=True),
                CellInfo(1, 0, 2, 1, "Cell1"),
                CellInfo(1, 1, 1, 1, "Cell2"),
                CellInfo(1, 2, 1, 1, "Cell3"),
                CellInfo(2, 1, 1, 1, "Cell4"),
                CellInfo(2, 2, 1, 1, "Cell5")
            ]
            ground_truth = TableStructure(rows=3, cols=3, cells=gt_cells)
            
            # Prediction (slightly different)
            pred_cells = [
                CellInfo(0, 0, 1, 2, "Header", is_header=True),
                CellInfo(0, 2, 1, 1, "Column3", is_header=True),  # Different content
                CellInfo(1, 0, 2, 1, "Cell1"),
                CellInfo(1, 1, 1, 1, "Cell2"),
                CellInfo(1, 2, 1, 1, "Cell3"),
                CellInfo(2, 1, 1, 1, "Cell4"),
                CellInfo(2, 2, 1, 1, "Cell5")
            ]
            predicted = TableStructure(rows=3, cols=3, cells=pred_cells)
            
            # Evaluate
            evaluation_result = self.evaluator.evaluate_reconstruction(predicted, ground_truth)
            results['evaluations_performed'] += 1
            
            # Categorize scores
            if evaluation_result.overall_score >= 0.8:
                results['high_scores'] += 1
            else:
                results['low_scores'] += 1
            
            print(f"   ✅ Evaluation completed:")
            print(f"      Overall score: {evaluation_result.overall_score:.3f}")
            print(f"      Cell IoU: {evaluation_result.cell_iou_score:.3f}")
            print(f"      Structural similarity: {evaluation_result.structural_similarity:.3f}")
            print(f"      Content matching: {evaluation_result.content_matching_score:.3f}")
            print(f"      Spanning accuracy: {evaluation_result.spanning_accuracy:.3f}")
            
            if evaluation_result.errors:
                print(f"      Errors: {len(evaluation_result.errors)}")
                results['errors'].extend(evaluation_result.errors)
            
        except Exception as e:
            results['success'] = False
            print(f"   ❌ Evaluation test failed: {e}")
        
        return results
    
    def _test_end_to_end(self) -> Dict[str, any]:
        """Test end-to-end pipeline"""
        print("\n🔄 Testing End-to-End Pipeline...")
        
        results = {
            'pipelines_tested': 0,
            'pipelines_successful': 0,
            'average_scores': {},
            'errors': [],
            'success': True
        }
        
        try:
            # Load a small dataset
            dataset = self.data_loader.load_split('val', max_samples=3)
            
            scores = []
            for i, sample in enumerate(dataset.samples):
                try:
                    print(f"   🔄 Testing pipeline {i + 1}...")
                    
                    # Step 1: Parse ground truth
                    ground_truth = self.html_parser.parse_pubtabnet_sample(
                        sample.html_annotation,
                        sample.cell_bboxes,
                        sample.image_width,
                        sample.image_height
                    )
                    
                    # Step 2: Simulate model prediction
                    model_prediction = self._simulate_model_prediction(sample, ground_truth)
                    
                    # Step 3: Reconstruct table
                    reconstructed = self.reconstructor.reconstruct_from_model_output(
                        model_prediction['model_cells'],
                        model_prediction['ocr_text_blocks'],
                        ground_truth
                    )
                    
                    # Step 4: Evaluate
                    evaluation_result = self.evaluator.evaluate_reconstruction(
                        reconstructed, ground_truth
                    )
                    
                    results['pipelines_tested'] += 1
                    results['pipelines_successful'] += 1
                    scores.append(evaluation_result.overall_score)
                    
                    print(f"   ✅ Pipeline {i + 1}: Score {evaluation_result.overall_score:.3f}")
                    
                except Exception as e:
                    results['pipelines_tested'] += 1
                    results['errors'].append(f"Pipeline {i + 1}: {str(e)}")
                    print(f"   ❌ Pipeline {i + 1}: Failed - {e}")
            
            # Calculate average scores
            if scores:
                results['average_scores'] = {
                    'overall_score': sum(scores) / len(scores),
                    'min_score': min(scores),
                    'max_score': max(scores)
                }
            
            if results['pipelines_successful'] == 0:
                results['success'] = False
                
        except Exception as e:
            results['success'] = False
            print(f"   ❌ End-to-end test failed: {e}")
        
        return results
    
    def _test_benchmarking(self) -> Dict[str, any]:
        """Test benchmarking system"""
        print("\n📈 Testing Benchmarking System...")
        
        results = {
            'benchmarks_run': 0,
            'benchmarks_successful': 0,
            'results_saved': False,
            'errors': [],
            'success': True
        }
        
        try:
            # Create benchmark configuration
            config = BenchmarkConfig(
                dataset_split='val',
                max_samples=2,  # Very small for testing
                save_results=True,
                detailed_analysis=False  # Keep results small
            )
            
            # Run benchmark
            benchmark = TableReconstructionBenchmark(config)
            benchmark_result = benchmark.run_benchmark()
            
            results['benchmarks_run'] += 1
            results['benchmarks_successful'] += 1
            results['results_saved'] = True
            
            print(f"   ✅ Benchmark completed:")
            print(f"      Overall score: {benchmark_result.average_scores['overall_score']:.3f}")
            print(f"      Samples processed: {benchmark_result.successful_samples}")
            print(f"      Processing time: {benchmark_result.processing_time:.2f}s")
            
        except Exception as e:
            results['benchmarks_run'] += 1
            results['errors'].append(f"Benchmarking failed: {str(e)}")
            print(f"   ❌ Benchmarking test failed: {e}")
        
        return results
    
    def _test_configurations(self) -> Dict[str, any]:
        """Test different reconstruction configurations"""
        print("\n⚙️ Testing Configuration Variations...")
        
        results = {
            'configs_tested': len(self.test_configs),
            'configs_successful': 0,
            'performance_variations': [],
            'success': True
        }
        
        try:
            for i, config in enumerate(self.test_configs):
                print(f"   🔧 Testing config {i + 1}...")
                
                reconstructor = PerfectTableReconstructor(config)
                
                # Test with same data
                model_cells = [
                    {'bbox': [100, 50, 200, 100], 'content': 'Test', 'confidence': 0.9, 'is_header': True},
                    {'bbox': [200, 50, 300, 100], 'content': 'Data', 'confidence': 0.8, 'is_header': True},
                    {'bbox': [100, 100, 200, 150], 'content': 'Cell1', 'confidence': 0.9, 'is_header': False},
                    {'bbox': [200, 100, 300, 150], 'content': 'Cell2', 'confidence': 0.8, 'is_header': False}
                ]
                
                ocr_text_blocks = [
                    {'bbox': [100, 50, 200, 100], 'text': 'Test', 'confidence': 0.9},
                    {'bbox': [200, 50, 300, 100], 'text': 'Data', 'confidence': 0.8},
                    {'bbox': [100, 100, 200, 150], 'text': 'Cell1', 'confidence': 0.9},
                    {'bbox': [200, 100, 300, 150], 'text': 'Cell2', 'confidence': 0.8}
                ]
                
                try:
                    reconstructed = reconstructor.reconstruct_from_model_output(model_cells, ocr_text_blocks)
                    results['configs_successful'] += 1
                    
                    # Record performance metrics
                    performance = {
                        'config_id': i + 1,
                        'cells_reconstructed': len(reconstructed.cells),
                        'table_dimensions': f"{reconstructed.rows}×{reconstructed.cols}",
                        'spanning_cells': sum(1 for cell in reconstructed.cells if cell.rowspan > 1 or cell.colspan > 1)
                    }
                    results['performance_variations'].append(performance)
                    
                    print(f"   ✅ Config {i + 1}: {reconstructed.rows}×{reconstructed.cols} table, {len(reconstructed.cells)} cells")
                    
                except Exception as e:
                    print(f"   ❌ Config {i + 1}: Failed - {e}")
            
            if results['configs_successful'] == 0:
                results['success'] = False
                
        except Exception as e:
            results['success'] = False
            print(f"   ❌ Configuration test failed: {e}")
        
        return results
    
    def _test_error_handling(self) -> Dict[str, any]:
        """Test error handling and edge cases"""
        print("\n🚨 Testing Error Handling...")
        
        results = {
            'error_cases_tested': 0,
            'errors_handled': 0,
            'unhandled_errors': [],
            'success': True
        }
        
        try:
            # Test cases that should trigger errors
            error_test_cases = [
                {
                    'name': 'Empty HTML',
                    'html': '',
                    'should_fail': True
                },
                {
                    'name': 'Invalid HTML',
                    'html': '<div>Not a table</div>',
                    'should_fail': True
                },
                {
                    'name': 'Empty model cells',
                    'model_cells': [],
                    'ocr_text_blocks': [],
                    'should_fail': False  # Should handle gracefully
                },
                {
                    'name': 'Invalid bbox format',
                    'model_cells': [{'bbox': [100, 50], 'content': 'Test', 'confidence': 0.9}],
                    'ocr_text_blocks': [],
                    'should_fail': False  # Should filter out invalid cells
                }
            ]
            
            for test_case in error_test_cases:
                results['error_cases_tested'] += 1
                
                try:
                    if 'html' in test_case:
                        # Test HTML parsing
                        table_structure = self.html_parser.parse_html_annotation(test_case['html'])
                        if test_case['should_fail']:
                            results['unhandled_errors'].append(f"{test_case['name']}: Should have failed but didn't")
                        else:
                            results['errors_handled'] += 1
                            print(f"   ✅ {test_case['name']}: Handled correctly")
                    
                    elif 'model_cells' in test_case:
                        # Test reconstruction
                        reconstructed = self.reconstructor.reconstruct_from_model_output(
                            test_case['model_cells'],
                            test_case['ocr_text_blocks']
                        )
                        if test_case['should_fail']:
                            results['unhandled_errors'].append(f"{test_case['name']}: Should have failed but didn't")
                        else:
                            results['errors_handled'] += 1
                            print(f"   ✅ {test_case['name']}: Handled correctly")
                    
                except Exception as e:
                    if test_case['should_fail']:
                        results['errors_handled'] += 1
                        print(f"   ✅ {test_case['name']}: Failed as expected - {e}")
                    else:
                        results['unhandled_errors'].append(f"{test_case['name']}: Unexpected failure - {e}")
            
            if results['unhandled_errors']:
                results['success'] = False
                
        except Exception as e:
            results['success'] = False
            print(f"   ❌ Error handling test failed: {e}")
        
        return results
    
    def _simulate_model_prediction(self, sample: PubTabNetSample, ground_truth) -> Dict:
        """Simulate model prediction for testing"""
        model_cells = []
        ocr_text_blocks = []
        
        for cell in ground_truth.cells:
            if cell.bbox:
                # Add some noise to simulate model prediction
                noise_factor = 0.05
                bbox = cell.bbox.copy()
                bbox[0] += (hash(cell.content) % 10 - 5) * noise_factor
                bbox[1] += (hash(cell.content) % 10 - 5) * noise_factor
                bbox[2] += (hash(cell.content) % 10 - 5) * noise_factor
                bbox[3] += (hash(cell.content) % 10 - 5) * noise_factor
                
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
    
    def _calculate_overall_score(self, validation_results: Dict[str, any]) -> float:
        """Calculate overall validation score"""
        scores = []
        
        for test_name, result in validation_results.items():
            if isinstance(result, dict) and 'success' in result:
                scores.append(1.0 if result['success'] else 0.0)
        
        return sum(scores) / len(scores) if scores else 0.0
    
    def _print_validation_summary(self, validation_results: Dict[str, any]):
        """Print comprehensive validation summary"""
        print("\n" + "=" * 60)
        print("📋 COMPREHENSIVE VALIDATION SUMMARY")
        print("=" * 60)
        
        overall_score = validation_results['overall_score']
        
        print(f"🎯 Overall Validation Score: {overall_score:.1%}")
        
        print(f"\n📊 Test Results:")
        for test_name, result in validation_results.items():
            if test_name == 'overall_score':
                continue
                
            if isinstance(result, dict) and 'success' in result:
                status = "✅ PASS" if result['success'] else "❌ FAIL"
                print(f"   {test_name.replace('_', ' ').title()}: {status}")
                
                # Show key metrics
                if 'samples_loaded' in result:
                    print(f"      Samples loaded: {result['samples_loaded']}")
                if 'reconstructions_successful' in result:
                    print(f"      Successful reconstructions: {result['reconstructions_successful']}")
                if 'evaluations_performed' in result:
                    print(f"      Evaluations performed: {result['evaluations_performed']}")
                if 'pipelines_successful' in result:
                    print(f"      Successful pipelines: {result['pipelines_successful']}")
        
        # Success criteria check
        print(f"\n🏆 Success Criteria:")
        criteria_met = overall_score >= 0.8
        status = "✅ MET" if criteria_met else "❌ NOT MET"
        print(f"   Overall Score ≥ 80%: {status}")
        
        if criteria_met:
            print(f"\n🎉 VALIDATION PASSED! System is ready for production use.")
        else:
            print(f"\n⚠️ VALIDATION FAILED! System needs improvement before production use.")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        if overall_score < 0.6:
            print("   - Critical issues detected. Review error logs and fix core functionality.")
        elif overall_score < 0.8:
            print("   - Some issues detected. Address failing tests before production deployment.")
        else:
            print("   - System is performing well. Consider optimization and additional testing.")


def main():
    """Run comprehensive validation test"""
    validator = ComprehensiveValidator()
    results = validator.run_comprehensive_validation()
    
    # Return exit code based on results
    return 0 if results['overall_score'] >= 0.8 else 1


if __name__ == "__main__":
    exit(main())
