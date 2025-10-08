#!/usr/bin/env python3
"""
Test Enhanced Dimension-Preserving Reconstruction V2 with More Samples.
"""

from enhanced_dimension_preserving_reconstructor_v2 import EnhancedDimensionPreservingReconstructorV2

def main():
    """Test enhanced reconstruction V2 with more samples."""
    print("Testing Enhanced Table Reconstruction V2 with More Samples")
    print("=" * 60)
    
    # Initialize reconstructor
    reconstructor = EnhancedDimensionPreservingReconstructorV2()
    
    # Test reconstruction with more samples
    analysis = reconstructor.test_dimension_preserving_reconstruction(max_samples=10)
    
    if analysis:
        print(f"\nEnhanced Reconstruction Results V2:")
        print(f"   Total samples: {analysis['total_samples']}")
        print(f"   Success rate: {analysis['success_rate']:.1f}%")
        print(f"   Average dimension preservation: {analysis['average_dimension_preservation']:.3f}")
        print(f"   Average structural accuracy: {analysis['average_structural_accuracy']:.3f}")
        print(f"   Average HTML similarity: {analysis['average_html_similarity']:.3f}")
        print(f"   Average overall quality: {analysis['average_overall_quality']:.3f}")
        print(f"   Total spanning cells: {analysis['total_spanning_cells']}")
        
        if analysis['spanning_cell_analysis']:
            spanning = analysis['spanning_cell_analysis']
            print(f"   Average spanning cells per table: {spanning['avg_spanning_per_table']:.1f}")
            print(f"   Tables with spanning cells: {spanning['tables_with_spanning']}")

if __name__ == "__main__":
    main()
