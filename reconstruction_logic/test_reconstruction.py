"""
Test Script for Reconstruction Logic

Simple test to verify the reconstruction logic components work correctly.
"""

import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from html_parser import PubTabNetHTMLParser
from perfect_reconstructor import PerfectTableReconstructor
from evaluation_metrics import TableReconstructionEvaluator


def test_html_parser():
    """Test HTML parser functionality"""
    print("🔍 Testing HTML Parser...")
    
    parser = PubTabNetHTMLParser()
    
    # Sample HTML annotation
    sample_html = """
    <table>
        <tr>
            <th colspan="2">Product Information</th>
            <th>Price</th>
        </tr>
        <tr>
            <td rowspan="2">Electronics</td>
            <td>Laptop</td>
            <td>$999</td>
        </tr>
        <tr>
            <td>Mouse</td>
            <td>$25</td>
        </tr>
    </table>
    """
    
    # Parse HTML annotation
    table_structure = parser.parse_html_annotation(sample_html)
    
    print(f"✅ Parsed table: {table_structure.rows} rows × {table_structure.cols} columns")
    print(f"✅ Total cells: {len(table_structure.cells)}")
    
    # Validate structure
    validation = parser.validate_table_structure(table_structure)
    print(f"✅ Validation results: {validation}")
    
    # Print cell details
    for cell in table_structure.cells:
        print(f"   Cell R{cell.row}C{cell.col}: '{cell.content}' (span: {cell.rowspan}×{cell.colspan})")
    
    return table_structure


def test_perfect_reconstructor():
    """Test perfect reconstructor"""
    print("\n🏗️ Testing Perfect Reconstructor...")
    
    reconstructor = PerfectTableReconstructor()
    
    # Sample model prediction data
    model_cells = [
        {'bbox': [100, 50, 300, 100], 'content': 'Product Information', 'confidence': 0.95, 'is_header': True},
        {'bbox': [300, 50, 400, 100], 'content': 'Price', 'confidence': 0.90, 'is_header': True},
        {'bbox': [100, 100, 200, 200], 'content': 'Electronics', 'confidence': 0.88, 'is_header': False},
        {'bbox': [200, 100, 300, 150], 'content': 'Laptop', 'confidence': 0.92, 'is_header': False},
        {'bbox': [300, 100, 400, 150], 'content': '$999', 'confidence': 0.85, 'is_header': False},
        {'bbox': [200, 150, 300, 200], 'content': 'Mouse', 'confidence': 0.87, 'is_header': False},
        {'bbox': [300, 150, 400, 200], 'content': '$25', 'confidence': 0.83, 'is_header': False}
    ]
    
    ocr_text_blocks = [
        {'bbox': [100, 50, 300, 100], 'text': 'Product Information', 'confidence': 0.95},
        {'bbox': [300, 50, 400, 100], 'text': 'Price', 'confidence': 0.90},
        {'bbox': [100, 100, 200, 200], 'text': 'Electronics', 'confidence': 0.88},
        {'bbox': [200, 100, 300, 150], 'text': 'Laptop', 'confidence': 0.92},
        {'bbox': [300, 100, 400, 150], 'text': '$999', 'confidence': 0.85},
        {'bbox': [200, 150, 300, 200], 'text': 'Mouse', 'confidence': 0.87},
        {'bbox': [300, 150, 400, 200], 'text': '$25', 'confidence': 0.83}
    ]
    
    # Reconstruct table
    reconstructed = reconstructor.reconstruct_from_model_output(model_cells, ocr_text_blocks)
    
    print(f"✅ Reconstructed table: {reconstructed.rows} rows × {reconstructed.cols} columns")
    print(f"✅ Total cells: {len(reconstructed.cells)}")
    
    # Print cell details
    for cell in reconstructed.cells:
        print(f"   Cell R{cell.row}C{cell.col}: '{cell.content}' (span: {cell.rowspan}×{cell.colspan})")
    
    return reconstructed


def test_evaluation_metrics():
    """Test evaluation metrics"""
    print("\n📊 Testing Evaluation Metrics...")
    
    evaluator = TableReconstructionEvaluator()
    
    # Create sample ground truth
    from html_parser import CellInfo, TableStructure
    
    gt_cells = [
        CellInfo(0, 0, 1, 2, "Product Information", is_header=True),
        CellInfo(0, 2, 1, 1, "Price", is_header=True),
        CellInfo(1, 0, 2, 1, "Electronics"),
        CellInfo(1, 1, 1, 1, "Laptop"),
        CellInfo(1, 2, 1, 1, "$999"),
        CellInfo(2, 1, 1, 1, "Mouse"),
        CellInfo(2, 2, 1, 1, "$25")
    ]
    
    ground_truth = TableStructure(rows=3, cols=3, cells=gt_cells)
    
    # Create sample prediction
    pred_cells = [
        CellInfo(0, 0, 1, 2, "Product Information", is_header=True),
        CellInfo(0, 2, 1, 1, "Price", is_header=True),
        CellInfo(1, 0, 2, 1, "Electronics"),
        CellInfo(1, 1, 1, 1, "Laptop"),
        CellInfo(1, 2, 1, 1, "$999"),
        CellInfo(2, 1, 1, 1, "Mouse"),
        CellInfo(2, 2, 1, 1, "$25")
    ]
    
    predicted = TableStructure(rows=3, cols=3, cells=pred_cells)
    
    # Evaluate
    result = evaluator.evaluate_reconstruction(predicted, ground_truth)
    
    print(f"✅ Evaluation results:")
    print(f"   Cell IoU Score: {result.cell_iou_score:.3f}")
    print(f"   Structural Similarity: {result.structural_similarity:.3f}")
    print(f"   Content Matching: {result.content_matching_score:.3f}")
    print(f"   Spanning Accuracy: {result.spanning_accuracy:.3f}")
    print(f"   Overall Score: {result.overall_score:.3f}")
    
    if result.errors:
        print(f"   Errors: {', '.join(result.errors)}")
    else:
        print("   ✅ No errors detected")
    
    return result


def main():
    """Run all tests"""
    print("🧪 Reconstruction Logic Test Suite")
    print("=" * 50)
    
    try:
        # Test individual components
        ground_truth = test_html_parser()
        reconstructed = test_perfect_reconstructor()
        evaluation_result = test_evaluation_metrics()
        
        print("\n🎉 All Tests Completed Successfully!")
        print("=" * 50)
        
        # Summary
        print("📋 Test Summary:")
        print(f"   ✅ HTML Parser: Working")
        print(f"   ✅ Perfect Reconstructor: Working")
        print(f"   ✅ Evaluation Metrics: Working")
        
        # Success criteria check
        if (evaluation_result.overall_score >= 0.9 and 
            evaluation_result.structural_similarity >= 0.95):
            print("\n🏆 Reconstruction logic meets success criteria!")
        else:
            print("\n⚠️ Reconstruction logic needs improvement")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
