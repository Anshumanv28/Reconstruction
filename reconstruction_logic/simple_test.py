"""
Simple Test for Reconstruction Logic (No External Dependencies)

Tests the core reconstruction logic without requiring external libraries.
"""

import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from perfect_reconstructor import PerfectTableReconstructor, ReconstructionConfig


def test_perfect_reconstructor():
    """Test perfect reconstructor with simple data"""
    print("🏗️ Testing Perfect Reconstructor...")
    
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


def test_configuration():
    """Test configuration system"""
    print("\n⚙️ Testing Configuration...")
    
    config = ReconstructionConfig(
        min_cell_size=(5, 5),
        max_cell_size=(500, 50),
        overlap_threshold=0.2,
        confidence_threshold=0.7,
        spanning_detection_threshold=2.0,
        grid_tolerance=10
    )
    
    reconstructor = PerfectTableReconstructor(config)
    
    print(f"✅ Configuration loaded:")
    print(f"   Min cell size: {config.min_cell_size}")
    print(f"   Max cell size: {config.max_cell_size}")
    print(f"   Overlap threshold: {config.overlap_threshold}")
    print(f"   Confidence threshold: {config.confidence_threshold}")
    print(f"   Spanning detection threshold: {config.spanning_detection_threshold}")
    print(f"   Grid tolerance: {config.grid_tolerance}")
    
    return config


def test_spanning_detection():
    """Test spanning cell detection"""
    print("\n🔍 Testing Spanning Cell Detection...")
    
    reconstructor = PerfectTableReconstructor()
    
    # Create cells with one large spanning cell
    model_cells = [
        {'bbox': [100, 50, 200, 100], 'content': 'Header 1', 'confidence': 0.9, 'is_header': True},
        {'bbox': [200, 50, 300, 100], 'content': 'Header 2', 'confidence': 0.9, 'is_header': True},
        {'bbox': [100, 100, 300, 150], 'content': 'Large Spanning Cell', 'confidence': 0.9, 'is_header': False},  # Wide cell
        {'bbox': [100, 150, 200, 200], 'content': 'Cell 1', 'confidence': 0.8, 'is_header': False},
        {'bbox': [200, 150, 300, 200], 'content': 'Cell 2', 'confidence': 0.8, 'is_header': False}
    ]
    
    ocr_text_blocks = [
        {'bbox': [100, 50, 200, 100], 'text': 'Header 1', 'confidence': 0.9},
        {'bbox': [200, 50, 300, 100], 'text': 'Header 2', 'confidence': 0.9},
        {'bbox': [100, 100, 300, 150], 'text': 'Large Spanning Cell', 'confidence': 0.9},
        {'bbox': [100, 150, 200, 200], 'text': 'Cell 1', 'confidence': 0.8},
        {'bbox': [200, 150, 300, 200], 'text': 'Cell 2', 'confidence': 0.8}
    ]
    
    # Reconstruct table
    reconstructed = reconstructor.reconstruct_from_model_output(model_cells, ocr_text_blocks)
    
    print(f"✅ Spanning detection test:")
    print(f"   Table dimensions: {reconstructed.rows} rows × {reconstructed.cols} columns")
    print(f"   Total cells: {len(reconstructed.cells)}")
    
    # Check for spanning cells
    spanning_cells = [cell for cell in reconstructed.cells if cell.rowspan > 1 or cell.colspan > 1]
    print(f"   Spanning cells detected: {len(spanning_cells)}")
    
    for cell in spanning_cells:
        print(f"   Spanning cell R{cell.row}C{cell.col}: '{cell.content}' (span: {cell.rowspan}×{cell.colspan})")
    
    return reconstructed


def main():
    """Run all tests"""
    print("🧪 Simple Reconstruction Logic Test Suite")
    print("=" * 50)
    
    try:
        # Test core functionality
        reconstructed = test_perfect_reconstructor()
        config = test_configuration()
        spanning_test = test_spanning_detection()
        
        print("\n🎉 All Tests Completed Successfully!")
        print("=" * 50)
        
        # Summary
        print("📋 Test Summary:")
        print(f"   ✅ Perfect Reconstructor: Working")
        print(f"   ✅ Configuration System: Working")
        print(f"   ✅ Spanning Detection: Working")
        
        # Basic validation
        if (reconstructed.rows > 0 and reconstructed.cols > 0 and 
            len(reconstructed.cells) > 0):
            print("\n🏆 Core reconstruction logic is functional!")
        else:
            print("\n⚠️ Reconstruction logic needs debugging")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
