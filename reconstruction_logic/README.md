# Bottom-Up Table Reconstruction Logic

This directory contains the core reconstruction logic for creating perfect table structures using PubTabNet as the gold standard.

## 🎯 Purpose

This branch focuses **only** on the reconstruction logic itself. The main branch handles:
- Model integration (TableFormer, Table Transformer)
- Pipeline orchestration
- Visualization and output generation
- Data loading and preprocessing

## 📁 Contents

- `perfect_reconstructor.py` - Core reconstruction algorithms
- `html_parser.py` - PubTabNet HTML annotation parser
- `evaluation_metrics.py` - Reconstruction quality metrics
- `README.md` - This file

## 🔧 Core Components

### Perfect Reconstructor
- Hybrid approach combining TableFormer cells + OCR text
- Intelligent spanning cell detection
- Weighted averaging of element positions
- Model-agnostic design

### HTML Parser
- Extracts cell coordinates, spans, and content from PubTabNet
- Validates table structure
- Handles complex nested tables

### Evaluation Metrics
- Cell-level IoU scoring
- Structural similarity metrics
- Content matching algorithms
- Spanning cell accuracy

## 🚀 Usage

```python
from reconstruction_logic.perfect_reconstructor import PerfectTableReconstructor
from reconstruction_logic.html_parser import PubTabNetHTMLParser

# Parse ground truth
parser = PubTabNetHTMLParser()
ground_truth = parser.parse_html_annotation(html_string)

# Reconstruct from model output
reconstructor = PerfectTableReconstructor()
reconstructed = reconstructor.reconstruct_from_model_output(
    model_cells, ocr_text_blocks, ground_truth
)
```

## 📊 Success Criteria

- **IoU Score**: >0.9 for cell boundaries
- **Structural Accuracy**: >95% row/column matches
- **Content Fidelity**: >90% text overlap
- **Spanning Detection**: >85% accuracy
- **Overall Performance**: >90% reconstruction quality

This focused approach ensures the reconstruction logic is robust, testable, and can be easily integrated into the main pipeline.
