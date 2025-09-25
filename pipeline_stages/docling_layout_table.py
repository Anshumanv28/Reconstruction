#!/usr/bin/env python3
"""
Stage 1: Layout & Table Detection

This module handles:
- Layout element detection using IBM Docling Layout Predictor
- Table detection using TableFormer
- Creates visualization output (PNG)
- Creates coordinate JSON with structured data
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont
import os
import numpy as np

# Add the docling-ibm-models path to sys.path
sys.path.append(str(Path(__file__).parent.parent / "docling-ibm-models"))

try:
    from docling_ibm_models.layoutmodel.layout_predictor import LayoutPredictor
    LAYOUT_PREDICTOR_AVAILABLE = True
except ImportError:
    LAYOUT_PREDICTOR_AVAILABLE = False
    print("Warning: LayoutPredictor not available")

try:
    from docling_ibm_models.tableformer.data_management.tf_predictor import TFPredictor
    TABLE_PREDICTOR_AVAILABLE = True
except ImportError:
    TABLE_PREDICTOR_AVAILABLE = False
    print("Warning: TFPredictor not available")

class LayoutTableDetector:
    """Handles layout and table detection using IBM Docling models"""
    
    def __init__(self):
        self.stage_name = "layout_table_detection"
        self.layout_predictor = None
        self.table_predictor = None
        
        # Initialize predictors if available
        self._initialize_predictors()
    
    def _initialize_predictors(self):
        """Initialize the layout and table predictors using proper model downloads"""
        try:
            if LAYOUT_PREDICTOR_AVAILABLE and TABLE_PREDICTOR_AVAILABLE:
                # Download models from Hugging Face (same as batch_pipeline.py)
                print("📥 Downloading models from Hugging Face...")
                from huggingface_hub import snapshot_download
                
                # Download layout model
                self.layout_artifact_path = snapshot_download(repo_id="ds4sd/docling-layout-old")
                print("✅ Layout model downloaded")
                
                # Download tableformer model
                self.tableformer_artifact_path = snapshot_download(repo_id="ds4sd/docling-models", revision="v2.1.0")
                print("✅ TableFormer model downloaded")
                
                # Initialize layout predictor
                print("🔧 Initializing Layout Predictor...")
                self.layout_predictor = LayoutPredictor(
                    self.layout_artifact_path, 
                    device="cpu", 
                    num_threads=4
                )
                print("✅ Layout predictor initialized")
                
                # Initialize table predictor
                print("🔧 Initializing TableFormer...")
                self.table_predictor = self._init_tableformer()
                print("✅ Table predictor initialized")
                
            else:
                print("⚠️  Required modules not available, using fallback")
                
        except Exception as e:
            print(f"⚠️  Error initializing predictors: {e}")
            print("   Using fallback mode")
    
    def _init_tableformer(self):
        """Initialize TableFormer with correct configuration (from batch_pipeline.py)"""
        save_dir = os.path.join(self.tableformer_artifact_path, "model_artifacts/tableformer/fast")
        
        config = {
            "dataset": {
                "type": "TF_prepared",
                "name": "TF",
                "raw_data_dir": "./tests/test_data/model_artifacts/",
                "load_cells": True,
                "bbox_format": "5plet",
                "resized_image": 448,
                "keep_AR": False,
                "up_scaling_enabled": True,
                "down_scaling_enabled": True,
                "padding_mode": "null",
                "padding_color": [0, 0, 0],
                "image_normalization": {
                    "state": True,
                    "mean": [0.94247851, 0.94254675, 0.94292611],
                    "std": [0.17910956, 0.17940403, 0.17931663],
                },
                "color_jitter": True,
                "rand_crop": True,
                "rand_pad": True,
                "image_grayscale": False,
            },
            "model": {
                "type": "TableModel04_rs",
                "name": "14_128_256_4_true",
                "save_dir": save_dir,
                "backbone": "resnet18",
                "enc_image_size": 28,
                "tag_embed_dim": 16,
                "hidden_dim": 512,
                "tag_decoder_dim": 512,
                "bbox_embed_dim": 256,
                "tag_attention_dim": 256,
                "bbox_attention_dim": 512,
                "enc_layers": 4,  # 4 for fast model
                "dec_layers": 2,  # 2 for fast model
                "nheads": 8,
                "dropout": 0.1,
                "bbox_classes": 2,
            },
            "train": {
                "save_periodicity": 1,
                "disable_cuda": False,
                "epochs": 23,
                "batch_size": 50,
                "clip_gradient": 0.1,
                "clip_max_norm": 0.1,
                "bbox": True,
                "validation": False,
            },
            "predict": {
                "max_steps": 1024,
                "beam_size": 5,
                "bbox": True,
                "predict_dir": "./tests/test_data/samples",
                "pdf_cell_iou_thres": 0.05,
                "padding": False,
                "padding_size": 50,
                "disable_post_process": False,
                "profiling": False,
                "device_mode": "auto",
            },
            "dataset_wordmap": {
                "word_map_tag": {
                    "<pad>": 0, "<unk>": 1, "<start>": 2, "<end>": 3,
                    "ecel": 4, "fcel": 5, "lcel": 6, "ucel": 7, "xcel": 8,
                    "nl": 9, "ched": 10, "rhed": 11, "srow": 12,
                },
                "word_map_cell": {
                    " ": 13, "!": 179, '"': 126, "#": 101, "$": 119,
                    "%": 18, "&": 114, "'": 108, "(": 29, ")": 32,
                    "*": 26, "+": 97, ",": 71, "-": 63, ".": 34,
                    "/": 66, "0": 33, "1": 36, "2": 43, "3": 41,
                    "4": 45, "5": 17, "6": 37, "7": 35, "8": 40,
                    "9": 16, ":": 88, ";": 92, "<": 73, "=": 99,
                    ">": 39, "?": 96, "@": 125, "A": 27, "B": 86,
                    "C": 19, "D": 57, "E": 64, "F": 47, "G": 44,
                    "H": 10, "I": 20, "J": 80, "K": 81, "L": 52,
                    "M": 46, "N": 69, "O": 65, "P": 62, "Q": 59,
                    "R": 60, "S": 58, "T": 48, "U": 55, "V": 2,
                    "W": 83, "X": 104, "Y": 89, "Z": 113,
                    "a": 3, "b": 6, "c": 54, "d": 12, "e": 8,
                    "f": 50, "g": 28, "h": 56, "i": 5, "j": 82,
                    "k": 95, "l": 7, "m": 30, "n": 31, "o": 15,
                    "p": 22, "q": 67, "r": 4, "s": 51, "t": 14,
                    "u": 25, "v": 24, "w": 53, "x": 61, "y": 49,
                    "z": 11, "<pad>": 0, "<start>": 279, "<end>": 280,
                    "<unk>": 278
                }
            }
        }
        
        return TFPredictor(config)
    
    def process_image(self, input_file: str, output_prefix: str, intermediate_dir: Path) -> Dict:
        """Process a single image for layout and table detection"""
        print(f"🔍 Processing: {input_file}")
        
        try:
            # Run layout detection
            layout_result = self._run_layout_detection(input_file, output_prefix, intermediate_dir)
            
            # Run table detection
            table_result = self._run_table_detection(input_file, output_prefix, intermediate_dir)
            
            # Create visualization
            visualization_path = self._create_visualization(input_file, output_prefix, intermediate_dir, layout_result, table_result)
            
            # Create coordinates JSON
            coordinates_path = self._create_coordinates_json(output_prefix, intermediate_dir, layout_result, table_result)
            
            return {
                "success": True,
                "layout_count": len(layout_result.get("layout_elements", [])),
                "table_count": len(table_result.get("tables", [])),
                "visualization_path": str(visualization_path),
                "coordinates_path": str(coordinates_path)
            }
            
        except Exception as e:
            print(f"❌ Error in layout/table detection: {e}")
            return {"success": False, "error": str(e)}
    
    def _run_layout_detection(self, input_file: str, output_prefix: str, intermediate_dir: Path) -> Dict:
        """Run layout detection using IBM Docling Layout Predictor"""
        print("  📐 Running layout detection...")
        
        # Create separate layout output directory inside stage1
        stage1_dir = intermediate_dir / "stage1_layout_table"
        layout_dir = stage1_dir / "layout_outputs"
        layout_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = layout_dir / f"{output_prefix}_layout_predictions.json"
        
        try:
            if self.layout_predictor:
                # Load image
                image = Image.open(input_file)
                
                # Run layout prediction
                predictions = list(self.layout_predictor.predict(image))
                
                # Convert predictions to standard format
                layout_elements = []
                for pred in predictions:
                    layout_elements.append({
                        "label": pred.get("label", "unknown"),
                        "confidence": pred.get("confidence", 0.0),
                        "bbox": [pred.get("l", 0), pred.get("t", 0), pred.get("r", 0), pred.get("b", 0)]
                    })
                
                # Save results
                layout_data = {
                    "layout_elements": layout_elements,
                    "total_elements": len(layout_elements),
                    "image_path": input_file
                }
                
                with open(output_file, 'w') as f:
                    json.dump(layout_data, f, indent=2)
                
                print(f"    ✅ Layout detection completed: {len(layout_elements)} elements")
                return layout_data
                
            else:
                # Fallback: create empty layout data
                layout_data = {"layout_elements": [], "total_elements": 0}
                with open(output_file, 'w') as f:
                    json.dump(layout_data, f, indent=2)
                print(f"    ⚠️  Layout predictor not available, created empty layout data")
                return layout_data
                
        except Exception as e:
            print(f"    ❌ Layout detection failed: {e}")
            # Create fallback data
            layout_data = {"layout_elements": [], "total_elements": 0}
            with open(output_file, 'w') as f:
                json.dump(layout_data, f, indent=2)
            return layout_data
    
    def _run_table_detection(self, input_file: str, output_prefix: str, intermediate_dir: Path) -> Dict:
        """Run table detection using TableFormer"""
        print("  📋 Running table detection...")
        
        # Create separate tableformer output directory inside stage1
        stage1_dir = intermediate_dir / "stage1_layout_table"
        tableformer_dir = stage1_dir / "tableformer_outputs"
        tableformer_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = tableformer_dir / f"{output_prefix}_tableformer_results.json"
        
        try:
            if self.table_predictor:
                # First, run layout detection to find tables
                layout_predictions = list(self.layout_predictor.predict(Image.open(input_file)))
                
                # Extract table bounding boxes from layout predictions
                table_bboxes = []
                for pred in layout_predictions:
                    if pred.get("label", "").lower() == "table":
                        bbox = [pred["l"], pred["t"], pred["r"], pred["b"]]
                        table_bboxes.append(bbox)
                
                if not table_bboxes:
                    print("    ⚠️  No tables found in layout detection")
                    table_data = {"tables": [], "total_tables": 0}
                    with open(output_file, 'w') as f:
                        json.dump(table_data, f, indent=2)
                    return table_data
                
                print(f"    📋 Found {len(table_bboxes)} table(s) in layout detection")
                
                # Create mock OCR data structure for TableFormer (from batch_pipeline.py)
                image = Image.open(input_file)
                page_data = {
                    "width": image.width,
                    "height": image.height,
                    "tokens": [],
                    "blocks": [],
                    "cells": [],
                    "text_lines": [],
                    "fonts": [],
                    "links": [],
                    "rotation": 0.0,
                    "rectangles": [],
                    "textPositions": [],
                    "localized_image_locations": [],
                    "scanned_elements": [],
                    "paths": [],
                    "pageNumber": 1,
                    "page_image": {},
                    "lang": ["en"]
                }
                
                # Load image for TableFormer
                import cv2
                img_array = cv2.imread(input_file)
                page_data["image"] = img_array
                
                # Run TableFormer analysis
                multi_tf_output = self.table_predictor.multi_table_predict(page_data, table_bboxes, True)
                
                # Save results in the same format as batch_pipeline.py
                enhanced_results = {
                    'tableformer_results': multi_tf_output,
                    'processing_metadata': {
                        'total_tables': len(table_bboxes),
                        'processing_timestamp': self._get_timestamp()
                    }
                }
                
                with open(output_file, 'w') as f:
                    json.dump(enhanced_results, f, indent=2)
                
                print(f"    ✅ TableFormer analysis completed: {len(table_bboxes)} tables")
                return enhanced_results
                
            else:
                # Fallback: create empty table data
                table_data = {"tables": [], "total_tables": 0}
                with open(output_file, 'w') as f:
                    json.dump(table_data, f, indent=2)
                print(f"    ⚠️  Table predictor not available, created empty table data")
                return table_data
                
        except Exception as e:
            print(f"    ❌ Table detection failed: {e}")
            # Create fallback data
            table_data = {"tables": [], "total_tables": 0}
            with open(output_file, 'w') as f:
                json.dump(table_data, f, indent=2)
            return table_data
    
    def _create_visualization(self, input_file: str, output_prefix: str, intermediate_dir: Path, 
                            layout_result: Dict, table_result: Dict) -> Path:
        """Create visualization of layout and table detection results"""
        print("  🎨 Creating layout/table visualization...")
        
        # Create separate visualization directories inside stage1
        stage1_dir = intermediate_dir / "stage1_layout_table"
        layout_viz_dir = stage1_dir / "layout_outputs"
        tableformer_viz_dir = stage1_dir / "tableformer_outputs"
        layout_viz_dir.mkdir(parents=True, exist_ok=True)
        tableformer_viz_dir.mkdir(parents=True, exist_ok=True)
        
        # Create layout visualization
        layout_viz_path = layout_viz_dir / f"{output_prefix}_layout_annotated.png"
        tableformer_viz_path = tableformer_viz_dir / f"{output_prefix}_tableformer_annotated.png"
        
        try:
            # Load original image
            original_img = Image.open(input_file)
            img_width, img_height = original_img.size
            
            # Try to load font
            try:
                font = ImageFont.truetype("arial.ttf", 12)
                small_font = ImageFont.truetype("arial.ttf", 10)
            except:
                font = ImageFont.load_default()
                small_font = ImageFont.load_default()
            
            # Create layout visualization (layout elements only)
            layout_canvas = Image.new('RGB', (img_width, img_height), 'white')
            layout_canvas.paste(original_img)
            layout_draw = ImageDraw.Draw(layout_canvas)
            
            # Draw only layout elements
            layout_elements = layout_result.get("layout_elements", [])
            for i, element in enumerate(layout_elements):
                self._draw_layout_element(layout_draw, element, i, font)
            
            # Create tableformer visualization (table cells only)
            tableformer_canvas = Image.new('RGB', (img_width, img_height), 'white')
            tableformer_canvas.paste(original_img)
            tableformer_draw = ImageDraw.Draw(tableformer_canvas)
            
            # Draw only table cells from TableFormer results
            tableformer_results = table_result.get("tableformer_results", [])
            for i, tf_result in enumerate(tableformer_results):
                self._draw_tableformer_result(tableformer_draw, tf_result, i, small_font)
            
            # Save visualizations
            layout_canvas.save(layout_viz_path)
            tableformer_canvas.save(tableformer_viz_path)
            print(f"    ✅ Layout visualization saved: {layout_viz_path}")
            print(f"    ✅ TableFormer visualization saved: {tableformer_viz_path}")
            
            return layout_viz_path
            
        except Exception as e:
            print(f"    ❌ Error creating visualization: {e}")
            # Create a simple placeholder image
            placeholder = Image.new('RGB', (800, 600), 'lightgray')
            placeholder.save(layout_viz_path)
            placeholder.save(tableformer_viz_path)
            return layout_viz_path
    
    def _draw_layout_element(self, draw: ImageDraw.Draw, element: Dict, index: int, font):
        """Draw a single layout element"""
        # Get bounding box coordinates
        if 'bbox' in element:
            x1, y1, x2, y2 = element['bbox']
        elif all(key in element for key in ['l', 't', 'r', 'b']):
            x1, y1, x2, y2 = element['l'], element['t'], element['r'], element['b']
        else:
            return
        
        # Convert to integers
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        label = element.get('label', 'unknown')
        
        # Choose color based on layout type
        if label.lower() == 'section-header':
            color = (0, 255, 0)  # Green
        elif label.lower() == 'table':
            color = (255, 0, 0)  # Red
        elif label.lower() == 'key-value':
            color = (255, 0, 255)  # Magenta
        elif label.lower() == 'list-item':
            color = (0, 0, 255)  # Blue
        else:
            color = (128, 128, 128)  # Gray
        
        # Draw border
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        
        # Draw label
        label_text = f"{label} {index+1}"
        draw.text((x1+2, y1+2), label_text, fill=color, font=font)
    
    def _draw_tableformer_result(self, draw: ImageDraw.Draw, tf_result: Dict, index: int, font):
        """Draw TableFormer result (table cells)"""
        predict_details = tf_result.get("predict_details", {})
        table_bbox = predict_details.get("table_bbox", [0, 0, 100, 100])
        cell_bboxes = predict_details.get("prediction_bboxes_page", [])
        
        # Draw table boundary
        if len(table_bbox) >= 4:
            x1, y1, x2, y2 = map(int, table_bbox[:4])
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            draw.text((x1, y1-20), f"Table {index+1}", fill="red")
        
        # Draw individual cells
        for j, cell_bbox in enumerate(cell_bboxes):
            if len(cell_bbox) >= 4:
                cx1, cy1, cx2, cy2 = map(int, cell_bbox[:4])
                # Draw cell boundary in green
                draw.rectangle([cx1, cy1, cx2, cy2], outline="green", width=2)
                # Add cell number
                draw.text((cx1+2, cy1+2), str(j+1), fill="green")
    
    def _draw_table(self, draw: ImageDraw.Draw, table: Dict, index: int, font):
        """Draw a single table"""
        # Get table bounding box
        if 'bbox' in table:
            x1, y1, x2, y2 = table['bbox']
        elif 'table_bbox' in table:
            x1, y1, x2, y2 = table['table_bbox']
        else:
            return
        
        # Convert to integers
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Draw table border
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
        
        # Draw table label
        label_text = f"Table {index+1}"
        draw.text((x1+5, y1+5), label_text, fill=(255, 0, 0), font=font)
        
        # Draw cells if available
        cells = table.get('cells', [])
        for cell in cells:
            if 'bbox' in cell:
                bbox = cell['bbox']
                if isinstance(bbox, dict):
                    cx1, cy1, cx2, cy2 = bbox.get('l', 0), bbox.get('t', 0), bbox.get('r', 0), bbox.get('b', 0)
                else:
                    cx1, cy1, cx2, cy2 = bbox
                cx1, cy1, cx2, cy2 = int(cx1), int(cy1), int(cx2), int(cy2)
                draw.rectangle([cx1, cy1, cx2, cy2], outline=(200, 0, 0), width=1)
    
    def _create_coordinates_json(self, output_prefix: str, intermediate_dir: Path, 
                               layout_result: Dict, table_result: Dict) -> Path:
        """Create structured coordinate JSON"""
        print("  📄 Creating coordinates JSON...")
        
        # Create separate coordinate directories inside stage1
        stage1_dir = intermediate_dir / "stage1_layout_table"
        layout_coords_dir = stage1_dir / "layout_outputs"
        tableformer_coords_dir = stage1_dir / "tableformer_outputs"
        layout_coords_dir.mkdir(parents=True, exist_ok=True)
        tableformer_coords_dir.mkdir(parents=True, exist_ok=True)
        
        # Create layout coordinates
        layout_coords_path = layout_coords_dir / f"{output_prefix}_layout_coordinates.json"
        tableformer_coords_path = tableformer_coords_dir / f"{output_prefix}_tableformer_coordinates.json"
        
        # Create structured data
        coordinates_data = {
            "stage": 1,
            "stage_name": "layout_table_detection",
            "processing_timestamp": self._get_timestamp(),
            "layout_elements": layout_result.get("layout_elements", []),
            "tables": table_result.get("tables", []),
            "summary": {
                "total_layout_elements": len(layout_result.get("layout_elements", [])),
                "total_tables": len(table_result.get("tables", [])),
                "layout_element_types": self._count_element_types(layout_result.get("layout_elements", []))
            }
        }
        
        # Save layout coordinates
        layout_coords_data = {
            "stage": 1,
            "stage_name": "layout_detection",
            "processing_timestamp": self._get_timestamp(),
            "layout_elements": layout_result.get("layout_elements", []),
            "summary": {
                "total_layout_elements": len(layout_result.get("layout_elements", [])),
                "layout_element_types": self._count_element_types(layout_result.get("layout_elements", []))
            }
        }
        
        with open(layout_coords_path, 'w') as f:
            json.dump(layout_coords_data, f, indent=2)
        
        # Save tableformer coordinates
        tableformer_coords_data = {
            "stage": 1,
            "stage_name": "table_detection",
            "processing_timestamp": self._get_timestamp(),
            "tableformer_results": table_result.get("tableformer_results", []),
            "summary": {
                "total_tables": table_result.get("processing_metadata", {}).get("total_tables", 0)
            }
        }
        
        with open(tableformer_coords_path, 'w') as f:
            json.dump(tableformer_coords_data, f, indent=2)
        
        print(f"    ✅ Layout coordinates JSON saved: {layout_coords_path}")
        print(f"    ✅ TableFormer coordinates JSON saved: {tableformer_coords_path}")
        return layout_coords_path
    
    def _count_element_types(self, layout_elements: List[Dict]) -> Dict[str, int]:
        """Count different types of layout elements"""
        type_counts = {}
        for element in layout_elements:
            label = element.get('label', 'unknown')
            type_counts[label] = type_counts.get(label, 0) + 1
        return type_counts
    
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()

def main():
    """Main function for testing Stage 1"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Stage 1: Layout & Table Detection")
    parser.add_argument("--input", "-i", required=True, help="Input image file")
    parser.add_argument("--output-prefix", "-o", help="Output prefix (default: input filename)")
    parser.add_argument("--intermediate-dir", default="intermediate_outputs", help="Intermediate outputs directory")
    
    args = parser.parse_args()
    
    # Set default output prefix
    if not args.output_prefix:
        args.output_prefix = Path(args.input).stem
    
    # Initialize detector
    detector = LayoutTableDetector()
    
    # Process image
    result = detector.process_image(args.input, args.output_prefix, Path(args.intermediate_dir))
    
    # Print results
    if result.get("success", False):
        print(f"\n✅ Stage 1 completed successfully!")
        print(f"   Layout elements: {result.get('layout_count', 0)}")
        print(f"   Tables: {result.get('table_count', 0)}")
        print(f"   Visualization: {result.get('visualization_path', 'N/A')}")
        print(f"   Coordinates: {result.get('coordinates_path', 'N/A')}")
    else:
        print(f"\n❌ Stage 1 failed: {result.get('error', 'Unknown error')}")
        sys.exit(1)

if __name__ == "__main__":
    main()