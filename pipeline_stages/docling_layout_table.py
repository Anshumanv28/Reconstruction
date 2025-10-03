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

try:
    from transformers import TableTransformerForObjectDetection, DetrFeatureExtractor
    import torch
    TABLE_TRANSFORMER_AVAILABLE = True
except ImportError:
    TABLE_TRANSFORMER_AVAILABLE = False
    print("Warning: TableTransformer not available")

class LayoutTableDetector:
    """Handles layout and table detection using IBM Docling models"""
    
    def __init__(self):
        self.stage_name = "layout_table_detection"
        self.layout_predictor = None
        self.table_predictor = None
        self.table_transformer_model = None
        self.feature_extractor = None
        
        # Initialize predictors if available
        self._initialize_predictors()
    
    def _initialize_predictors(self):
        """Initialize the layout and table predictors using proper model downloads"""
        try:
            if LAYOUT_PREDICTOR_AVAILABLE and TABLE_PREDICTOR_AVAILABLE:
                # Download models from Hugging Face (same as batch_pipeline.py)
                print("DOWNLOADING Downloading models from Hugging Face...")
                from huggingface_hub import snapshot_download
                
                # Download layout model
                self.layout_artifact_path = snapshot_download(repo_id="ds4sd/docling-layout-old")
                print("SUCCESS Layout model downloaded")
                
                # Download tableformer model
                self.tableformer_artifact_path = snapshot_download(repo_id="ds4sd/docling-models", revision="v2.1.0")
                print("SUCCESS TableFormer model downloaded")
                
                # Initialize layout predictor
                print("INITIALIZING Initializing Layout Predictor...")
                self.layout_predictor = LayoutPredictor(
                    self.layout_artifact_path, 
                    device="cpu", 
                    num_threads=4
                )
                print("SUCCESS Layout predictor initialized")
                
                # Initialize table predictor
                print("INITIALIZING Initializing TableFormer...")
                self.table_predictor = self._init_tableformer()
                print("SUCCESS Table predictor initialized")
                
                # Initialize Table Transformer for structure recognition
                if TABLE_TRANSFORMER_AVAILABLE:
                    print("INITIALIZING Initializing Table Transformer...")
                    self.table_transformer_model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-structure-recognition")
                    self.feature_extractor = DetrFeatureExtractor()
                    print("SUCCESS Table Transformer initialized")
                
            else:
                print("WARNING  Required modules not available, using fallback")
                
        except Exception as e:
            print(f"WARNING  Error initializing predictors: {e}")
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
        print(f"PROCESSING Processing: {input_file}")
        
        try:
            # Run layout detection
            layout_result = self._run_layout_detection(input_file, output_prefix, intermediate_dir)
            
            # Run table detection using layout results
            table_result = self._run_table_detection(input_file, output_prefix, intermediate_dir, layout_result)
            
            # Run Table Transformer structure recognition
            table_transformer_result = self._run_table_transformer_structure_recognition(input_file, output_prefix, intermediate_dir, table_result)
            
            # Create visualization
            visualization_path = self._create_visualization(input_file, output_prefix, intermediate_dir, layout_result, table_result, table_transformer_result)
            
            # Create coordinates JSON
            coordinates_path = self._create_coordinates_json(output_prefix, intermediate_dir, layout_result, table_result)
            
            # Create coordinate reference visualization
            coordinate_ref_path = self._create_coordinate_reference_visualization(input_file, output_prefix, intermediate_dir)
            
            return {
                "success": True,
                "layout_count": len(layout_result.get("layout_elements", [])),
                "table_count": len(table_result.get("tables", [])),
                "visualization_path": str(visualization_path),
                "coordinates_path": str(coordinates_path),
                "coordinate_reference_path": str(coordinate_ref_path)
            }
            
        except Exception as e:
            print(f"ERROR Error in layout/table detection: {e}")
            return {"success": False, "error": str(e)}
    
    def _run_layout_detection(self, input_file: str, output_prefix: str, intermediate_dir: Path) -> Dict:
        """Run layout detection using IBM Docling Layout Predictor"""
        print("  LAYOUT Running layout detection...")
        
        # Create separate layout output directory inside stage1
        stage1_dir = intermediate_dir / "stage1_layout_table"
        layout_dir = stage1_dir / "layout_outputs"
        layout_dir.mkdir(parents=True, exist_ok=True)
        
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
                
                # Create layout data (no longer saving predictions file)
                layout_data = {
                    "layout_elements": layout_elements,
                    "total_elements": len(layout_elements),
                    "image_path": input_file
                }
                
                print(f"    SUCCESS Layout detection completed: {len(layout_elements)} elements")
                return layout_data
                
            else:
                # Fallback: create empty layout data
                layout_data = {"layout_elements": [], "total_elements": 0}
                print(f"    WARNING  Layout predictor not available, created empty layout data")
                return layout_data
                
        except Exception as e:
            print(f"    ERROR Layout detection failed: {e}")
            # Create fallback data
            layout_data = {"layout_elements": [], "total_elements": 0}
            return layout_data
    
    def _run_table_detection(self, input_file: str, output_prefix: str, intermediate_dir: Path, layout_result: Dict) -> Dict:
        """Run unified table detection using normalized coordinates from layout detection"""
        print("  TABLES Running unified table detection...")
        
        # Create separate tableformer output directory inside stage1
        stage1_dir = intermediate_dir / "stage1_layout_table"
        tableformer_dir = stage1_dir / "tableformer_outputs"
        tableformer_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Extract table bounding boxes from layout detection results
            layout_elements = layout_result.get("layout_elements", [])
            table_bboxes = []
            
            for element in layout_elements:
                if element.get("label", "").lower() == "table":
                    bbox = element.get("bbox", [])
                    if len(bbox) >= 4:
                        table_bboxes.append(bbox)
            
            if not table_bboxes:
                print("    WARNING  No tables found in layout detection")
                return {"tables": [], "total_tables": 0, "normalized_coordinates": []}
            
            print(f"    TABLES Found {len(table_bboxes)} table(s) from layout detection")
            
            # Load original image for cropping
            image = Image.open(input_file)
            
            # Normalize coordinates: create padded coordinates for all models
            normalized_tables = []
            padding = 20  # 20 pixels padding around the table
            
            for table_idx, table_bbox in enumerate(table_bboxes):
                print(f"      Normalizing table {table_idx + 1} coordinates...")
                
                # Extract original coordinates
                x1, y1, x2, y2 = map(int, table_bbox[:4])
                
                # Apply padding with image boundary checks
                padded_x1 = max(0, x1 - padding)
                padded_y1 = max(0, y1 - padding)
                padded_x2 = min(image.width, x2 + padding)
                padded_y2 = min(image.height, y2 + padding)
                
                print(f"        Original bbox: [{x1}, {y1}, {x2}, {y2}]")
                print(f"        Padded bbox: [{padded_x1}, {padded_y1}, {padded_x2}, {padded_y2}]")
                
                # Create normalized table data structure
                normalized_table = {
                    "table_id": f"table_{table_idx + 1}",
                    "original_bbox": [x1, y1, x2, y2],
                    "padded_bbox": [padded_x1, padded_y1, padded_x2, padded_y2],
                    "crop_size": [padded_x2 - padded_x1, padded_y2 - padded_y1],
                    "layout_confidence": next((elem.get("confidence", 0) for elem in layout_elements 
                                             if elem.get("bbox") == table_bbox), 0),
                    "tableformer_results": None,
                    "table_transformer_results": None
                }
                
                normalized_tables.append(normalized_table)
            
            # Process with TableFormer using normalized coordinates
            if self.table_predictor:
                print("    PROCESSING Running TableFormer on normalized coordinates...")
                for table_idx, normalized_table in enumerate(normalized_tables):
                    print(f"      Processing table {table_idx + 1} with TableFormer...")
                    
                    padded_bbox = normalized_table["padded_bbox"]
                    padded_x1, padded_y1, padded_x2, padded_y2 = padded_bbox
                    
                    # Crop padded table region from original image
                    table_crop = image.crop((padded_x1, padded_y1, padded_x2, padded_y2))
                    
                    # Create mock OCR data structure for cropped table region
                    crop_width, crop_height = table_crop.size
                    page_data = {
                        "width": crop_width,
                        "height": crop_height,
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
                    
                    # Load cropped image for TableFormer
                    import cv2
                    import numpy as np
                    crop_array = np.array(table_crop)
                    crop_array = cv2.cvtColor(crop_array, cv2.COLOR_RGB2BGR)
                    page_data["image"] = crop_array
                    
                    # Create bbox for cropped region (full crop)
                    crop_bbox = [0, 0, crop_width, crop_height]
                    
                    # Run TableFormer analysis on cropped table
                    tf_output = self.table_predictor.multi_table_predict(page_data, [crop_bbox], True)
                    
                    # Adjust coordinates back to original image space
                    if tf_output:
                        original_bbox = normalized_table["original_bbox"]
                        adjusted_output = self._adjust_tableformer_coordinates(tf_output[0], original_bbox, padded_bbox)
                        normalized_table["tableformer_results"] = adjusted_output
                    else:
                        normalized_table["tableformer_results"] = {}
            
            # Enhanced processing: Extract detailed table structure
            enhanced_tables = self._extract_detailed_table_structure(
                [table["tableformer_results"] for table in normalized_tables if table["tableformer_results"]], 
                [table["original_bbox"] for table in normalized_tables], 
                image
            )
            
            # Create unified results
            unified_results = {
                'normalized_tables': normalized_tables,
                'enhanced_table_structure': enhanced_tables,
                'processing_metadata': {
                    'total_tables': len(table_bboxes),
                    'total_cells': sum(len(table.get('cells', [])) for table in enhanced_tables),
                    'total_rows': sum(len(table.get('rows', [])) for table in enhanced_tables),
                    'total_columns': sum(len(table.get('columns', [])) for table in enhanced_tables),
                    'processing_timestamp': self._get_timestamp(),
                    'coordinate_normalization': 'unified_padded_coordinates'
                }
            }
            
            print(f"    SUCCESS Unified table detection completed:")
            print(f"       Tables: {len(table_bboxes)}")
            print(f"       Total cells: {unified_results['processing_metadata']['total_cells']}")
            print(f"       Total rows: {unified_results['processing_metadata']['total_rows']}")
            print(f"       Total columns: {unified_results['processing_metadata']['total_columns']}")
            return unified_results
            
        except Exception as e:
            print(f"    ERROR Unified table detection failed: {e}")
            # Create fallback data
            return {"tables": [], "total_tables": 0, "normalized_coordinates": []}
    
    def _run_table_transformer_structure_recognition(self, input_file: str, output_prefix: str, intermediate_dir: Path, table_result: Dict) -> Dict:
        """Run Table Transformer structure recognition using normalized coordinates"""
        print("  TABLES Running Table Transformer structure recognition on normalized coordinates...")
        
        # Create separate table transformer output directory inside stage1
        stage1_dir = intermediate_dir / "stage1_layout_table"
        table_transformer_dir = stage1_dir / "table_transformer_outputs"
        table_transformer_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            if self.table_transformer_model and self.feature_extractor:
                # Load original image
                original_image = Image.open(input_file).convert("RGB")
                
                # Get normalized table data
                normalized_tables = table_result.get("normalized_tables", [])
                all_table_structures = []
                
                print(f"    PROCESSING Analyzing {len(normalized_tables)} normalized table regions...")
                
                for table_idx, normalized_table in enumerate(normalized_tables):
                    print(f"      Processing table {table_idx + 1} with Table Transformer...")
                    
                    # Use the same padded coordinates as TableFormer
                    padded_bbox = normalized_table["padded_bbox"]
                    padded_x1, padded_y1, padded_x2, padded_y2 = padded_bbox
                    
                    print(f"        Using normalized padded bbox: [{padded_x1}, {padded_y1}, {padded_x2}, {padded_y2}]")
                    
                    # Crop padded table region from original image
                    table_crop = original_image.crop((padded_x1, padded_y1, padded_x2, padded_y2))
                    
                    # Prepare cropped table image for Table Transformer
                    encoding = self.feature_extractor(table_crop, return_tensors="pt")
                    
                    # Run Table Transformer structure recognition on cropped table
                    with torch.no_grad():
                        outputs = self.table_transformer_model(**encoding)
                    
                    # Post-process results for cropped table
                    target_sizes = [table_crop.size[::-1]]
                    results = self.feature_extractor.post_process_object_detection(
                        outputs, threshold=0.6, target_sizes=target_sizes
                    )[0]
                    
                    # Extract detailed structure for this table
                    original_bbox = normalized_table["original_bbox"]
                    table_structure = self._extract_table_transformer_structure_for_table(
                        results, table_crop, table_idx, original_bbox, padded_bbox
                    )
                    
                    # Store Table Transformer results in normalized table
                    normalized_table["table_transformer_results"] = table_structure
                    all_table_structures.append(table_structure)
                    
                    print(f"        Extracted: {table_structure.get('total_rows', 0)} rows, {table_structure.get('total_columns', 0)} columns")
                    print(f"        Headers: {table_structure.get('total_column_headers', 0)} column headers, {table_structure.get('total_row_headers', 0)} row headers")
                    print(f"        Cells: {table_structure.get('total_cells', 0)} spanning cells")
                
                # Combine all table structures
                combined_structure = self._combine_table_structures(all_table_structures)
                
                # Create unified results with normalized tables
                unified_results = {
                    'normalized_tables': normalized_tables,
                    'table_transformer_results': {
                        'individual_tables': all_table_structures,
                        'combined_structure': combined_structure
                    },
                    'enhanced_table_structure': combined_structure,
                    'processing_metadata': {
                        'total_tables_processed': len(normalized_tables),
                        'model_config': self.table_transformer_model.config.id2label,
                        'processing_timestamp': self._get_timestamp(),
                        'coordinate_normalization': 'unified_padded_coordinates'
                    }
                }
                
                print(f"    SUCCESS Table Transformer structure recognition completed:")
                print(f"       Tables processed: {len(normalized_tables)}")
                print(f"       Total rows: {combined_structure.get('total_rows', 0)}")
                print(f"       Total columns: {combined_structure.get('total_columns', 0)}")
                print(f"       Total cells: {combined_structure.get('total_cells', 0)}")
                return unified_results
                
            else:
                # Fallback: create empty data
                print(f"    WARNING  Table Transformer not available, created empty data")
                return {"enhanced_table_structure": {}, "processing_metadata": {}, "normalized_tables": []}
                
        except Exception as e:
            print(f"    ERROR Table Transformer structure recognition failed: {e}")
            # Create fallback data
            return {"enhanced_table_structure": {}, "processing_metadata": {}, "normalized_tables": []}
    
    def _extract_table_transformer_structure(self, results: Dict, image) -> Dict:
        """Extract detailed table structure from Table Transformer results"""
        print("    PROCESSING Extracting Table Transformer structure...")
        
        # Table Transformer label mapping
        id2label = {
            0: 'table',
            1: 'table column', 
            2: 'table row',
            3: 'table column header',
            4: 'table projected row header',
            5: 'table spanning cell'
        }
        
        # Group elements by type
        elements_by_type = {}
        for score, label, box in zip(results['scores'], results['labels'], results['boxes']):
            element_type = id2label.get(label.item(), 'unknown')
            if element_type not in elements_by_type:
                elements_by_type[element_type] = []
            
            elements_by_type[element_type].append({
                'score': score.item(),
                'label': element_type,
                'bbox': box.tolist()
            })
        
        # Extract table structure
        tables = elements_by_type.get('table', [])
        table_columns = elements_by_type.get('table column', [])
        table_rows = elements_by_type.get('table row', [])
        column_headers = elements_by_type.get('table column header', [])
        row_headers = elements_by_type.get('table projected row header', [])
        spanning_cells = elements_by_type.get('table spanning cell', [])
        
        # Create enhanced structure
        enhanced_structure = {
            'tables': tables,
            'table_columns': table_columns,
            'table_rows': table_rows,
            'column_headers': column_headers,
            'row_headers': row_headers,
            'spanning_cells': spanning_cells,
            'total_tables': len(tables),
            'total_columns': len(table_columns),
            'total_rows': len(table_rows),
            'total_column_headers': len(column_headers),
            'total_row_headers': len(row_headers),
            'total_cells': len(spanning_cells),
            'element_counts': {k: len(v) for k, v in elements_by_type.items()}
        }
        
        print(f"        Extracted: {len(tables)} tables, {len(table_rows)} rows, {len(table_columns)} columns")
        print(f"        Headers: {len(column_headers)} column headers, {len(row_headers)} row headers")
        print(f"        Cells: {len(spanning_cells)} spanning cells")
        
        return enhanced_structure
    
    def _extract_table_transformer_structure_for_table(self, results: Dict, table_crop, table_idx: int, original_bbox: List, padded_bbox: List = None) -> Dict:
        """Extract detailed table structure from Table Transformer results for a single cropped table"""
        
        # Table Transformer label mapping
        id2label = {
            0: 'table',
            1: 'table column', 
            2: 'table row',
            3: 'table column header',
            4: 'table projected row header',
            5: 'table spanning cell'
        }
        
        # Group elements by type
        elements_by_type = {}
        for score, label, box in zip(results['scores'], results['labels'], results['boxes']):
            element_type = id2label.get(label.item(), 'unknown')
            if element_type not in elements_by_type:
                elements_by_type[element_type] = []
            
            # Convert coordinates back to original image coordinates
            x1, y1, x2, y2 = box.tolist()
            
            # Use padded bbox for coordinate adjustment if available
            if padded_bbox:
                pad_x1, pad_y1, pad_x2, pad_y2 = padded_bbox[:4]
            else:
                orig_x1, orig_y1, orig_x2, orig_y2 = original_bbox[:4]
                pad_x1, pad_y1, pad_x2, pad_y2 = orig_x1, orig_y1, orig_x2, orig_y2
            
            # Adjust coordinates to original image space using padded bbox
            adjusted_bbox = [
                x1 + pad_x1,
                y1 + pad_y1, 
                x2 + pad_x1,
                y2 + pad_y1
            ]
            
            elements_by_type[element_type].append({
                'score': score.item(),
                'label': element_type,
                'bbox': adjusted_bbox,
                'crop_bbox': box.tolist()
            })
        
        # Extract table structure
        tables = elements_by_type.get('table', [])
        table_columns = elements_by_type.get('table column', [])
        table_rows = elements_by_type.get('table row', [])
        column_headers = elements_by_type.get('table column header', [])
        row_headers = elements_by_type.get('table projected row header', [])
        spanning_cells = elements_by_type.get('table spanning cell', [])
        
        # Create enhanced structure for this table
        table_structure = {
            'table_index': table_idx,
            'original_bbox': original_bbox,
            'crop_size': table_crop.size,
            'tables': tables,
            'table_columns': table_columns,
            'table_rows': table_rows,
            'column_headers': column_headers,
            'row_headers': row_headers,
            'spanning_cells': spanning_cells,
            'total_tables': len(tables),
            'total_columns': len(table_columns),
            'total_rows': len(table_rows),
            'total_column_headers': len(column_headers),
            'total_row_headers': len(row_headers),
            'total_cells': len(spanning_cells),
            'element_counts': {k: len(v) for k, v in elements_by_type.items()}
        }
        
        return table_structure
    
    def _adjust_tableformer_coordinates(self, tf_output: Dict, original_bbox: List, padded_bbox: List = None) -> Dict:
        """Adjust TableFormer coordinates from cropped space back to original image space"""
        
        # Extract original table bounding box
        orig_x1, orig_y1, orig_x2, orig_y2 = original_bbox[:4]
        
        # If padded_bbox is provided, use it for coordinate adjustment
        if padded_bbox:
            pad_x1, pad_y1, pad_x2, pad_y2 = padded_bbox[:4]
        else:
            pad_x1, pad_y1, pad_x2, pad_y2 = orig_x1, orig_y1, orig_x2, orig_y2
        
        # Adjust table_bbox in predict_details
        if 'predict_details' in tf_output:
            predict_details = tf_output['predict_details'].copy()
            
            # Adjust table_bbox
            if 'table_bbox' in predict_details:
                table_bbox = predict_details['table_bbox']
                if len(table_bbox) >= 4:
                    # Convert from crop coordinates to original coordinates using padded bbox
                    adjusted_table_bbox = [
                        table_bbox[0] + pad_x1,
                        table_bbox[1] + pad_y1,
                        table_bbox[2] + pad_x1,
                        table_bbox[3] + pad_y1
                    ]
                    predict_details['table_bbox'] = adjusted_table_bbox
            
            # Adjust prediction_bboxes_page (cell coordinates)
            if 'prediction_bboxes_page' in predict_details:
                adjusted_bboxes = []
                for bbox in predict_details['prediction_bboxes_page']:
                    if len(bbox) >= 4:
                        adjusted_bbox = [
                            bbox[0] + pad_x1,
                            bbox[1] + pad_y1,
                            bbox[2] + pad_x1,
                            bbox[3] + pad_y1
                        ]
                        adjusted_bboxes.append(adjusted_bbox)
                    else:
                        adjusted_bboxes.append(bbox)
                predict_details['prediction_bboxes_page'] = adjusted_bboxes
            
            tf_output['predict_details'] = predict_details
        
        return tf_output
    
    def _combine_table_structures(self, all_table_structures: List[Dict]) -> Dict:
        """Combine structures from all individual tables into a single structure"""
        
        combined_tables = []
        combined_columns = []
        combined_rows = []
        combined_column_headers = []
        combined_row_headers = []
        combined_spanning_cells = []
        
        for table_structure in all_table_structures:
            combined_tables.extend(table_structure.get('tables', []))
            combined_columns.extend(table_structure.get('table_columns', []))
            combined_rows.extend(table_structure.get('table_rows', []))
            combined_column_headers.extend(table_structure.get('column_headers', []))
            combined_row_headers.extend(table_structure.get('row_headers', []))
            combined_spanning_cells.extend(table_structure.get('spanning_cells', []))
        
        # Combine element counts
        combined_element_counts = {}
        for table_structure in all_table_structures:
            element_counts = table_structure.get('element_counts', {})
            for element_type, count in element_counts.items():
                combined_element_counts[element_type] = combined_element_counts.get(element_type, 0) + count
        
        return {
            'tables': combined_tables,
            'table_columns': combined_columns,
            'table_rows': combined_rows,
            'column_headers': combined_column_headers,
            'row_headers': combined_row_headers,
            'spanning_cells': combined_spanning_cells,
            'total_tables': len(combined_tables),
            'total_columns': len(combined_columns),
            'total_rows': len(combined_rows),
            'total_column_headers': len(combined_column_headers),
            'total_row_headers': len(combined_row_headers),
            'total_cells': len(combined_spanning_cells),
            'element_counts': combined_element_counts,
            'individual_tables': all_table_structures
        }
    
    def _extract_detailed_table_structure(self, multi_tf_output: List, table_bboxes: List, image) -> List[Dict]:
        """Extract detailed table structure including rows, columns, cells, and headers like Table Transformer"""
        print("    PROCESSING Extracting detailed table structure...")
        
        enhanced_tables = []
        
        for table_idx, (tf_output, table_bbox) in enumerate(zip(multi_tf_output, table_bboxes)):
            print(f"      Processing table {table_idx + 1}...")
            
            # Extract table structure from TableFormer output
            table_structure = self._parse_tableformer_output(tf_output, table_bbox, table_idx)
            enhanced_tables.append(table_structure)
        
        return enhanced_tables
    
    def _parse_tableformer_output(self, tf_output: Dict, table_bbox: List, table_idx: int) -> Dict:
        """Parse TableFormer output to extract detailed table structure"""
        
        # Initialize table structure
        table_structure = {
            "table_id": f"table_{table_idx}",
            "table_bbox": table_bbox,
            "rows": [],
            "columns": [],
            "cells": [],
            "headers": {
                "column_headers": [],
                "row_headers": []
            },
            "grid_structure": {
                "num_rows": 0,
                "num_columns": 0,
                "cell_matrix": []
            }
        }
        
        try:
            # Extract cells from TableFormer output
            cells = []
            
            # Check for different possible cell data structures
            if 'cells' in tf_output:
                cells = tf_output['cells']
                print(f"        Found {len(cells)} cells in 'cells' field")
            elif 'prediction_bboxes_page' in tf_output.get('predict_details', {}):
                # TableFormer format: extract from prediction_bboxes_page
                prediction_bboxes = tf_output['predict_details']['prediction_bboxes_page']
                print(f"        Found {len(prediction_bboxes)} cell bboxes in TableFormer output")
                
                # Convert bboxes to cell format
                for cell_idx, bbox in enumerate(prediction_bboxes):
                    cell_data = {
                        'bbox': bbox,
                        'text': '',  # TableFormer doesn't provide text directly
                        'row': 0,    # Will be determined by position
                        'col': 0,    # Will be determined by position
                        'rowspan': 1,
                        'colspan': 1,
                        'is_header': False,
                        'confidence': 1.0,
                        'cell_type': 'data_cell'
                    }
                    cells.append(cell_data)
            elif 'tf_responses' in tf_output and tf_output['tf_responses']:
                # Alternative format
                tf_responses = tf_output['tf_responses']
                print(f"        Found {len(tf_responses)} responses in tf_responses")
                # Process tf_responses if needed
            else:
                print(f"        No recognizable cell data found in TableFormer output")
                print(f"        Available keys: {list(tf_output.keys())}")
            
            if cells:
                # Process each cell
                processed_cells = []
                for cell_idx, cell in enumerate(cells):
                    cell_data = self._process_tableformer_cell(cell, cell_idx)
                    processed_cells.append(cell_data)
                
                # Organize cells into grid structure based on position
                organized_cells = self._organize_cells_into_grid(processed_cells, table_bbox)
                table_structure["cells"] = organized_cells
                
                # Extract grid structure
                grid_info = self._extract_grid_structure(organized_cells)
                table_structure["grid_structure"] = grid_info
                
                # Extract rows and columns
                rows, columns = self._extract_rows_and_columns(organized_cells, grid_info)
                table_structure["rows"] = rows
                table_structure["columns"] = columns
                
                # Identify headers
                headers = self._identify_headers(organized_cells, grid_info)
                table_structure["headers"] = headers
                
                print(f"        Extracted: {grid_info['num_rows']} rows, {grid_info['num_columns']} columns")
                print(f"        Headers: {len(headers['column_headers'])} column headers, {len(headers['row_headers'])} row headers")
            
        except Exception as e:
            print(f"        WARNING  Error parsing TableFormer output: {e}")
        
        return table_structure
    
    def _organize_cells_into_grid(self, cells: List[Dict], table_bbox: List) -> List[Dict]:
        """Organize cells into a grid structure based on their positions"""
        if not cells:
            return []
        
        # Sort cells by y-position (top to bottom), then x-position (left to right)
        sorted_cells = sorted(cells, key=lambda cell: (cell['bbox'][1], cell['bbox'][0]))
        
        # Group cells into rows based on y-position
        rows = []
        current_row = []
        current_y = None
        y_tolerance = 20  # Pixels tolerance for grouping into same row
        
        for cell in sorted_cells:
            cell_y = cell['bbox'][1]
            
            if current_y is None or abs(cell_y - current_y) <= y_tolerance:
                # Same row
                current_row.append(cell)
                current_y = cell_y if current_y is None else min(current_y, cell_y)
            else:
                # New row
                if current_row:
                    rows.append(current_row)
                current_row = [cell]
                current_y = cell_y
        
        if current_row:
            rows.append(current_row)
        
        # Assign row and column indices
        organized_cells = []
        for row_idx, row_cells in enumerate(rows):
            # Sort cells in row by x-position
            row_cells.sort(key=lambda cell: cell['bbox'][0])
            
            for col_idx, cell in enumerate(row_cells):
                cell['row'] = row_idx
                cell['col'] = col_idx
                cell['cell_id'] = f"R{row_idx}C{col_idx}"
                organized_cells.append(cell)
        
        return organized_cells
    
    def _process_tableformer_cell(self, cell: Dict, cell_idx: int) -> Dict:
        """Process individual cell from TableFormer output"""
        cell_data = {
            "cell_id": f"cell_{cell_idx}",
            "bbox": cell.get('bbox', [0, 0, 0, 0]),
            "text": cell.get('text', ''),
            "row": cell.get('row', 0),
            "col": cell.get('col', 0),
            "rowspan": cell.get('rowspan', 1),
            "colspan": cell.get('colspan', 1),
            "is_header": cell.get('is_header', False),
            "confidence": cell.get('confidence', 0.0),
            "cell_type": self._determine_cell_type(cell)
        }
        return cell_data
    
    def _determine_cell_type(self, cell: Dict) -> str:
        """Determine cell type based on TableFormer output"""
        if cell.get('is_header', False):
            return "header"
        elif cell.get('row', 0) == 0:
            return "column_header"
        elif cell.get('col', 0) == 0:
            return "row_header"
        else:
            return "data_cell"
    
    def _extract_grid_structure(self, cells: List[Dict]) -> Dict:
        """Extract grid structure from cells"""
        if not cells:
            return {"num_rows": 0, "num_columns": 0, "cell_matrix": []}
        
        # Find maximum row and column
        max_row = max(cell.get('row', 0) for cell in cells)
        max_col = max(cell.get('col', 0) for cell in cells)
        
        num_rows = max_row + 1
        num_columns = max_col + 1
        
        # Create cell matrix
        cell_matrix = [[None for _ in range(num_columns)] for _ in range(num_rows)]
        
        for cell in cells:
            row = cell.get('row', 0)
            col = cell.get('col', 0)
            if 0 <= row < num_rows and 0 <= col < num_columns:
                cell_matrix[row][col] = cell
        
        return {
            "num_rows": num_rows,
            "num_columns": num_columns,
            "cell_matrix": cell_matrix
        }
    
    def _extract_rows_and_columns(self, cells: List[Dict], grid_info: Dict) -> Tuple[List[Dict], List[Dict]]:
        """Extract row and column information"""
        rows = []
        columns = []
        
        num_rows = grid_info["num_rows"]
        num_columns = grid_info["num_columns"]
        
        # Extract rows
        for row_idx in range(num_rows):
            row_cells = [cell for cell in cells if cell.get('row', 0) == row_idx]
            if row_cells:
                row_bbox = self._calculate_row_bbox(row_cells)
                rows.append({
                    "row_id": f"row_{row_idx}",
                    "row_index": row_idx,
                    "bbox": row_bbox,
                    "cells": row_cells,
                    "num_cells": len(row_cells)
                })
        
        # Extract columns
        for col_idx in range(num_columns):
            col_cells = [cell for cell in cells if cell.get('col', 0) == col_idx]
            if col_cells:
                col_bbox = self._calculate_column_bbox(col_cells)
                columns.append({
                    "column_id": f"col_{col_idx}",
                    "column_index": col_idx,
                    "bbox": col_bbox,
                    "cells": col_cells,
                    "num_cells": len(col_cells)
                })
        
        return rows, columns
    
    def _calculate_row_bbox(self, row_cells: List[Dict]) -> List[int]:
        """Calculate bounding box for a row"""
        if not row_cells:
            return [0, 0, 0, 0]
        
        x_min = min(cell['bbox'][0] for cell in row_cells)
        y_min = min(cell['bbox'][1] for cell in row_cells)
        x_max = max(cell['bbox'][2] for cell in row_cells)
        y_max = max(cell['bbox'][3] for cell in row_cells)
        
        return [x_min, y_min, x_max, y_max]
    
    def _calculate_column_bbox(self, col_cells: List[Dict]) -> List[int]:
        """Calculate bounding box for a column"""
        if not col_cells:
            return [0, 0, 0, 0]
        
        x_min = min(cell['bbox'][0] for cell in col_cells)
        y_min = min(cell['bbox'][1] for cell in col_cells)
        x_max = max(cell['bbox'][2] for cell in col_cells)
        y_max = max(cell['bbox'][3] for cell in col_cells)
        
        return [x_min, y_min, x_max, y_max]
    
    def _identify_headers(self, cells: List[Dict], grid_info: Dict) -> Dict:
        """Identify column and row headers"""
        column_headers = []
        row_headers = []
        
        # Column headers (first row)
        first_row_cells = [cell for cell in cells if cell.get('row', 0) == 0]
        for cell in first_row_cells:
            if cell.get('cell_type') in ['header', 'column_header']:
                column_headers.append(cell)
        
        # Row headers (first column)
        first_col_cells = [cell for cell in cells if cell.get('col', 0) == 0]
        for cell in first_col_cells:
            if cell.get('cell_type') in ['header', 'row_header']:
                row_headers.append(cell)
        
        return {
            "column_headers": column_headers,
            "row_headers": row_headers
        }
    
    def _create_visualization(self, input_file: str, output_prefix: str, intermediate_dir: Path, 
                            layout_result: Dict, table_result: Dict, table_transformer_result: Dict = None) -> Path:
        """Create visualization of layout and table detection results"""
        print("  VISUALIZATION Creating layout/table visualization...")
        
        # Create separate visualization directories inside stage1
        stage1_dir = intermediate_dir / "stage1_layout_table"
        layout_viz_dir = stage1_dir / "layout_outputs"
        tableformer_viz_dir = stage1_dir / "tableformer_outputs"
        table_transformer_viz_dir = stage1_dir / "table_transformer_outputs"
        layout_viz_dir.mkdir(parents=True, exist_ok=True)
        tableformer_viz_dir.mkdir(parents=True, exist_ok=True)
        table_transformer_viz_dir.mkdir(parents=True, exist_ok=True)
        
        # Create layout visualization
        layout_viz_path = layout_viz_dir / f"{output_prefix}_layout_annotated.png"
        tableformer_viz_path = tableformer_viz_dir / f"{output_prefix}_tableformer_annotated.png"
        table_transformer_viz_path = table_transformer_viz_dir / f"{output_prefix}_table_transformer_annotated.png"
        
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
            
            # Draw enhanced table structure from TableFormer results
            enhanced_tables = table_result.get("enhanced_table_structure", [])
            for i, table_structure in enumerate(enhanced_tables):
                self._draw_enhanced_table_structure(tableformer_draw, table_structure, i, small_font)
            
            # Create Table Transformer visualization
            if table_transformer_result:
                table_transformer_canvas = Image.new('RGB', (img_width, img_height), 'white')
                table_transformer_canvas.paste(original_img)
                table_transformer_draw = ImageDraw.Draw(table_transformer_canvas)
                
                # Draw Table Transformer structure
                enhanced_structure = table_transformer_result.get("enhanced_table_structure", {})
                self._draw_table_transformer_structure(table_transformer_draw, enhanced_structure, small_font)
                
                table_transformer_canvas.save(table_transformer_viz_path)
                print(f"    SUCCESS Table Transformer visualization saved: {table_transformer_viz_path}")
            
            # Save visualizations
            layout_canvas.save(layout_viz_path)
            tableformer_canvas.save(tableformer_viz_path)
            print(f"    SUCCESS Layout visualization saved: {layout_viz_path}")
            print(f"    SUCCESS TableFormer visualization saved: {tableformer_viz_path}")
            
            return layout_viz_path
            
        except Exception as e:
            print(f"    ERROR Error creating visualization: {e}")
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
    
    def _draw_enhanced_table_structure(self, draw: ImageDraw.Draw, table_structure: Dict, table_index: int, font):
        """Draw enhanced table structure like Table Transformer with different colors for different elements"""
        
        # Table Transformer-like colors
        colors = {
            'table': (0.000, 0.447, 0.741),           # Blue
            'table_column': (0.850, 0.325, 0.098),    # Red  
            'table_row': (0.929, 0.694, 0.125),       # Orange
            'table_column_header': (0.494, 0.184, 0.556),  # Purple
            'table_projected_row_header': (0.466, 0.674, 0.188),  # Green
            'table_spanning_cell': (0.301, 0.745, 0.933)  # Light Blue
        }
        
        # Convert colors to RGB (0-255)
        rgb_colors = {}
        for key, color in colors.items():
            rgb_colors[key] = tuple(int(c * 255) for c in color)
        
        # Draw table boundary
        table_bbox = table_structure.get("table_bbox", [0, 0, 100, 100])
        if len(table_bbox) >= 4:
            x1, y1, x2, y2 = map(int, table_bbox[:4])
            draw.rectangle([x1, y1, x2, y2], outline=rgb_colors['table'], width=3)
            draw.text((x1, y1-25), f"Table {table_index+1}", fill=rgb_colors['table'], font=font)
        
        # Draw rows
        rows = table_structure.get("rows", [])
        for row_idx, row in enumerate(rows):
            row_bbox = row.get("bbox", [0, 0, 0, 0])
            if len(row_bbox) >= 4:
                rx1, ry1, rx2, ry2 = map(int, row_bbox[:4])
                draw.rectangle([rx1, ry1, rx2, ry2], outline=rgb_colors['table_row'], width=2)
                draw.text((rx1+5, ry1+5), f"Row {row_idx}", fill=rgb_colors['table_row'], font=font)
        
        # Draw columns  
        columns = table_structure.get("columns", [])
        for col_idx, column in enumerate(columns):
            col_bbox = column.get("bbox", [0, 0, 0, 0])
            if len(col_bbox) >= 4:
                cx1, cy1, cx2, cy2 = map(int, col_bbox[:4])
                draw.rectangle([cx1, cy1, cx2, cy2], outline=rgb_colors['table_column'], width=2)
                draw.text((cx1+5, cy1+5), f"Col {col_idx}", fill=rgb_colors['table_column'], font=font)
        
        # Draw individual cells with different colors based on type
        cells = table_structure.get("cells", [])
        for cell in cells:
            cell_bbox = cell.get("bbox", [0, 0, 0, 0])
            cell_type = cell.get("cell_type", "data_cell")
            cell_id = cell.get("cell_id", "")
            
            if len(cell_bbox) >= 4:
                cx1, cy1, cx2, cy2 = map(int, cell_bbox[:4])
                
                # Choose color based on cell type
                if cell_type == "header" or cell_type == "column_header":
                    color = rgb_colors['table_column_header']
                elif cell_type == "row_header":
                    color = rgb_colors['table_projected_row_header']
                else:
                    color = rgb_colors['table_spanning_cell']
                
                # Draw cell boundary
                draw.rectangle([cx1, cy1, cx2, cy2], outline=color, width=1)
                
                # Draw cell ID
                draw.text((cx1+2, cy1+2), cell_id, fill=color, font=font)
        
        # Draw headers with special highlighting
        headers = table_structure.get("headers", {})
        column_headers = headers.get("column_headers", [])
        row_headers = headers.get("row_headers", [])
        
        # Highlight column headers
        for header in column_headers:
            header_bbox = header.get("bbox", [0, 0, 0, 0])
            if len(header_bbox) >= 4:
                hx1, hy1, hx2, hy2 = map(int, header_bbox[:4])
                # Draw thicker border for headers
                draw.rectangle([hx1, hy1, hx2, hy2], outline=rgb_colors['table_column_header'], width=3)
                draw.text((hx1+5, hy1+5), "HEADER", fill=rgb_colors['table_column_header'], font=font)
        
        # Highlight row headers
        for header in row_headers:
            header_bbox = header.get("bbox", [0, 0, 0, 0])
            if len(header_bbox) >= 4:
                hx1, hy1, hx2, hy2 = map(int, header_bbox[:4])
                # Draw thicker border for headers
                draw.rectangle([hx1, hy1, hx2, hy2], outline=rgb_colors['table_projected_row_header'], width=3)
                draw.text((hx1+5, hy1+5), "ROW_HEADER", fill=rgb_colors['table_projected_row_header'], font=font)
    
    def _draw_table_transformer_structure(self, draw: ImageDraw.Draw, enhanced_structure: Dict, font):
        """Draw Table Transformer structure with exact colors from the notebook"""
        
        # Table Transformer colors (exact from notebook)
        colors = {
            'table': (0.000, 0.447, 0.741),           # Blue
            'table_column': (0.850, 0.325, 0.098),    # Red  
            'table_row': (0.929, 0.694, 0.125),       # Orange
            'table_column_header': (0.494, 0.184, 0.556),  # Purple
            'table_projected_row_header': (0.466, 0.674, 0.188),  # Green
            'table_spanning_cell': (0.301, 0.745, 0.933)  # Light Blue
        }
        
        # Convert colors to RGB (0-255)
        rgb_colors = {}
        for key, color in colors.items():
            rgb_colors[key] = tuple(int(c * 255) for c in color)
        
        # Draw tables
        tables = enhanced_structure.get('tables', [])
        for i, table in enumerate(tables):
            bbox = table.get('bbox', [0, 0, 100, 100])
            score = table.get('score', 0.0)
            if len(bbox) >= 4:
                x1, y1, x2, y2 = map(int, bbox[:4])
                draw.rectangle([x1, y1, x2, y2], outline=rgb_colors['table'], width=3)
                draw.text((x1, y1-25), f"Table {i+1}: {score:.2f}", fill=rgb_colors['table'], font=font)
        
        # Draw table columns
        table_columns = enhanced_structure.get('table_columns', [])
        for i, column in enumerate(table_columns):
            bbox = column.get('bbox', [0, 0, 100, 100])
            score = column.get('score', 0.0)
            if len(bbox) >= 4:
                x1, y1, x2, y2 = map(int, bbox[:4])
                draw.rectangle([x1, y1, x2, y2], outline=rgb_colors['table_column'], width=2)
                draw.text((x1+5, y1+5), f"Col {i+1}: {score:.2f}", fill=rgb_colors['table_column'], font=font)
        
        # Draw table rows
        table_rows = enhanced_structure.get('table_rows', [])
        for i, row in enumerate(table_rows):
            bbox = row.get('bbox', [0, 0, 100, 100])
            score = row.get('score', 0.0)
            if len(bbox) >= 4:
                x1, y1, x2, y2 = map(int, bbox[:4])
                draw.rectangle([x1, y1, x2, y2], outline=rgb_colors['table_row'], width=2)
                draw.text((x1+5, y1+5), f"Row {i+1}: {score:.2f}", fill=rgb_colors['table_row'], font=font)
        
        # Draw column headers
        column_headers = enhanced_structure.get('column_headers', [])
        for i, header in enumerate(column_headers):
            bbox = header.get('bbox', [0, 0, 100, 100])
            score = header.get('score', 0.0)
            if len(bbox) >= 4:
                x1, y1, x2, y2 = map(int, bbox[:4])
                draw.rectangle([x1, y1, x2, y2], outline=rgb_colors['table_column_header'], width=3)
                draw.text((x1+5, y1+5), f"Col Header {i+1}: {score:.2f}", fill=rgb_colors['table_column_header'], font=font)
        
        # Draw row headers
        row_headers = enhanced_structure.get('row_headers', [])
        for i, header in enumerate(row_headers):
            bbox = header.get('bbox', [0, 0, 100, 100])
            score = header.get('score', 0.0)
            if len(bbox) >= 4:
                x1, y1, x2, y2 = map(int, bbox[:4])
                draw.rectangle([x1, y1, x2, y2], outline=rgb_colors['table_projected_row_header'], width=3)
                draw.text((x1+5, y1+5), f"Row Header {i+1}: {score:.2f}", fill=rgb_colors['table_projected_row_header'], font=font)
        
        # Draw spanning cells
        spanning_cells = enhanced_structure.get('spanning_cells', [])
        for i, cell in enumerate(spanning_cells):
            bbox = cell.get('bbox', [0, 0, 100, 100])
            score = cell.get('score', 0.0)
            if len(bbox) >= 4:
                x1, y1, x2, y2 = map(int, bbox[:4])
                draw.rectangle([x1, y1, x2, y2], outline=rgb_colors['table_spanning_cell'], width=1)
                draw.text((x1+2, y1+2), f"Cell {i+1}: {score:.2f}", fill=rgb_colors['table_spanning_cell'], font=font)
    
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
        print("  FILES Creating coordinates JSON...")
        
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
        
        # Save tableformer coordinates with enhanced structure
        # Extract tableformer results from normalized tables
        normalized_tables = table_result.get("normalized_tables", [])
        tableformer_results = []
        for table in normalized_tables:
            if table.get("tableformer_results"):
                tableformer_results.append(table["tableformer_results"])
        
        tableformer_coords_data = {
            "stage": 1,
            "stage_name": "table_detection",
            "processing_timestamp": self._get_timestamp(),
            "tableformer_results": tableformer_results,
            "enhanced_table_structure": table_result.get("enhanced_table_structure", []),
            "processing_metadata": table_result.get("processing_metadata", {}),
            "summary": {
                "total_tables": table_result.get("processing_metadata", {}).get("total_tables", 0),
                "total_cells": table_result.get("processing_metadata", {}).get("total_cells", 0),
                "total_rows": table_result.get("processing_metadata", {}).get("total_rows", 0),
                "total_columns": table_result.get("processing_metadata", {}).get("total_columns", 0)
            }
        }
        
        with open(tableformer_coords_path, 'w') as f:
            json.dump(tableformer_coords_data, f, indent=2)
        
        print(f"    SUCCESS Layout coordinates JSON saved: {layout_coords_path}")
        print(f"    SUCCESS TableFormer coordinates JSON saved: {tableformer_coords_path}")
        return layout_coords_path
    
    def _count_element_types(self, layout_elements: List[Dict]) -> Dict[str, int]:
        """Count different types of layout elements"""
        type_counts = {}
        for element in layout_elements:
            label = element.get('label', 'unknown')
            type_counts[label] = type_counts.get(label, 0) + 1
        return type_counts
    
    def _create_coordinate_reference_visualization(self, input_file: str, output_prefix: str, intermediate_dir: Path) -> Path:
        """Create coordinate reference visualization with grid overlay"""
        print("  COORDINATES Creating coordinate reference visualization...")
        
        # Create inputs_coordinates directory inside stage1
        stage1_dir = intermediate_dir / "stage1_layout_table"
        inputs_coords_dir = stage1_dir / "inputs_coordinates"
        inputs_coords_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = inputs_coords_dir / f"{output_prefix}_coordinate_reference.png"
        
        try:
            # Load original image
            image = Image.open(input_file).convert("RGB")
            width, height = image.size
            
            # Create a copy for drawing
            coord_image = image.copy()
            draw = ImageDraw.Draw(coord_image)
            
            # Try to load a font, fallback to default if not available
            try:
                font = ImageFont.truetype("arial.ttf", 12)
                small_font = ImageFont.truetype("arial.ttf", 10)
            except:
                font = ImageFont.load_default()
                small_font = ImageFont.load_default()
            
            # Define grid parameters
            grid_spacing = 100  # 100 pixel grid spacing
            line_color = (255, 0, 0)  # Red grid lines
            text_color = (255, 0, 0)  # Red text
            corner_color = (0, 255, 0)  # Green for corners
            
            # Draw vertical grid lines
            for x in range(0, width, grid_spacing):
                draw.line([(x, 0), (x, height)], fill=line_color, width=1)
                # Add x-coordinate labels
                if x > 0:  # Skip origin label
                    draw.text((x + 2, 2), str(x), fill=text_color, font=small_font)
            
            # Draw horizontal grid lines
            for y in range(0, height, grid_spacing):
                draw.line([(0, y), (width, y)], fill=line_color, width=1)
                # Add y-coordinate labels
                if y > 0:  # Skip origin label
                    draw.text((2, y + 2), str(y), fill=text_color, font=small_font)
            
            # Draw corner markers
            corner_size = 10
            # Top-left corner
            draw.rectangle([(0, 0), (corner_size, corner_size)], outline=corner_color, width=2)
            draw.text((corner_size + 2, 2), "(0,0)", fill=corner_color, font=font)
            
            # Top-right corner
            draw.rectangle([(width - corner_size, 0), (width, corner_size)], outline=corner_color, width=2)
            draw.text((width - 50, 2), f"({width},0)", fill=corner_color, font=font)
            
            # Bottom-left corner
            draw.rectangle([(0, height - corner_size), (corner_size, height)], outline=corner_color, width=2)
            draw.text((corner_size + 2, height - 20), f"(0,{height})", fill=corner_color, font=font)
            
            # Bottom-right corner
            draw.rectangle([(width - corner_size, height - corner_size), (width, height)], outline=corner_color, width=2)
            draw.text((width - 80, height - 20), f"({width},{height})", fill=corner_color, font=font)
            
            # Add image dimensions info
            info_text = f"Image: {width} x {height} pixels"
            draw.text((10, height - 40), info_text, fill=text_color, font=font)
            
            # Add coordinate system info
            coord_text = f"Grid: {grid_spacing}px spacing | Origin: Top-left (0,0)"
            draw.text((10, height - 25), coord_text, fill=text_color, font=small_font)
            
            # Save the coordinate reference image
            coord_image.save(output_path)
            
            print(f"    SUCCESS Coordinate reference saved: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"    ERROR Failed to create coordinate reference: {e}")
            # Return a placeholder path even if creation failed
            return output_path
    
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
        print(f"\nSUCCESS Stage 1 completed successfully!")
        print(f"   Layout elements: {result.get('layout_count', 0)}")
        print(f"   Tables: {result.get('table_count', 0)}")
        print(f"   Visualization: {result.get('visualization_path', 'N/A')}")
        print(f"   Coordinates: {result.get('coordinates_path', 'N/A')}")
        print(f"   Coordinate Reference: {result.get('coordinate_reference_path', 'N/A')}")
    else:
        print(f"\nERROR Stage 1 failed: {result.get('error', 'Unknown error')}")
        sys.exit(1)

if __name__ == "__main__":
    main()