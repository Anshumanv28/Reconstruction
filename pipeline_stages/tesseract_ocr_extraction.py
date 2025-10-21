#!/usr/bin/env python3
"""
Stage 2: Tesseract OCR Character Extraction

This module handles:
- OCR text extraction using Tesseract
- Text block detection and positioning
- Creates visualization output (PNG)
- Creates coordinate JSON with structured data
"""

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont
import pytesseract
import cv2
import numpy as np
from datetime import datetime

class TesseractOCRExtractor:
    """Handles OCR text extraction using Tesseract"""
    
    def __init__(self):
        self.stage_name = "ocr_extraction"
        self.ocr_config = {
            'psm': 6,  # Assume a single uniform block of text
            'oem': 3,  # Default OCR Engine Mode
            'lang': 'eng'
        }
    
    def _load_stage1_data(self, output_prefix: str, intermediate_dir: Path) -> Dict:
        """Load Stage 1 layout data"""
        stage1_dir = intermediate_dir / "stage1_layout_table" / "layout_outputs"
        stage1_file = stage1_dir / f"{output_prefix}_layout_coordinates.json"
        
        if stage1_file.exists():
            with open(stage1_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        print(f"    ⚠️  Stage 1 layout data not found, using fallback")
        return {"layout_elements": []}
    
    def process_image(self, input_file: str, output_prefix: str, intermediate_dir: Path) -> Dict:
        """Process a single image for OCR text extraction"""
        print(f"🔍 Processing: {input_file}")
        
        try:
            # Load Stage 1 layout data to guide OCR strategy
            print("  📥 Loading layout information from Stage 1...")
            layout_data = self._load_stage1_data(output_prefix, intermediate_dir)
            
            # Run layout-aware OCR extraction
            ocr_result = self._run_layout_aware_ocr(input_file, layout_data)
            
            # Create visualization
            visualization_path = self._create_visualization(
                input_file, output_prefix, intermediate_dir, ocr_result
            )
            
            # Create coordinate JSON
            coordinates_path = self._create_coordinates_json(
                output_prefix, intermediate_dir, ocr_result
            )
            
            return {
                "success": True,
                "text_block_count": len(ocr_result.get("text_blocks", [])),
                "total_text_length": len(ocr_result.get("full_text", "")),
                "visualization_path": str(visualization_path),
                "coordinates_path": str(coordinates_path),
                "ocr_result": ocr_result
            }
            
        except Exception as e:
            print(f"❌ Error in OCR extraction: {e}")
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e)}
    
    def _run_layout_aware_ocr(self, input_file: str, layout_data: Dict) -> Dict:
        """Layout-aware OCR: different strategies for tables vs paragraphs"""
        print("  📝 Running layout-aware OCR...")
        
        layout_elements = layout_data.get("layout_elements", [])
        
        # Categorize layout elements
        table_regions = []
        paragraph_regions = []
        other_regions = []
        
        for element in layout_elements:
            label = element.get("label", "").lower()
            bbox = element.get("bbox", [])
            
            if label == "table":
                table_regions.append({"type": "table", "bbox": bbox, "element": element})
            elif label in ["paragraph", "text", "caption"]:
                paragraph_regions.append({"type": "paragraph", "bbox": bbox, "element": element})
            else:
                other_regions.append({"type": label, "bbox": bbox, "element": element})
        
        print(f"    📊 Layout: {len(table_regions)} tables, {len(paragraph_regions)} paragraphs, {len(other_regions)} other")
        
        # Load image
        image = cv2.imread(input_file)
        if image is None:
            raise ValueError(f"Could not load image: {input_file}")
        
        all_text_blocks = []
        
        # Process tables with WORD-LEVEL OCR
        if table_regions:
            print(f"    📋 Processing {len(table_regions)} table regions (word-level)...")
            table_text_blocks = self._extract_table_ocr(image, table_regions)
            all_text_blocks.extend(table_text_blocks)
            print(f"       ✓ Extracted {len(table_text_blocks)} word blocks from tables")
        
        # Process paragraphs with PARAGRAPH-LEVEL OCR
        if paragraph_regions:
            print(f"    📝 Processing {len(paragraph_regions)} paragraph regions (paragraph-level)...")
            paragraph_text_blocks = self._extract_paragraph_ocr(image, paragraph_regions)
            all_text_blocks.extend(paragraph_text_blocks)
            print(f"       ✓ Extracted {len(paragraph_text_blocks)} paragraph blocks")
        
        # Process other regions
        if other_regions:
            print(f"    📄 Processing {len(other_regions)} other regions...")
            other_text_blocks = self._extract_other_ocr(image, other_regions)
            all_text_blocks.extend(other_text_blocks)
            print(f"       ✓ Extracted {len(other_text_blocks)} text blocks")
        
        # Extract full text for reference
        full_text = pytesseract.image_to_string(image, config="--psm 6 --oem 3 -l eng")
        
        print(f"    ✅ Layout-aware OCR completed: {len(all_text_blocks)} total blocks")
        
        return {
            "full_text": full_text,
            "text_blocks": all_text_blocks,
            "ocr_config": {"layout_aware": True, "oem": 3, "lang": "eng"},
            "image_size": image.shape[:2][::-1],
            "processing_timestamp": datetime.now().isoformat(),
            "layout_breakdown": {
                "table_blocks": len([b for b in all_text_blocks if b.get("region_type") == "table"]),
                "paragraph_blocks": len([b for b in all_text_blocks if b.get("region_type") == "paragraph"]),
                "other_blocks": len([b for b in all_text_blocks if b.get("region_type") not in ["table", "paragraph"]])
            }
        }
    
    def _extract_table_ocr(self, image: np.ndarray, table_regions: List[Dict]) -> List[Dict]:
        """Extract WORD-LEVEL OCR for table regions"""
        text_blocks = []
        
        for region in table_regions:
            bbox = region["bbox"]
            if len(bbox) < 4:
                continue
            
            x1, y1, x2, y2 = map(int, bbox)
            
            # Crop table region
            table_crop = image[y1:y2, x1:x2]
            
            # Preprocess for table OCR
            processed = self._preprocess_image(table_crop)
            
            # Extract WORD-LEVEL data (PSM 6 is good for tables)
            config = "--psm 6 --oem 3 -l eng"
            ocr_data = pytesseract.image_to_data(
                processed,
                config=config,
                output_type=pytesseract.Output.DICT
            )
            
            # Process each word individually (fine-grained for table cells)
            for i in range(len(ocr_data['text'])):
                text = ocr_data['text'][i].strip()
                conf = int(ocr_data['conf'][i])
                
                if text and conf >= -1:  # Accept all detected text
                    # Adjust coordinates back to original image
                    word_bbox = [
                        x1 + int(ocr_data['left'][i]),
                        y1 + int(ocr_data['top'][i]),
                        x1 + int(ocr_data['left'][i]) + int(ocr_data['width'][i]),
                        y1 + int(ocr_data['top'][i]) + int(ocr_data['height'][i])
                    ]
                    
                    text_blocks.append({
                        'text': text,
                        'confidence': max(0, conf),
                        'bbox': word_bbox,
                        'region_type': 'table',
                        'granularity': 'word'
                    })
        
        return text_blocks
    
    def _extract_paragraph_ocr(self, image: np.ndarray, paragraph_regions: List[Dict]) -> List[Dict]:
        """Extract PARAGRAPH-LEVEL OCR for paragraph regions"""
        text_blocks = []
        
        for region in paragraph_regions:
            bbox = region["bbox"]
            if len(bbox) < 4:
                continue
            
            x1, y1, x2, y2 = map(int, bbox)
            
            # Crop paragraph region
            para_crop = image[y1:y2, x1:x2]
            
            # Preprocess for paragraph OCR
            processed = self._preprocess_image(para_crop)
            
            # Extract PARAGRAPH-LEVEL text (PSM 6 for block of text)
            config = "--psm 6 --oem 3 -l eng"
            full_text = pytesseract.image_to_string(processed, config=config).strip()
            
            # Also get confidence data
            ocr_data = pytesseract.image_to_data(
                processed,
                config=config,
                output_type=pytesseract.Output.DICT
            )
            
            # Calculate average confidence for this paragraph
            confidences = [int(c) for c in ocr_data['conf'] if int(c) >= 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            if full_text:
                text_blocks.append({
                    'text': full_text,
                    'confidence': int(avg_confidence),
                    'bbox': [x1, y1, x2, y2],
                    'region_type': 'paragraph',
                    'granularity': 'paragraph'
                })
        
        return text_blocks
    
    def _extract_other_ocr(self, image: np.ndarray, other_regions: List[Dict]) -> List[Dict]:
        """Extract OCR for other regions (headers, captions, etc.)"""
        text_blocks = []
        
        for region in other_regions:
            bbox = region["bbox"]
            region_type = region["type"]
            
            if len(bbox) < 4:
                continue
            
            x1, y1, x2, y2 = map(int, bbox)
            
            # Crop region
            region_crop = image[y1:y2, x1:x2]
            
            # Preprocess
            processed = self._preprocess_image(region_crop)
            
            # Extract text (single line for most elements)
            config = "--psm 7 --oem 3 -l eng"  # PSM 7 = single line
            full_text = pytesseract.image_to_string(processed, config=config).strip()
            
            # Get confidence
            ocr_data = pytesseract.image_to_data(
                processed,
                config=config,
                output_type=pytesseract.Output.DICT
            )
            
            confidences = [int(c) for c in ocr_data['conf'] if int(c) >= 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            if full_text:
                text_blocks.append({
                    'text': full_text,
                    'confidence': int(avg_confidence),
                    'bbox': [x1, y1, x2, y2],
                    'region_type': region_type,
                    'granularity': 'single'
                })
        
        return text_blocks
    
    def _run_tesseract_ocr(self, input_file: str) -> Dict:
        """IMPROVED: Streamlined OCR with table-focused processing"""
        print("  📝 Running Tesseract OCR...")
        
        try:
            # Load and preprocess image
            image = cv2.imread(input_file)
            if image is None:
                raise ValueError(f"Could not load image: {input_file}")
            
            processed_image = self._preprocess_image(image)
            
            # IMPROVEMENT: Use targeted PSM modes for table processing
            table_psm_modes = [6, 11, 12]  # Focus on modes good for tables
            
            best_ocr_data = None
            best_score = 0
            best_psm = 6
            
            print(f"    🔍 Testing table-optimized PSM modes...")
            
            for psm_value in table_psm_modes:
                try:
                    config = f"--psm {psm_value} --oem 3 -l eng"
                    ocr_data = pytesseract.image_to_data(
                        processed_image, 
                        config=config,
                        output_type=pytesseract.Output.DICT
                    )
                    
                    # IMPROVED: Table-specific quality scoring
                    score = self._calculate_table_ocr_quality(ocr_data)
                    
                    print(f"       PSM {psm_value}: Score {score:.1f}")
                    
                    if score > best_score:
                        best_score = score
                        best_ocr_data = ocr_data
                        best_psm = psm_value
                        
                except Exception as e:
                    print(f"       PSM {psm_value}: Failed - {e}")
                    continue
            
            if best_ocr_data is None:
                # Fallback to basic mode
                config = "--psm 6 --oem 3 -l eng"
                best_ocr_data = pytesseract.image_to_data(
                    processed_image, 
                    config=config,
                    output_type=pytesseract.Output.DICT
                )
                best_psm = 6
            
            print(f"    ✅ Selected PSM {best_psm} (Score: {best_score:.1f})")
            
            # Extract full text
            full_text = pytesseract.image_to_string(
                processed_image,
                config=f"--psm {best_psm} --oem 3 -l eng"
            )
            
            # Process into text blocks
            text_blocks = self._process_ocr_data(best_ocr_data)
            
            print(f"    ✅ OCR completed: {len(text_blocks)} text blocks")
            
            return {
                "full_text": full_text,
                "text_blocks": text_blocks,
                "ocr_config": {"psm": best_psm, "oem": 3, "lang": "eng"},
                "image_size": image.shape[:2][::-1],
                "processing_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"    ❌ Tesseract OCR failed: {e}")
            return {
                "full_text": "",
                "text_blocks": [],
                "error": str(e)
            }
    
    def _calculate_table_ocr_quality(self, ocr_data: Dict) -> float:
        """NEW: Calculate OCR quality score optimized for table content"""
        word_count = 0
        total_confidence = 0
        meaningful_words = 0
        numeric_content = 0
        header_words = 0
        
        # Table-specific quality indicators
        table_keywords = ['total', 'december', 'owned', 'optioned', 'controlled', 
                         'topic', 'facilitator', 'time', 'discussion', 'action']
        
        for i in range(len(ocr_data['text'])):
            text = ocr_data['text'][i].strip().lower()
            conf = int(ocr_data['conf'][i])
            
            if text and conf >= -1:
                word_count += 1
                total_confidence += max(0, conf)  # Don't let negative confidence hurt too much
                
                # Meaningful word detection
                if len(text) > 2:
                    meaningful_words += 1
                
                # Numeric content (common in tables)
                if any(char.isdigit() for char in text):
                    numeric_content += 1
                
                # Table header keywords
                if any(keyword in text for keyword in table_keywords):
                    header_words += 1
        
        if word_count == 0:
            return 0
        
        avg_confidence = total_confidence / word_count
        meaningful_ratio = meaningful_words / word_count
        numeric_ratio = numeric_content / word_count
        header_ratio = header_words / word_count
        
        # Combined score favoring table-like content
        score = (
            avg_confidence * 0.3 +           # Base confidence
            meaningful_words * 2.0 +         # Prefer meaningful words
            numeric_content * 1.5 +          # Tables often have numbers
            header_words * 3.0 +             # Table headers are valuable
            word_count * 0.1                 # More words generally better
        )
        
        return score
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """IMPROVED: Table-optimized image preprocessing"""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # IMPROVEMENT: Table-specific preprocessing pipeline
        
        # 1. Gentle noise reduction (preserve text edges)
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # 2. Enhance contrast for table text
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        # 3. Adaptive thresholding (better for varying lighting in tables)
        # Try adaptive threshold for table structures
        adaptive_thresh = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # 4. Morphological operations to clean up table structure
        # Remove small noise while preserving text
        kernel = np.ones((2,2), np.uint8)
        cleaned = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
    
    def _process_ocr_data(self, ocr_data: Dict) -> List[Dict]:
        """IMPROVED: Process raw OCR data with better table-aware grouping"""
        text_blocks = []
        
        # Extract all valid words with enhanced filtering
        words = []
        for i in range(len(ocr_data['text'])):
            if int(ocr_data['conf'][i]) >= -1:  # Accept ALL text
                text = ocr_data['text'][i].strip()
                if text:  # Only process non-empty text
                    word_data = {
                        'text': text,
                        'confidence': int(ocr_data['conf'][i]),
                        'bbox': [
                            int(ocr_data['left'][i]),
                            int(ocr_data['top'][i]),
                            int(ocr_data['left'][i]) + int(ocr_data['width'][i]),
                            int(ocr_data['top'][i]) + int(ocr_data['height'][i])
                        ],
                        'word_id': i,
                        'line_id': int(ocr_data['line_num'][i]),
                        'block_id': int(ocr_data['block_num'][i])
                    }
                    words.append(word_data)
        
        # IMPROVEMENT: Group words using table-aware logic
        text_blocks = self._group_words_for_table_cells(words)
        
        return text_blocks
    
    def _group_words_for_table_cells(self, words: List[Dict]) -> List[Dict]:
        """NEW: Table-aware word grouping logic"""
        if not words:
            return []
        
        # Sort words by position (top-to-bottom, left-to-right)
        sorted_words = sorted(words, key=lambda w: (w['bbox'][1], w['bbox'][0]))
        
        text_blocks = []
        used_indices = set()
        block_id = 0
        
        for i, word in enumerate(sorted_words):
            if i in used_indices:
                continue
            
            # Start new block
            current_block = {
                'text': word['text'],
                'confidence': word['confidence'],
                'bbox': word['bbox'].copy(),
                'words': [word],
                'block_id': block_id
            }
            used_indices.add(i)
            
            # Find nearby words to group (table cell proximity)
            for j, other_word in enumerate(sorted_words[i+1:], i+1):
                if j in used_indices:
                    continue
                
                if self._should_group_words_for_table(current_block, other_word):
                    # Add word to current block
                    current_block['text'] += ' ' + other_word['text']
                    # Use minimum confidence (most conservative)
                    current_block['confidence'] = min(current_block['confidence'], other_word['confidence'])
                    # Expand bounding box
                    current_block['bbox'] = self._merge_bboxes(current_block['bbox'], other_word['bbox'])
                    current_block['words'].append(other_word)
                    used_indices.add(j)
            
            text_blocks.append(current_block)
            block_id += 1
        
        return text_blocks
    
    def _should_group_words_for_table(self, current_block: Dict, word: Dict) -> bool:
        """NEW: Determine if words should be grouped for table processing"""
        block_bbox = current_block['bbox']
        word_bbox = word['bbox']
        
        # Calculate centers
        block_center_y = (block_bbox[1] + block_bbox[3]) / 2
        word_center_y = (word_bbox[1] + word_bbox[3]) / 2
        
        # Y proximity check (same row in table)
        y_distance = abs(block_center_y - word_center_y)
        same_row = y_distance <= 15  # Pixels tolerance for same table row
        
        if not same_row:
            return False
        
        # X proximity check (same cell or adjacent in same row)
        x_distance = word_bbox[0] - block_bbox[2]  # Gap between block end and word start
        close_horizontally = 0 <= x_distance <= 50  # Allow reasonable gap within table cells
        
        return close_horizontally
    
    def _merge_bboxes(self, bbox1: List[int], bbox2: List[int]) -> List[int]:
        """NEW: Merge two bounding boxes"""
        return [
            min(bbox1[0], bbox2[0]),  # min x
            min(bbox1[1], bbox2[1]),  # min y
            max(bbox1[2], bbox2[2]),  # max x
            max(bbox1[3], bbox2[3])   # max y
        ]
    
    def _create_visualization(self, input_file: str, output_prefix: str, intermediate_dir: Path, 
                            ocr_result: Dict) -> Path:
        """Create visualization of OCR results"""
        print("  🎨 Creating OCR visualization...")
        
        # Create output paths
        stage_dir = intermediate_dir / "stage2_ocr_extraction"
        viz_dir = stage_dir / "visualizations"
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = viz_dir / f"{output_prefix}_ocr_annotated.png"
        
        try:
            # Load original image
            original_img = Image.open(input_file)
            img_width, img_height = original_img.size
            
            # Create visualization canvas
            canvas = Image.new('RGB', (img_width, img_height), 'white')
            canvas.paste(original_img)
            draw = ImageDraw.Draw(canvas)
            
            # Try to load font
            try:
                font = ImageFont.truetype("arial.ttf", 10)
                small_font = ImageFont.truetype("arial.ttf", 8)
            except:
                font = ImageFont.load_default()
                small_font = ImageFont.load_default()
            
            # Draw text blocks
            text_blocks = ocr_result.get("text_blocks", [])
            for i, block in enumerate(text_blocks):
                self._draw_text_block(draw, block, i, font, small_font)
            
            # Save visualization
            canvas.save(output_path)
            print(f"    ✅ Visualization saved: {output_path}")
            
            return output_path
            
        except Exception as e:
            print(f"    ❌ Error creating visualization: {e}")
            # Create a simple placeholder image
            placeholder = Image.new('RGB', (800, 600), 'lightgray')
            placeholder.save(output_path)
            return output_path
    
    def _draw_text_block(self, draw: ImageDraw.Draw, block: Dict, index: int, font, small_font):
        """Draw a single text block with color coding by region type"""
        x1, y1, x2, y2 = block['bbox']
        text = block['text']
        confidence = block['confidence']
        region_type = block.get('region_type', 'unknown')
        granularity = block.get('granularity', 'word')
        
        # Convert to integers
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Choose color based on region type
        if region_type == 'table':
            color = (0, 128, 255)  # Blue for table text (word-level)
            width = 1
        elif region_type == 'paragraph':
            color = (0, 200, 100)  # Green for paragraph text (paragraph-level)
            width = 2
        elif region_type in ['section-header', 'title']:
            color = (255, 0, 255)  # Magenta for headers
            width = 2
        else:
            color = (128, 128, 128)  # Gray for other
            width = 1
        
        # Adjust alpha based on confidence (darker = higher confidence)
        if confidence < 50:
            # Draw lighter for low confidence
            color = tuple(min(255, c + 100) for c in color)
        
        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
        
        # Draw text (truncated if too long)
        if granularity == 'paragraph':
            # For paragraphs, show just first few words
            display_text = ' '.join(text.split()[:5])
            if len(text.split()) > 5:
                display_text += "..."
        else:
            display_text = text[:30] + "..." if len(text) > 30 else text
        
        draw.text((x1+2, y1+2), display_text, fill=color, font=font)
        
        # Draw region type and confidence
        label_text = f"{region_type[:4].upper()} {confidence}%"
        draw.text((x1+2, y2-12), label_text, fill=color, font=small_font)
    
    def _create_coordinates_json(self, output_prefix: str, intermediate_dir: Path, 
                               ocr_result: Dict) -> Path:
        """Create structured coordinate JSON"""
        print("  📄 Creating coordinates JSON...")
        
        # Create output paths
        stage_dir = intermediate_dir / "stage2_ocr_extraction"
        coordinates_dir = stage_dir / "coordinates"
        coordinates_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = coordinates_dir / f"{output_prefix}_stage2_coordinates.json"
        
        # Create structured data
        coordinates_data = {
            "stage": 2,
            "stage_name": "ocr_extraction",
            "processing_timestamp": ocr_result.get("processing_timestamp", datetime.now().isoformat()),
            "full_text": ocr_result.get("full_text", ""),
            "text_blocks": ocr_result.get("text_blocks", []),
            "ocr_config": ocr_result.get("ocr_config", self.ocr_config),
            "image_size": ocr_result.get("image_size", [0, 0]),
            "summary": {
                "total_text_blocks": len(ocr_result.get("text_blocks", [])),
                "total_characters": len(ocr_result.get("full_text", "")),
                "average_confidence": self._calculate_average_confidence(ocr_result.get("text_blocks", [])),
                "confidence_distribution": self._calculate_confidence_distribution(ocr_result.get("text_blocks", []))
            }
        }
        
        # Save coordinates JSON
        with open(output_path, 'w') as f:
            json.dump(coordinates_data, f, indent=2)
        
        print(f"    ✅ Coordinates JSON saved: {output_path}")
        return output_path
    
    def _calculate_average_confidence(self, text_blocks: List[Dict]) -> float:
        """Calculate average confidence of text blocks"""
        if not text_blocks:
            return 0.0
        
        total_confidence = sum(block.get('confidence', 0) for block in text_blocks)
        return total_confidence / len(text_blocks)
    
    def _calculate_confidence_distribution(self, text_blocks: List[Dict]) -> Dict[str, int]:
        """Calculate confidence distribution"""
        distribution = {
            "high_confidence": 0,    # > 80%
            "medium_confidence": 0,  # 60-80%
            "low_confidence": 0      # < 60%
        }
        
        for block in text_blocks:
            confidence = block.get('confidence', 0)
            if confidence > 80:
                distribution["high_confidence"] += 1
            elif confidence > 60:
                distribution["medium_confidence"] += 1
            else:
                distribution["low_confidence"] += 1
        
        return distribution

def main():
    """Main function for testing Stage 2"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Stage 2: Tesseract OCR Character Extraction")
    parser.add_argument("--input", "-i", required=True, help="Input image file")
    parser.add_argument("--output-prefix", "-o", help="Output prefix (default: input filename)")
    parser.add_argument("--intermediate-dir", default="intermediate_outputs", help="Intermediate outputs directory")
    
    args = parser.parse_args()
    
    # Set default output prefix
    if not args.output_prefix:
        args.output_prefix = Path(args.input).stem
    
    # Initialize extractor
    extractor = TesseractOCRExtractor()
    
    # Process image
    result = extractor.process_image(args.input, args.output_prefix, Path(args.intermediate_dir))
    
    # Print results
    if result.get("success", False):
        print(f"\n✅ Stage 2 completed successfully!")
        print(f"   Text blocks: {result.get('text_block_count', 0)}")
        print(f"   Total characters: {result.get('total_text_length', 0)}")
        print(f"   Visualization: {result.get('visualization_path', 'N/A')}")
        print(f"   Coordinates: {result.get('coordinates_path', 'N/A')}")
    else:
        print(f"\n❌ Stage 2 failed: {result.get('error', 'Unknown error')}")
        sys.exit(1)

if __name__ == "__main__":
    main()
