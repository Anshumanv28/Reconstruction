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
    
    def process_image(self, input_file: str, output_prefix: str, intermediate_dir: Path) -> Dict:
        """Process a single image for OCR text extraction"""
        print(f"🔍 Processing: {input_file}")
        
        try:
            # Run OCR extraction
            ocr_result = self._run_tesseract_ocr(input_file)
            
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
            return {"success": False, "error": str(e)}
    
    def _run_tesseract_ocr(self, input_file: str) -> Dict:
        """Run Tesseract OCR on the input image"""
        print("  📝 Running Tesseract OCR...")
        
        try:
            # Load and preprocess image
            image = cv2.imread(input_file)
            if image is None:
                raise ValueError(f"Could not load image: {input_file}")
            
            # Preprocess image for better OCR
            processed_image = self._preprocess_image(image)
            
            # Run OCR with detailed output
            ocr_data = pytesseract.image_to_data(
                processed_image, 
                config=f"--psm {self.ocr_config['psm']} --oem {self.ocr_config['oem']} -l {self.ocr_config['lang']}",
                output_type=pytesseract.Output.DICT
            )
            
            # Extract full text
            full_text = pytesseract.image_to_string(
                processed_image,
                config=f"--psm {self.ocr_config['psm']} --oem {self.ocr_config['oem']} -l {self.ocr_config['lang']}"
            )
            
            # Process OCR data into text blocks
            text_blocks = self._process_ocr_data(ocr_data)
            
            # Debug: Check for header text
            print(f"    🔍 Debug: Checking for header text...")
            header_words = []
            for i in range(len(ocr_data['text'])):
                text = ocr_data['text'][i].strip()
                if text and ('Topic' in text or 'Facilitator' in text or 'Discussion' in text):
                    header_words.append({
                        'text': text,
                        'confidence': int(ocr_data['conf'][i]),
                        'bbox': [int(ocr_data['left'][i]), int(ocr_data['top'][i]),
                                int(ocr_data['left'][i]) + int(ocr_data['width'][i]),
                                int(ocr_data['top'][i]) + int(ocr_data['height'][i])]
                    })
            
            if header_words:
                print(f"    ✅ Found {len(header_words)} header-related words:")
                for word in header_words:
                    print(f"       '{word['text']}' (conf: {word['confidence']}) at {word['bbox']}")
            else:
                print(f"    ⚠️  No header text found")
            
            print(f"    ✅ OCR completed: {len(text_blocks)} text blocks, {len(full_text)} characters")
            
            return {
                "full_text": full_text,
                "text_blocks": text_blocks,
                "ocr_config": self.ocr_config,
                "image_size": image.shape[:2][::-1],  # (width, height)
                "processing_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"    ❌ Tesseract OCR failed: {e}")
            return {
                "full_text": "",
                "text_blocks": [],
                "error": str(e)
            }
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for better OCR results"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply denoising
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        # Apply sharpening
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        return sharpened
    
    def _process_ocr_data(self, ocr_data: Dict) -> List[Dict]:
        """Process raw OCR data into structured text blocks"""
        text_blocks = []
        
        # Group words into text blocks based on proximity
        words = []
        for i in range(len(ocr_data['text'])):
            if int(ocr_data['conf'][i]) >= 0:  # Accept all text, even very low confidence
                word_data = {
                    'text': ocr_data['text'][i].strip(),
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
                if word_data['text']:
                    words.append(word_data)
        
        # Group words into blocks
        current_block = None
        block_id = 0
        
        for word in words:
            if current_block is None:
                # Start new block
                current_block = {
                    'text': word['text'],
                    'confidence': word['confidence'],
                    'bbox': word['bbox'].copy(),
                    'words': [word],
                    'block_id': block_id
                }
                block_id += 1
            else:
                # Check if word belongs to current block (same line and close horizontally)
                word_bbox = word['bbox']
                block_bbox = current_block['bbox']
                
                # Check if word is on same line (y-coordinate overlap)
                y_overlap = (word_bbox[1] < block_bbox[3] and word_bbox[3] > block_bbox[1])
                
                # Check if word is close horizontally (within reasonable distance)
                x_distance = word_bbox[0] - block_bbox[2]
                close_horizontally = x_distance < 100  # Increased to 100 pixels to capture spread-out header text
                
                if y_overlap and close_horizontally:
                    # Add word to current block
                    current_block['text'] += ' ' + word['text']
                    current_block['confidence'] = min(current_block['confidence'], word['confidence'])
                    current_block['bbox'][2] = max(current_block['bbox'][2], word_bbox[2])
                    current_block['bbox'][3] = max(current_block['bbox'][3], word_bbox[3])
                    current_block['words'].append(word)
                else:
                    # Save current block and start new one
                    text_blocks.append(current_block)
                    current_block = {
                        'text': word['text'],
                        'confidence': word['confidence'],
                        'bbox': word['bbox'].copy(),
                        'words': [word],
                        'block_id': block_id
                    }
                    block_id += 1
        
        # Add final block
        if current_block is not None:
            text_blocks.append(current_block)
        
        return text_blocks
    
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
        """Draw a single text block"""
        x1, y1, x2, y2 = block['bbox']
        text = block['text']
        confidence = block['confidence']
        
        # Convert to integers
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Choose color based on confidence
        if confidence > 80:
            color = (0, 255, 0)  # Green for high confidence
        elif confidence > 60:
            color = (255, 255, 0)  # Yellow for medium confidence
        else:
            color = (255, 0, 0)  # Red for low confidence
        
        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=1)
        
        # Draw text (truncated if too long)
        display_text = text[:30] + "..." if len(text) > 30 else text
        draw.text((x1+2, y1+2), display_text, fill=color, font=font)
        
        # Draw confidence
        conf_text = f"{confidence}%"
        draw.text((x1+2, y2-12), conf_text, fill=color, font=small_font)
    
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
