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
        
        # Alternative PSM modes for better word separation
        self.psm_modes = {
            'uniform_block': 6,    # Assume a single uniform block of text
            'single_line': 7,      # Treat the image as a single text line
            'single_word': 8,      # Treat the image as a single word
            'raw_line': 13,        # Raw line. Treat the image as a single text line
            'sparse_text': 11,     # Sparse text. Find as much text as possible
            'sparse_text_osd': 12  # Sparse text with OSD
        }
        
        # Alternative configurations for better word detection
        self.word_detection_config = {
            'psm': 8,  # Treat the image as a single word
            'oem': 3,  # Default OCR Engine Mode
            'lang': 'eng'
        }
        
        self.line_detection_config = {
            'psm': 7,  # Treat the image as a single text line
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
            
            # Try different PSM modes to find the best word separation
            best_ocr_data = None
            best_score = 0
            best_psm_mode = None
            
            print(f"    🔍 Testing different PSM modes for optimal word detection...")
            
            for mode_name, psm_value in self.psm_modes.items():
                try:
                    config = f"--psm {psm_value} --oem {self.ocr_config['oem']} -l {self.ocr_config['lang']}"
                    ocr_data_test = pytesseract.image_to_data(
                        processed_image, 
                        config=config,
                        output_type=pytesseract.Output.DICT
                    )
                    
                    # Count individual words and calculate quality score
                    words = []
                    single_chars = 0
                    meaningful_words = 0
                    
                    for i in range(len(ocr_data_test['text'])):
                        text = ocr_data_test['text'][i].strip()
                        if text and int(ocr_data_test['conf'][i]) >= -1:  # Accept ALL text
                            words.append(text)
                            if len(text) == 1:
                                single_chars += 1
                            elif len(text) > 2:  # Meaningful words (more than 2 characters)
                                meaningful_words += 1
                    
                    word_count = len(words)
                    
                    # Calculate quality score: prefer modes with more meaningful words and fewer single characters
                    quality_score = meaningful_words * 2 - single_chars * 0.5 + word_count * 0.1
                    
                    print(f"       {mode_name} (PSM {psm_value}): {word_count} words ({meaningful_words} meaningful, {single_chars} single chars) - Score: {quality_score:.1f}")
                    
                    # Keep the mode with the best quality score
                    if quality_score > best_score:
                        best_score = quality_score
                        best_ocr_data = ocr_data_test
                        best_psm_mode = mode_name
                        
                except Exception as e:
                    print(f"       {mode_name} (PSM {psm_value}): Failed - {e}")
                    continue
            
            if best_ocr_data is None:
                raise ValueError("All PSM modes failed")
            
            print(f"    ✅ Best PSM mode: {best_psm_mode} (Score: {best_score:.1f})")
            
            # Try to combine results from multiple PSM modes for maximum word detection
            print(f"    🔍 Combining results from multiple PSM modes for maximum word detection...")
            all_words = {}  # Use dict to avoid duplicates based on text and position
            
            # Collect words from all PSM modes
            for mode_name, psm_value in self.psm_modes.items():
                try:
                    config = f"--psm {psm_value} --oem {self.ocr_config['oem']} -l {self.ocr_config['lang']}"
                    mode_ocr_data = pytesseract.image_to_data(
                        processed_image, 
                        config=config,
                        output_type=pytesseract.Output.DICT
                    )
                    
                    for i in range(len(mode_ocr_data['text'])):
                        text = mode_ocr_data['text'][i].strip()
                        if text and int(mode_ocr_data['conf'][i]) >= -1:  # Accept ALL text
                            # Create a unique key based on text and approximate position
                            x = int(mode_ocr_data['left'][i])
                            y = int(mode_ocr_data['top'][i])
                            key = f"{text}_{x//10}_{y//10}"  # Group by text and approximate position
                            
                            if key not in all_words:
                                all_words[key] = {
                                    'text': text,
                                    'confidence': int(mode_ocr_data['conf'][i]),
                                    'bbox': [x, y, x + int(mode_ocr_data['width'][i]), y + int(mode_ocr_data['height'][i])],
                                    'source_mode': mode_name
                                }
                            else:
                                # Keep the one with higher confidence
                                if int(mode_ocr_data['conf'][i]) > all_words[key]['confidence']:
                                    all_words[key]['confidence'] = int(mode_ocr_data['conf'][i])
                                    all_words[key]['source_mode'] = mode_name
                                    
                except Exception as e:
                    continue
            
            # Convert back to OCR data format for processing
            combined_word_count = len(all_words)
            print(f"    📊 Combined word detection: {combined_word_count} unique words from all PSM modes")
            
            # Use the best single mode as base, but we'll process the combined results
            ocr_data = best_ocr_data
            
            
            # Extract full text using the best PSM mode
            if "(fallback)" in best_psm_mode:
                best_psm_value = self.psm_modes['uniform_block']
            else:
                best_psm_value = self.psm_modes[best_psm_mode]
            
            full_text = pytesseract.image_to_string(
                processed_image,
                config=f"--psm {best_psm_value} --oem {self.ocr_config['oem']} -l {self.ocr_config['lang']}"
            )
            
            # Process OCR data into text blocks
            text_blocks = self._process_ocr_data(ocr_data)
            
            # Debug: Show individual word detection
            print(f"    🔍 Debug: Analyzing individual word detection...")
            individual_words = []
            
            # Use combined results from all PSM modes for maximum word detection
            for word_data in all_words.values():
                individual_words.append({
                    'text': word_data['text'],
                    'confidence': word_data['confidence'],
                    'bbox': word_data['bbox']
                })
            
            # Also add words from the best single mode (in case we missed some)
            for i in range(len(ocr_data['text'])):
                text = ocr_data['text'][i].strip()
                if text and int(ocr_data['conf'][i]) >= -1:  # Accept ALL text
                    # Check if we already have this word from combined results
                    x = int(ocr_data['left'][i])
                    y = int(ocr_data['top'][i])
                    key = f"{text}_{x//10}_{y//10}"
                    
                    if key not in all_words:
                        # Try to split words that contain multiple parts
                        split_words = self._split_compound_word(text, ocr_data, i)
                        individual_words.extend(split_words)
            
            print(f"    📊 Individual words detected: {len(individual_words)}")
            
            # Show first 10 words for debugging
            if individual_words:
                print(f"    🔍 First 10 individual words:")
                for i, word in enumerate(individual_words[:10]):
                    print(f"       {i+1}. '{word['text']}' (conf: {word['confidence']}) at {word['bbox']}")
            
            # Debug: Check for header text
            print(f"    🔍 Debug: Checking for header text...")
            header_words = []
            topic_found = False
            facilitator_found = False
            
            for word in individual_words:
                if 'Topic' in word['text'] or 'Facilitator' in word['text'] or 'Discussion' in word['text']:
                    header_words.append(word)
                    if 'Topic' in word['text']:
                        topic_found = True
                    if 'Facilitator' in word['text']:
                        facilitator_found = True
            
            if header_words:
                print(f"    ✅ Found {len(header_words)} header-related words:")
                for word in header_words:
                    print(f"       '{word['text']}' (conf: {word['confidence']}) at {word['bbox']}")
            
            # Check if we're missing specific header words
            if not topic_found or not facilitator_found:
                missing_words = []
                if not topic_found:
                    missing_words.append("Topic")
                if not facilitator_found:
                    missing_words.append("Facilitator")
                print(f"    ⚠️  Missing header words: {', '.join(missing_words)}")
                
                # Try to find missing header text with lower confidence threshold
                print(f"    🔍 Attempting to find missing header text with lower confidence...")
                low_conf_config = f"--psm {self.psm_modes['uniform_block']} --oem {self.ocr_config['oem']} -l {self.ocr_config['lang']} -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789():|/ "
                low_conf_data = pytesseract.image_to_data(
                    processed_image, 
                    config=low_conf_config,
                    output_type=pytesseract.Output.DICT
                )
                
                # Check if we found header text with lower confidence
                low_conf_words = []
                for i in range(len(low_conf_data['text'])):
                    text = low_conf_data['text'][i].strip()
                    if text and int(low_conf_data['conf'][i]) >= -1:  # Accept ALL confidence levels
                        low_conf_words.append({
                            'text': text,
                            'confidence': int(low_conf_data['conf'][i]),
                            'bbox': [int(low_conf_data['left'][i]), int(low_conf_data['top'][i]),
                                    int(low_conf_data['left'][i]) + int(low_conf_data['width'][i]),
                                    int(low_conf_data['top'][i]) + int(low_conf_data['height'][i])]
                        })
                
                # Check for header text in low confidence results
                header_found_low_conf = any('Topic' in word['text'] or 'Facilitator' in word['text'] 
                                          for word in low_conf_words if word['text'])
                
                if header_found_low_conf:
                    print(f"    ✅ Found header text with low confidence threshold!")
                    # Merge the low confidence header words with the main results
                    for word in low_conf_words:
                        if 'Topic' in word['text'] or 'Facilitator' in word['text']:
                            print(f"       Found: '{word['text']}' (conf: {word['confidence']}) at {word['bbox']}")
                            # Add to individual_words for processing
                            individual_words.append(word)
                else:
                    print(f"    ❌ Still no header text found even with confidence 0")
                    
                    # Try region-specific OCR on the header area
                    print(f"    🔍 Trying region-specific OCR on header area...")
                    try:
                        # Crop the header area (around y=600-650, x=400-1000)
                        header_crop = processed_image[600:650, 400:1000]
                        
                        # Try different preprocessing on the cropped region
                        # 1. Increase contrast
                        header_enhanced = cv2.convertScaleAbs(header_crop, alpha=2.0, beta=30)
                        
                        # 2. Try OCR on the enhanced header region
                        header_ocr_data = pytesseract.image_to_data(
                            header_enhanced,
                            config=f"--psm {self.psm_modes['uniform_block']} --oem {self.ocr_config['oem']} -l {self.ocr_config['lang']}",
                            output_type=pytesseract.Output.DICT
                        )
                        
                        # Check for header text in the cropped region
                        header_region_words = []
                        for i in range(len(header_ocr_data['text'])):
                            text = header_ocr_data['text'][i].strip()
                            if text and int(header_ocr_data['conf'][i]) >= -1:  # Accept ALL confidence levels
                                # Adjust coordinates back to full image
                                adjusted_bbox = [
                                    int(header_ocr_data['left'][i]) + 400,  # Add x offset
                                    int(header_ocr_data['top'][i]) + 600,   # Add y offset
                                    int(header_ocr_data['left'][i]) + int(header_ocr_data['width'][i]) + 400,
                                    int(header_ocr_data['top'][i]) + int(header_ocr_data['height'][i]) + 600
                                ]
                                header_region_words.append({
                                    'text': text,
                                    'confidence': int(header_ocr_data['conf'][i]),
                                    'bbox': adjusted_bbox
                                })
                        
                        # Check if we found header text in the region
                        header_found_region = any('Topic' in word['text'] or 'Facilitator' in word['text'] 
                                                for word in header_region_words if word['text'])
                        
                        if header_found_region:
                            print(f"    ✅ Found header text in region-specific OCR!")
                            for word in header_region_words:
                                if 'Topic' in word['text'] or 'Facilitator' in word['text']:
                                    print(f"       Found: '{word['text']}' (conf: {word['confidence']}) at {word['bbox']}")
                                    individual_words.append(word)
                        else:
                            print(f"    ❌ No header text found in region-specific OCR")
                            
                            # Try with even more aggressive preprocessing
                            print(f"    🔍 Trying aggressive preprocessing on header region...")
                            
                            # Apply multiple preprocessing steps
                            # 1. Convert to grayscale if not already
                            if len(header_crop.shape) == 3:
                                header_gray = cv2.cvtColor(header_crop, cv2.COLOR_BGR2GRAY)
                            else:
                                header_gray = header_crop
                            
                            # 2. Apply adaptive thresholding
                            header_thresh = cv2.adaptiveThreshold(
                                header_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
                            )
                            
                            # 3. Try OCR on thresholded image
                            aggressive_ocr_data = pytesseract.image_to_data(
                                header_thresh,
                                config=f"--psm {self.psm_modes['uniform_block']} --oem {self.ocr_config['oem']} -l {self.ocr_config['lang']}",
                                output_type=pytesseract.Output.DICT
                            )
                            
                            # Check for header text in aggressive preprocessing
                            aggressive_words = []
                            for i in range(len(aggressive_ocr_data['text'])):
                                text = aggressive_ocr_data['text'][i].strip()
                                if text and int(aggressive_ocr_data['conf'][i]) >= -1:  # Accept ALL confidence levels
                                    adjusted_bbox = [
                                        int(aggressive_ocr_data['left'][i]) + 400,
                                        int(aggressive_ocr_data['top'][i]) + 600,
                                        int(aggressive_ocr_data['left'][i]) + int(aggressive_ocr_data['width'][i]) + 400,
                                        int(aggressive_ocr_data['top'][i]) + int(aggressive_ocr_data['height'][i]) + 600
                                    ]
                                    aggressive_words.append({
                                        'text': text,
                                        'confidence': int(aggressive_ocr_data['conf'][i]),
                                        'bbox': adjusted_bbox
                                    })
                            
                            header_found_aggressive = any('Topic' in word['text'] or 'Facilitator' in word['text'] 
                                                         for word in aggressive_words if word['text'])
                            
                            if header_found_aggressive:
                                print(f"    ✅ Found header text with aggressive preprocessing!")
                                for word in aggressive_words:
                                    if 'Topic' in word['text'] or 'Facilitator' in word['text']:
                                        print(f"       Found: '{word['text']}' (conf: {word['confidence']}) at {word['bbox']}")
                                        individual_words.append(word)
                            else:
                                print(f"    ❌ No header text found even with aggressive preprocessing")
                                
                                # Show what text was found in the header region for debugging
                                if header_region_words:
                                    print(f"    📍 Text found in header region:")
                                    for word in header_region_words[:10]:  # Show first 10
                                        print(f"       '{word['text']}' (conf: {word['confidence']}) at {word['bbox']}")
                                
                    except Exception as e:
                        print(f"    ❌ Region-specific OCR failed: {e}")
                
                # Try to find text in the header area (around y=600-650)
                print(f"    🔍 Looking for any text in header area (y=600-650)...")
                header_area_words = []
                for word in individual_words:
                    y_center = (word['bbox'][1] + word['bbox'][3]) / 2
                    if 600 <= y_center <= 650:
                        header_area_words.append(word)
                
                if header_area_words:
                    print(f"    📍 Found {len(header_area_words)} words in header area:")
                    for word in header_area_words:
                        print(f"       '{word['text']}' (conf: {word['confidence']}) at {word['bbox']}")
                else:
                    print(f"    ❌ No words found in header area")
            else:
                print(f"    ✅ All header words found")
            
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
    
    def _split_compound_word(self, text: str, ocr_data: Dict, word_index: int) -> List[Dict]:
        """Split compound words that contain multiple parts separated by special characters"""
        # Define patterns that indicate multiple words
        split_patterns = [
            r'\)//',  # Pattern like "Topic)//Discussion"
            r'\|',    # Pattern like "item|"
            r'//',    # Double slash
            r'\)/',   # Closing parenthesis followed by slash
        ]
        
        # Check if the word contains any split patterns
        needs_splitting = any(re.search(pattern, text) for pattern in split_patterns)
        
        if not needs_splitting:
            # Return the original word
            return [{
                'text': text,
                'confidence': int(ocr_data['conf'][word_index]),
                'bbox': [int(ocr_data['left'][word_index]), int(ocr_data['top'][word_index]),
                        int(ocr_data['left'][word_index]) + int(ocr_data['width'][word_index]),
                        int(ocr_data['top'][word_index]) + int(ocr_data['height'][word_index])]
            }]
        
        # Split the word using the patterns
        parts = [text]
        for pattern in split_patterns:
            new_parts = []
            for part in parts:
                new_parts.extend(re.split(pattern, part))
            parts = new_parts
        
        # Filter out empty parts and create word objects
        split_words = []
        bbox = [int(ocr_data['left'][word_index]), int(ocr_data['top'][word_index]),
                int(ocr_data['left'][word_index]) + int(ocr_data['width'][word_index]),
                int(ocr_data['top'][word_index]) + int(ocr_data['height'][word_index])]
        
        for part in parts:
            part = part.strip()
            if part:
                # Estimate bbox for each part (rough approximation)
                part_bbox = bbox.copy()
                # This is a rough approximation - in practice, you might want more sophisticated bbox calculation
                split_words.append({
                    'text': part,
                    'confidence': int(ocr_data['conf'][word_index]),
                    'bbox': part_bbox
                })
        
        return split_words if split_words else [{
            'text': text,
            'confidence': int(ocr_data['conf'][word_index]),
            'bbox': bbox
        }]
    
    def _process_ocr_data(self, ocr_data: Dict) -> List[Dict]:
        """Process raw OCR data into structured text blocks"""
        text_blocks = []
        
        # Group words into text blocks based on proximity
        words = []
        for i in range(len(ocr_data['text'])):
            if int(ocr_data['conf'][i]) >= -1:  # Accept ALL text, including negative confidence
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
                close_horizontally = x_distance < 30  # Reduced to 30 pixels for better word separation
                
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
