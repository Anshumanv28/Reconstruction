"""
OCR Pipeline for Document Reconstruction

This script creates a comprehensive OCR pipeline that:
1. Extracts text from document images using Tesseract OCR
2. Processes the text for better quality
3. Integrates with the reconstruction system
4. Provides detailed OCR results and analysis
"""

import os
import json
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from typing import Dict, List, Tuple, Optional
import re
import subprocess
import sys


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)

# Add system Python path to find pytesseract
sys.path.append(r'C:\Users\anshu\AppData\Local\Programs\Python\Python311\Lib\site-packages')

# Try to import Tesseract
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    pytesseract = None
    print("Warning: pytesseract not available. Install with: pip install pytesseract")

# Import font analyzer
# Font analyzer disabled for now
FONT_ANALYZER_AVAILABLE = False
print("Running basic OCR without font analysis")

class OCRPipeline:
    def __init__(self, tesseract_path: Optional[str] = None):
        """
        Initialize the OCR pipeline.
        
        Args:
            tesseract_path: Path to tesseract executable (if not in PATH)
        """
        self.tesseract_path = tesseract_path
        self.input_dir = "../pipe_input"
        self.output_dir = "../intermediate_outputs/ocr_outputs"
        
        # Create directories if they don't exist
        os.makedirs(self.input_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # OCR configuration
        self.ocr_config = {
            'psm': 6,  # Assume a single uniform block of text
            'oem': 3,  # Default OCR Engine Mode
            'lang': 'eng',  # English language
        }
        
        # Image preprocessing settings
        self.preprocessing = {
            'resize_factor': 2.0,  # Scale up for better OCR
            'denoise': True,
            'enhance_contrast': True,
            'enhance_sharpness': True,
            'binarize': False,  # Let Tesseract handle binarization
        }
        
        # Text processing settings
        self.text_processing = {
            'remove_extra_whitespace': True,
            'remove_special_chars': False,
            'normalize_unicode': True,
            'min_confidence': 30,  # Minimum confidence threshold
        }
    
    def check_tesseract_installation(self) -> bool:
        """Check if Tesseract is properly installed."""
        if not TESSERACT_AVAILABLE or pytesseract is None:
            print("Tesseract not found: pytesseract module not available")
            print("Please install Tesseract OCR:")
            print("Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki")
            print("Linux: sudo apt-get install tesseract-ocr")
            print("macOS: brew install tesseract")
            return False
            
        try:
            if self.tesseract_path:
                pytesseract.pytesseract.tesseract_cmd = self.tesseract_path
            
            # Test Tesseract installation
            version = pytesseract.get_tesseract_version()
            print(f"Tesseract version: {version}")
            return True
        except Exception as e:
            print(f"Tesseract not found: {e}")
            print("Please install Tesseract OCR:")
            print("Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki")
            print("Linux: sudo apt-get install tesseract-ocr")
            print("macOS: brew install tesseract")
            return False
    
    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """Preprocess image for better OCR results."""
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize for better OCR
        if self.preprocessing['resize_factor'] != 1.0:
            width, height = image.size
            new_width = int(width * self.preprocessing['resize_factor'])
            new_height = int(height * self.preprocessing['resize_factor'])
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Enhance contrast
        if self.preprocessing['enhance_contrast']:
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.5)
        
        # Enhance sharpness
        if self.preprocessing['enhance_sharpness']:
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(2.0)
        
        # Denoise
        if self.preprocessing['denoise']:
            # Convert to numpy array for OpenCV processing
            img_array = np.array(image)
            img_array = cv2.fastNlMeansDenoisingColored(img_array, None, 10, 10, 7, 21)
            image = Image.fromarray(img_array)
        
        return image
    
    def extract_text_from_image(self, image_path: str) -> Dict:
        """Extract text from a single image using OCR."""
        if not TESSERACT_AVAILABLE:
            return {
                'success': False,
                'error': 'Tesseract not available',
                'text': '',
                'confidence': 0
            }
        
        try:
            # Load image
            image = Image.open(image_path)
            original_size = image.size
            
            # Preprocess image
            processed_image = self.preprocess_image(image)
            
            # Extract text with confidence scores
            data = pytesseract.image_to_data(
                processed_image, 
                config=f"--psm {self.ocr_config['psm']} --oem {self.ocr_config['oem']}",
                output_type=pytesseract.Output.DICT
            )
            
            # Process OCR results
            text_blocks = []
            full_text = ""
            total_confidence = 0
            valid_blocks = 0
            
            for i in range(len(data['text'])):
                text = data['text'][i].strip()
                confidence = int(data['conf'][i])
                
                if text and confidence > self.text_processing['min_confidence']:
                    # Get bounding box (scale back to original size)
                    x = int(data['left'][i] / self.preprocessing['resize_factor'])
                    y = int(data['top'][i] / self.preprocessing['resize_factor'])
                    w = int(data['width'][i] / self.preprocessing['resize_factor'])
                    h = int(data['height'][i] / self.preprocessing['resize_factor'])
                    
                    text_block = {
                        'text': text,
                        'confidence': confidence,
                        'bbox': [x, y, x + w, y + h],
                        'word_id': data['word_num'][i],
                        'line_id': data['line_num'][i],
                        'block_id': data['block_num'][i]
                    }
                    
                    text_blocks.append(text_block)
                    full_text += text + " "
                    total_confidence += confidence
                    valid_blocks += 1
            
            # Calculate average confidence
            avg_confidence = total_confidence / valid_blocks if valid_blocks > 0 else 0
            
            # Process full text
            processed_text = self.process_text(full_text.strip())
            
            return {
                'success': True,
                'text': processed_text,
                'raw_text': full_text.strip(),
                'confidence': avg_confidence,
                'text_blocks': text_blocks,
                'total_blocks': len(text_blocks),
                'image_size': original_size,
                'processed_image_size': processed_image.size
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'text': '',
                'confidence': 0
            }
    
    def process_text(self, text: str) -> str:
        """Process and clean extracted text."""
        if not text:
            return ""
        
        # Remove extra whitespace
        if self.text_processing['remove_extra_whitespace']:
            text = re.sub(r'\s+', ' ', text.strip())
        
        # Normalize unicode
        if self.text_processing['normalize_unicode']:
            text = text.encode('utf-8').decode('utf-8')
        
        # Remove special characters if requested
        if self.text_processing['remove_special_chars']:
            text = re.sub(r'[^\w\s\.\,\:\;\-\(\)\[\]\/]', '', text)
        
        return text
    
    def extract_text_from_region(self, image_path: str, bbox: List[float]) -> Dict:
        """Extract text from a specific region of an image."""
        try:
            # Load and crop image
            image = Image.open(image_path)
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            cropped_image = image.crop((x1, y1, x2, y2))
            
            # Save cropped image temporarily
            temp_path = os.path.join(self.output_dir, "temp_crop.png")
            cropped_image.save(temp_path)
            
            # Extract text from cropped region
            result = self.extract_text_from_image(temp_path)
            
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            # Adjust bounding boxes to original image coordinates
            if result['success'] and result['text_blocks']:
                for block in result['text_blocks']:
                    block['bbox'] = [
                        block['bbox'][0] + x1,
                        block['bbox'][1] + y1,
                        block['bbox'][2] + x1,
                        block['bbox'][3] + y1
                    ]
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'text': '',
                'confidence': 0
            }
    
    def process_document(self, image_path: str) -> Dict:
        """Process a complete document and extract all text."""
        print(f"Processing document: {image_path}")
        
        # Extract text from entire image
        full_result = self.extract_text_from_image(image_path)
        
        # Create comprehensive result
        result = {
            'document_path': image_path,
            'processing_timestamp': str(pd.Timestamp.now()) if 'pd' in globals() else "N/A",
            'full_document_ocr': full_result,
            'ocr_config': self.ocr_config,
            'preprocessing_config': self.preprocessing,
            'text_processing_config': self.text_processing
        }
        
        return result
    
    def save_ocr_results(self, results: Dict, output_filename: str, image_path: str = None) -> str:
        """Save OCR results to JSON file with font analysis."""
        # Enhance OCR data with font analysis if available
        # Font analysis disabled for now
        
        output_path = os.path.join(self.output_dir, output_filename)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        
        print(f"OCR results saved to: {output_path}")
        return output_path
    
    def create_text_summary(self, results: Dict) -> str:
        """Create a human-readable text summary of OCR results."""
        if not results['full_document_ocr']['success']:
            return f"OCR failed: {results['full_document_ocr']['error']}"
        
        ocr_data = results['full_document_ocr']
        
        summary = f"""
OCR Results Summary
==================

Document: {results['document_path']}
Processing Time: {results['processing_timestamp']}

Text Extraction:
- Success: {ocr_data['success']}
- Total Text Length: {len(ocr_data['text'])} characters
- Average Confidence: {ocr_data['confidence']:.1f}%
- Text Blocks Found: {ocr_data['total_blocks']}

Image Processing:
- Original Size: {ocr_data['image_size']}
- Processed Size: {ocr_data['processed_image_size']}
- Resize Factor: {self.preprocessing['resize_factor']}

Extracted Text:
{ocr_data['text'][:500]}{'...' if len(ocr_data['text']) > 500 else ''}

Configuration:
- PSM Mode: {self.ocr_config['psm']}
- OEM Mode: {self.ocr_config['oem']}
- Language: {self.ocr_config['lang']}
- Min Confidence: {self.text_processing['min_confidence']}%
        """
        
        return summary.strip()
    
    def batch_process_images(self, image_directory: str = None) -> List[Dict]:
        """Process all images in a directory."""
        if image_directory is None:
            image_directory = self.input_dir
        
        # Find all image files
        image_extensions = ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']
        image_files = []
        
        for file in os.listdir(image_directory):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(image_directory, file))
        
        if not image_files:
            print(f"No image files found in {image_directory}")
            return []
        
        print(f"Found {len(image_files)} image files to process")
        
        # Process each image
        results = []
        for image_path in image_files:
            result = self.process_document(image_path)
            results.append(result)
            
            # Save individual results
            filename = os.path.basename(image_path)
            base_name = os.path.splitext(filename)[0]
            self.save_ocr_results(result, f"{base_name}_ocr_results.json", image_path)
            
            # Create and save text summary
            summary = self.create_text_summary(result)
            summary_path = os.path.join(self.output_dir, f"{base_name}_ocr_summary.txt")
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(summary)
            print(f"Text summary saved to: {summary_path}")
        
        # Save batch results
        batch_results = {
            'batch_processing_info': {
                'total_images': len(image_files),
                'processed_images': len(results),
                'processing_timestamp': str(pd.Timestamp.now()) if 'pd' in globals() else "N/A"
            },
            'individual_results': results
        }
        
        self.save_ocr_results(batch_results, "batch_ocr_results.json")
        
        return results


def main():
    """Example usage of the OCR pipeline."""
    # Initialize OCR pipeline
    ocr_pipeline = OCRPipeline()
    
    # Check Tesseract installation
    if not ocr_pipeline.check_tesseract_installation():
        print("Please install Tesseract OCR before running this script.")
        return
    
    print("OCR Pipeline initialized successfully!")
    print(f"Input directory: {ocr_pipeline.input_dir}")
    print(f"Output directory: {ocr_pipeline.output_dir}")
    
    # Process all images in input directory
    results = ocr_pipeline.batch_process_images()
    
    if results:
        print(f"\nProcessed {len(results)} images successfully!")
        print("Check the output directory for detailed results.")
    else:
        print("No images were processed. Please add image files to the input directory.")


if __name__ == "__main__":
    main()
