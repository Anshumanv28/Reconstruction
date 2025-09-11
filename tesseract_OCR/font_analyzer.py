#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pytesseract
from typing import Dict, List, Tuple, Any, Optional
import logging
import re
from collections import defaultdict, Counter

_log = logging.getLogger(__name__)


class OCRFontAnalyzer:
    """
    Analyze OCR text to extract font properties, styling, and formatting information.
    This module enhances OCR output with visual styling information.
    """
    
    def __init__(self):
        """Initialize the OCR font analyzer."""
        self.font_size_estimates = {
            'tiny': (6, 10),
            'small': (10, 14),
            'medium': (14, 18),
            'large': (18, 24),
            'xlarge': (24, 36),
            'huge': (36, 100)
        }
        
        self.font_style_patterns = {
            'bold': [r'\b[A-Z]+\b', r'\*\*.*?\*\*', r'__.*?__'],
            'italic': [r'\*.*?\*', r'_.*?_', r'<em>.*?</em>'],
            'underline': [r'<u>.*?</u>', r'<ins>.*?</ins>'],
            'strikethrough': [r'<s>.*?</s>', r'<del>.*?</del>', r'~~.*?~~']
        }
    
    def _convert_image_for_opencv(self, image: Image.Image) -> Image.Image:
        """Convert PIL image to RGB format suitable for OpenCV processing."""
        if image.mode == 'P':
            image = image.convert('RGBA')
        if image.mode == 'RGBA':
            # Create white background for RGBA images
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[-1])
            image = background
        elif image.mode != 'RGB':
            image = image.convert('RGB')
        return image
    
    def analyze_ocr_fonts(self, image: Image.Image, ocr_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze OCR text to extract font and styling information.
        
        Parameters
        ----------
        image : PIL.Image
            The document image
        ocr_data : Dict[str, Any]
            OCR results from Tesseract
            
        Returns
        -------
        Dict[str, Any]
            Enhanced OCR data with font and styling information
        """
        # Get text blocks from OCR data
        text_blocks = self._extract_text_blocks(ocr_data)
        
        # Analyze each text block
        enhanced_blocks = []
        for block in text_blocks:
            enhanced_block = self._analyze_text_block(image, block)
            enhanced_blocks.append(enhanced_block)
        
        # Analyze document-level font patterns
        document_font_analysis = self._analyze_document_fonts(enhanced_blocks)
        
        return {
            'text_blocks': enhanced_blocks,
            'document_font_analysis': document_font_analysis,
            'font_statistics': self._calculate_font_statistics(enhanced_blocks)
        }
    
    def _extract_text_blocks(self, ocr_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract text blocks from OCR data.
        
        Parameters
        ----------
        ocr_data : Dict[str, Any]
            OCR results from Tesseract
            
        Returns
        -------
        List[Dict[str, Any]]
            List of text blocks
        """
        text_blocks = []
        
        # Handle different OCR data structures
        if 'full_document_ocr' in ocr_data:
            # New structure
            blocks = ocr_data['full_document_ocr'].get('text_blocks', [])
        elif 'text_blocks' in ocr_data:
            # Direct structure
            blocks = ocr_data['text_blocks']
        else:
            # Fallback: try to extract from raw data
            blocks = self._extract_blocks_from_raw_data(ocr_data)
        
        for block in blocks:
            if isinstance(block, dict) and 'bbox' in block:
                text_blocks.append(block)
        
        return text_blocks
    
    def _extract_blocks_from_raw_data(self, ocr_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract text blocks from raw OCR data structure.
        
        Parameters
        ----------
        ocr_data : Dict[str, Any]
            Raw OCR data
            
        Returns
        -------
        List[Dict[str, Any]]
            Extracted text blocks
        """
        blocks = []
        
        # Try to find text information in various possible locations
        for key, value in ocr_data.items():
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict) and 'bbox' in item:
                        blocks.append(item)
            elif isinstance(value, dict) and 'bbox' in value:
                blocks.append(value)
        
        return blocks
    
    def _analyze_text_block(self, image: Image.Image, text_block: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a single text block for font and styling information.
        
        Parameters
        ----------
        image : PIL.Image
            The document image
        text_block : Dict[str, Any]
            Text block information
            
        Returns
        -------
        Dict[str, Any]
            Enhanced text block with font analysis
        """
        # Get text content
        text = text_block.get('text', '')
        bbox = text_block.get('bbox', [])
        
        if not text or len(bbox) < 4:
            return {**text_block, 'font_analysis': {'error': 'Invalid text block'}}
        
        # Crop text region
        x1, y1, x2, y2 = map(int, bbox[:4])
        text_region = image.crop((x1, y1, x2, y2))
        
        # Analyze font properties
        font_analysis = {
            'estimated_font_size': self._estimate_font_size(text_region, text),
            'font_style': self._detect_font_style(text_region, text),
            'text_alignment': self._detect_text_alignment(text_region, text),
            'line_spacing': self._estimate_line_spacing(text_region, text),
            'character_spacing': self._estimate_character_spacing(text_region, text),
            'text_density': self._calculate_text_density(text_region),
            'color_analysis': self._analyze_text_colors(text_region),
            'bold_detection': self._detect_bold_text(text_region, text),
            'italic_detection': self._detect_italic_text(text_region, text),
            'underline_detection': self._detect_underline(text_region, text)
        }
        
        return {
            **text_block,
            'font_analysis': font_analysis
        }
    
    def _estimate_font_size(self, text_region: Image.Image, text: str) -> Dict[str, Any]:
        """
        Estimate font size from text region.
        
        Parameters
        ----------
        text_region : PIL.Image
            Cropped text region
        text : str
            Text content
            
        Returns
        -------
        Dict[str, Any]
            Font size estimation
        """
        # Convert to OpenCV format with proper mode handling
        text_region = self._convert_image_for_opencv(text_region)
        region_cv = cv2.cvtColor(np.array(text_region), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(region_cv, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold to separate text from background
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours (text characters)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {'size': 12, 'confidence': 0.0, 'method': 'fallback'}
        
        # Calculate character heights
        heights = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if h > 5 and w > 3:  # Filter noise
                heights.append(h)
        
        if heights:
            avg_height = np.mean(heights)
            median_height = np.median(heights)
            
            # Estimate font size (rough conversion from pixel height)
            estimated_size = int(avg_height * 0.75)  # Approximate conversion
            
            # Classify font size
            size_category = self._classify_font_size(estimated_size)
            
            return {
                'size': estimated_size,
                'avg_height': float(avg_height),
                'median_height': float(median_height),
                'size_category': size_category,
                'confidence': 0.7,
                'method': 'contour_analysis'
            }
        
        return {'size': 12, 'confidence': 0.0, 'method': 'fallback'}
    
    def _classify_font_size(self, font_size: int) -> str:
        """Classify font size into categories."""
        for category, (min_size, max_size) in self.font_size_estimates.items():
            if min_size <= font_size <= max_size:
                return category
        return 'unknown'
    
    def _detect_font_style(self, text_region: Image.Image, text: str) -> Dict[str, Any]:
        """
        Detect font style (bold, italic, etc.) from text region.
        
        Parameters
        ----------
        text_region : PIL.Image
            Cropped text region
        text : str
            Text content
            
        Returns
        -------
        Dict[str, Any]
            Font style detection results
        """
        # Convert to grayscale
        text_region = self._convert_image_for_opencv(text_region)
        gray = cv2.cvtColor(np.array(text_region), cv2.COLOR_RGB2GRAY)
        
        # Analyze stroke width (bold detection)
        stroke_width = self._estimate_stroke_width(gray)
        
        # Analyze text slant (italic detection)
        text_slant = self._estimate_text_slant(gray)
        
        # Pattern-based style detection
        pattern_styles = self._detect_style_patterns(text)
        
        return {
            'stroke_width': stroke_width,
            'text_slant': text_slant,
            'pattern_styles': pattern_styles,
            'is_bold': stroke_width > 2.0,
            'is_italic': abs(text_slant) > 5.0,
            'confidence': 0.6
        }
    
    def _estimate_stroke_width(self, gray_image: np.ndarray) -> float:
        """Estimate average stroke width in the text region."""
        # Apply morphological operations to estimate stroke width
        kernel_sizes = [1, 2, 3, 4, 5]
        
        for kernel_size in kernel_sizes:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            eroded = cv2.erode(gray_image, kernel, iterations=1)
            
            # If erosion removes significant content, this is likely the stroke width
            diff = cv2.absdiff(gray_image, eroded)
            if np.sum(diff) < np.sum(gray_image) * 0.1:
                return float(kernel_size)
        
        return 2.0  # Default stroke width
    
    def _estimate_text_slant(self, gray_image: np.ndarray) -> float:
        """Estimate text slant angle."""
        # Use Hough line transform to detect text lines
        edges = cv2.Canny(gray_image, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, 
                               minLineLength=20, maxLineGap=10)
        
        if lines is not None:
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                angles.append(angle)
            
            if angles:
                # Return median angle
                return float(np.median(angles))
        
        return 0.0  # No slant detected
    
    def _detect_style_patterns(self, text: str) -> Dict[str, bool]:
        """Detect style patterns in text using regex."""
        patterns = {}
        
        for style, pattern_list in self.font_style_patterns.items():
            patterns[style] = any(re.search(pattern, text) for pattern in pattern_list)
        
        return patterns
    
    def _detect_text_alignment(self, text_region: Image.Image, text: str) -> Dict[str, Any]:
        """
        Detect text alignment (left, center, right, justify).
        
        Parameters
        ----------
        text_region : PIL.Image
            Cropped text region
        text : str
            Text content
            
        Returns
        -------
        Dict[str, Any]
            Text alignment information
        """
        # Convert to grayscale
        text_region = self._convert_image_for_opencv(text_region)
        gray = cv2.cvtColor(np.array(text_region), cv2.COLOR_RGB2GRAY)
        
        # Apply threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find text lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
        
        # Analyze line positions
        contours, _ = cv2.findContours(lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {'alignment': 'unknown', 'confidence': 0.0}
        
        # Get line start positions
        line_starts = []
        line_ends = []
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            line_starts.append(x)
            line_ends.append(x + w)
        
        if not line_starts:
            return {'alignment': 'unknown', 'confidence': 0.0}
        
        # Analyze alignment patterns
        start_variance = np.var(line_starts)
        end_variance = np.var(line_ends)
        
        region_width = text_region.width
        
        if start_variance < 10:  # Lines start at similar positions
            if np.mean(line_starts) < region_width * 0.1:
                alignment = 'left'
            elif np.mean(line_starts) > region_width * 0.9:
                alignment = 'right'
            else:
                alignment = 'center'
        elif end_variance < 10:  # Lines end at similar positions
            alignment = 'right'
        else:
            alignment = 'justify'
        
        confidence = 0.7 if start_variance < 50 or end_variance < 50 else 0.3
        
        return {
            'alignment': alignment,
            'confidence': confidence,
            'start_variance': float(start_variance),
            'end_variance': float(end_variance)
        }
    
    def _estimate_line_spacing(self, text_region: Image.Image, text: str) -> Dict[str, Any]:
        """Estimate line spacing in the text region."""
        # Convert to grayscale
        text_region = self._convert_image_for_opencv(text_region)
        gray = cv2.cvtColor(np.array(text_region), cv2.COLOR_RGB2GRAY)
        
        # Apply threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find horizontal lines (text lines)
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
        
        # Get line positions
        contours, _ = cv2.findContours(lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) < 2:
            return {'spacing': 0, 'confidence': 0.0}
        
        # Calculate line positions
        line_positions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            line_positions.append(y + h // 2)  # Center of line
        
        line_positions.sort()
        
        # Calculate spacing between lines
        spacings = []
        for i in range(1, len(line_positions)):
            spacing = line_positions[i] - line_positions[i-1]
            spacings.append(spacing)
        
        if spacings:
            avg_spacing = np.mean(spacings)
            return {
                'spacing': float(avg_spacing),
                'line_count': len(line_positions),
                'confidence': 0.8
            }
        
        return {'spacing': 0, 'confidence': 0.0}
    
    def _estimate_character_spacing(self, text_region: Image.Image, text: str) -> Dict[str, Any]:
        """Estimate character spacing in the text."""
        # Convert to grayscale
        text_region = self._convert_image_for_opencv(text_region)
        gray = cv2.cvtColor(np.array(text_region), cv2.COLOR_RGB2GRAY)
        
        # Apply threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find character contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) < 2:
            return {'spacing': 0, 'confidence': 0.0}
        
        # Get character positions
        char_positions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 3 and h > 5:  # Filter noise
                char_positions.append(x + w // 2)  # Center of character
        
        char_positions.sort()
        
        # Calculate spacing between characters
        spacings = []
        for i in range(1, len(char_positions)):
            spacing = char_positions[i] - char_positions[i-1]
            if spacing < 50:  # Filter large gaps (likely word boundaries)
                spacings.append(spacing)
        
        if spacings:
            avg_spacing = np.mean(spacings)
            return {
                'spacing': float(avg_spacing),
                'character_count': len(char_positions),
                'confidence': 0.7
            }
        
        return {'spacing': 0, 'confidence': 0.0}
    
    def _calculate_text_density(self, text_region: Image.Image) -> Dict[str, Any]:
        """Calculate text density in the region."""
        # Convert to grayscale
        text_region = self._convert_image_for_opencv(text_region)
        gray = cv2.cvtColor(np.array(text_region), cv2.COLOR_RGB2GRAY)
        
        # Apply threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Calculate text pixel density
        text_pixels = np.sum(binary == 0)  # Assuming text is black
        total_pixels = binary.size
        density = text_pixels / total_pixels * 100
        
        return {
            'density': float(density),
            'text_pixels': int(text_pixels),
            'total_pixels': int(total_pixels)
        }
    
    def _analyze_text_colors(self, text_region: Image.Image) -> Dict[str, Any]:
        """Analyze text colors in the region."""
        # Convert to numpy array
        region_array = np.array(text_region)
        
        # Ensure we have the right number of channels
        if len(region_array.shape) == 3:
            pixels = region_array.reshape(-1, region_array.shape[2])
        else:
            # Grayscale image, convert to RGB
            pixels = np.stack([region_array.flatten()] * 3, axis=1)
        
        # Use K-means to find dominant colors
        from sklearn.cluster import KMeans
        
        # Sample pixels for efficiency
        if len(pixels) > 5000:
            indices = np.random.choice(len(pixels), 5000, replace=False)
            pixels = pixels[indices]
        
        if len(pixels) < 2:
            return {'dominant_colors': [], 'text_color': '#000000'}
        
        kmeans = KMeans(n_clusters=min(3, len(pixels)), random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        # Get cluster centers
        colors = kmeans.cluster_centers_.astype(int)
        labels = kmeans.labels_
        
        # Count pixels in each cluster
        color_counts = np.bincount(labels)
        color_percentages = color_counts / len(labels) * 100
        
        # Sort by frequency
        sorted_indices = np.argsort(color_percentages)[::-1]
        
        dominant_colors = []
        for idx in sorted_indices:
            rgb_color = colors[idx]
            hex_color = f"#{rgb_color[0]:02x}{rgb_color[1]:02x}{rgb_color[2]:02x}"
            dominant_colors.append({
                'rgb': rgb_color.tolist(),
                'hex': hex_color,
                'percentage': float(color_percentages[idx])
            })
        
        # Assume the darkest color is the text color
        text_color = dominant_colors[0] if dominant_colors else {'hex': '#000000'}
        
        return {
            'dominant_colors': dominant_colors,
            'text_color': text_color
        }
    
    def _detect_bold_text(self, text_region: Image.Image, text: str) -> Dict[str, Any]:
        """Detect bold text using stroke width analysis."""
        text_region = self._convert_image_for_opencv(text_region)
        gray = cv2.cvtColor(np.array(text_region), cv2.COLOR_RGB2GRAY)
        
        stroke_width = self._estimate_stroke_width(gray)
        
        return {
            'is_bold': stroke_width > 2.5,
            'stroke_width': float(stroke_width),
            'confidence': 0.6
        }
    
    def _detect_italic_text(self, text_region: Image.Image, text: str) -> Dict[str, Any]:
        """Detect italic text using slant analysis."""
        text_region = self._convert_image_for_opencv(text_region)
        gray = cv2.cvtColor(np.array(text_region), cv2.COLOR_RGB2GRAY)
        
        slant = self._estimate_text_slant(gray)
        
        return {
            'is_italic': abs(slant) > 5.0,
            'slant_angle': float(slant),
            'confidence': 0.5
        }
    
    def _detect_underline(self, text_region: Image.Image, text: str) -> Dict[str, Any]:
        """Detect underlined text."""
        text_region = self._convert_image_for_opencv(text_region)
        gray = cv2.cvtColor(np.array(text_region), cv2.COLOR_RGB2GRAY)
        
        # Look for horizontal lines at the bottom of text
        height, width = gray.shape
        
        # Check bottom 20% of the region for horizontal lines
        bottom_region = gray[int(height * 0.8):, :]
        
        # Apply edge detection
        edges = cv2.Canny(bottom_region, 50, 150)
        
        # Look for horizontal lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
        horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)
        
        # Count horizontal line pixels
        line_pixels = np.sum(horizontal_lines > 0)
        total_pixels = horizontal_lines.size
        
        has_underline = line_pixels > total_pixels * 0.05  # 5% threshold
        
        return {
            'has_underline': has_underline,
            'line_pixels': int(line_pixels),
            'confidence': 0.4
        }
    
    def _analyze_document_fonts(self, enhanced_blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze document-level font patterns.
        
        Parameters
        ----------
        enhanced_blocks : List[Dict[str, Any]]
            Enhanced text blocks with font analysis
            
        Returns
        -------
        Dict[str, Any]
            Document-level font analysis
        """
        if not enhanced_blocks:
            return {'error': 'No text blocks to analyze'}
        
        # Collect font statistics
        font_sizes = []
        font_styles = []
        alignments = []
        
        for block in enhanced_blocks:
            font_analysis = block.get('font_analysis', {})
            
            if 'estimated_font_size' in font_analysis:
                size_info = font_analysis['estimated_font_size']
                if 'size' in size_info:
                    font_sizes.append(size_info['size'])
            
            if 'font_style' in font_analysis:
                style_info = font_analysis['font_style']
                if style_info.get('is_bold'):
                    font_styles.append('bold')
                if style_info.get('is_italic'):
                    font_styles.append('italic')
            
            if 'text_alignment' in font_analysis:
                align_info = font_analysis['text_alignment']
                if 'alignment' in align_info:
                    alignments.append(align_info['alignment'])
        
        # Calculate statistics
        font_size_stats = {
            'min': float(np.min(font_sizes)) if font_sizes else 0,
            'max': float(np.max(font_sizes)) if font_sizes else 0,
            'mean': float(np.mean(font_sizes)) if font_sizes else 0,
            'median': float(np.median(font_sizes)) if font_sizes else 0
        }
        
        style_counts = Counter(font_styles)
        alignment_counts = Counter(alignments)
        
        return {
            'font_size_statistics': font_size_stats,
            'style_distribution': dict(style_counts),
            'alignment_distribution': dict(alignment_counts),
            'total_blocks': len(enhanced_blocks),
            'analyzed_blocks': len([b for b in enhanced_blocks if 'font_analysis' in b])
        }
    
    def _calculate_font_statistics(self, enhanced_blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate comprehensive font statistics."""
        if not enhanced_blocks:
            return {}
        
        # Collect all font sizes
        all_sizes = []
        for block in enhanced_blocks:
            font_analysis = block.get('font_analysis', {})
            size_info = font_analysis.get('estimated_font_size', {})
            if 'size' in size_info:
                all_sizes.append(size_info['size'])
        
        if not all_sizes:
            return {}
        
        # Calculate statistics
        return {
            'font_size_range': {
                'min': float(np.min(all_sizes)),
                'max': float(np.max(all_sizes)),
                'mean': float(np.mean(all_sizes)),
                'std': float(np.std(all_sizes)),
                'median': float(np.median(all_sizes))
            },
            'size_distribution': dict(Counter([self._classify_font_size(size) for size in all_sizes])),
            'total_text_blocks': len(enhanced_blocks),
            'blocks_with_font_info': len(all_sizes)
        }


def analyze_ocr_fonts(image_path: str, ocr_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze OCR fonts and styling for a document.
    
    Parameters
    ----------
    image_path : str
        Path to the document image
    ocr_data : Dict[str, Any]
        OCR results from Tesseract
        
    Returns
    -------
    Dict[str, Any]
        Enhanced OCR data with font analysis
    """
    # Load image
    image = Image.open(image_path)
    
    # Initialize analyzer
    analyzer = OCRFontAnalyzer()
    
    # Analyze fonts
    enhanced_ocr = analyzer.analyze_ocr_fonts(image, ocr_data)
    
    return enhanced_ocr
