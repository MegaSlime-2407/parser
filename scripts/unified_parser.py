#!/usr/bin/env python3
"""
Smart Document Parser - Reads documents like a human would
Extracts text, images, tables, and charts with precise locations
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

# Computer vision and image processing
import cv2
import numpy as np
import pytesseract
from PIL import Image
import sys

# Try to use GPU if available (makes things faster)
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

# Import our helper modules
sys.path.insert(0, str(Path(__file__).parent))
from document_parser import DocumentParser
from graphics_handler import GraphicsHandler

# Set up logging so we can see what's happening
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class UnifiedParser:
    """
    A smart document reader that understands what it sees
    Like having a human read a document and tell you where everything is
    """
    
    def __init__(self, tesseract_path: Optional[str] = None, use_gpu: bool = True):
        """
        Set up our document reader
        
        Args:
            tesseract_path: Where to find the text reading engine
            use_gpu: Whether to use the graphics card for speed (default: True)
        """
        # Create our helpers - one for text, one for pictures
        self.parser = DocumentParser(tesseract_path=tesseract_path)
        self.graphics_handler = GraphicsHandler(tesseract_path=tesseract_path)
        
        # Tell the text reader where to find its engine
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        
        # Figure out if we can use the graphics card to go faster
        self.device = self._setup_device(use_gpu)
        
    def _setup_device(self, use_gpu: bool = True) -> str:
        """
        Figure out the best way to process images (fast graphics card or regular CPU)
        
        Args:
            use_gpu: Whether we should try to use the graphics card
            
        Returns:
            What device we'll use ('cuda' for graphics card, 'cpu' for regular processing)
        """
        # If we don't have the right software, we can't use the graphics card
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available - GPU acceleration disabled")
            return 'cpu'
        
        # If the user said no to graphics card, respect that
        if not use_gpu:
            logger.info("GPU disabled by user - using CPU")
            return 'cpu'
        
        # Check if we have a graphics card that can help us
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            logger.info(f"Found graphics card: {device_name}")
            logger.info(f"CUDA version: {torch.version.cuda}")
            return 'cuda'
        else:
            logger.info("No graphics card found - using regular CPU")
            return 'cpu'
    
    def get_device_info(self) -> Dict[str, Any]:
        """
        Tell us what kind of computer we're running on
        
        Returns:
            Information about our processing power
        """
        info = {
            'device': self.device,
            'torch_available': TORCH_AVAILABLE
        }
        
        # If we have a graphics card, tell us about it
        if TORCH_AVAILABLE and torch.cuda.is_available():
            info['cuda_available'] = True
            info['cuda_version'] = torch.version.cuda
            info['gpu_name'] = torch.cuda.get_device_name(0)
            info['gpu_count'] = torch.cuda.device_count()
            info['gpu_memory'] = {
                'allocated': f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB",
                'reserved': f"{torch.cuda.memory_reserved() / 1024**3:.2f} GB"
            }
        else:
            info['cuda_available'] = False
        
        return info
    
    def parse_with_positions(self, file_path: str) -> Dict[str, Any]:
        """
        Read a document and tell us where everything is located
        
        Args:
            file_path: The document we want to read (image or PDF)
            
        Returns:
            Everything we found, with exact locations
        """
        logger.info(f"Reading document: {file_path}")
        
        # Start with an empty result - we'll fill it as we go
        result = {
            'file_path': file_path,
            'blocks': [],
            'summary': {
                'total_blocks': 0,
                'text_blocks': 0,
                'image_blocks': 0,
                'table_blocks': 0,
                'chart_blocks': 0,
                'shape_blocks': 0
            }
        }
        
        try:
            # Figure out what kind of file this is
            file_ext = Path(file_path).suffix.lower()
            
            if file_ext in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
                # It's a picture - read it directly
                result = self._parse_image_with_positions(file_path)
            elif file_ext == '.pdf':
                # It's a PDF - convert to pictures first, then read
                result = self._parse_pdf_with_positions(file_path)
            else:
                raise ValueError(f"Sorry, I don't know how to read {file_ext} files yet")
            
            # Count up what we found
            result['summary'] = self._generate_summary(result['blocks'])
            
            logger.info(f"Done! Found {result['summary']['total_blocks']} things in the document")
            
        except Exception as e:
            logger.error(f"Oops, something went wrong reading {file_path}: {e}")
            result['error'] = str(e)
        
        return result
    
    def _parse_image_with_positions(self, image_path: str) -> Dict[str, Any]:
        """
        Look at a picture and tell us what's in it and where
        
        Args:
            image_path: The picture file to examine
            
        Returns:
            Everything we found in the picture with exact locations
        """
        result = {
            'file_path': image_path,
            'blocks': []
        }
        
        # Load the image so we can look at it
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # First, let's find all the text and where it is
        text_blocks, detected_tables = self._extract_text_blocks_with_positions(image_rgb)
        result['blocks'].extend(text_blocks)
        
        # Add any tables or charts we found in the text
        for element in detected_tables:
            result['blocks'].append(element)
        
        # Now let's look for other things like pictures, charts, and shapes
        visual_elements = self.graphics_handler.detect_visual_elements(image_path)
        
        if 'visual_elements' in visual_elements:
            elements = visual_elements['visual_elements']
            
            # Remember how big the image is so we can filter out weird detections
            img_height, img_width = image.shape[:2]
            
            # Keep track of where text is so we don't double-count things
            text_regions = [(b['x1'], b['y1'], b['x2'], b['y2']) for b in text_blocks]
            
            # Look for charts and graphs
            charts = elements.get('charts', [])
            for chart in charts:
                conf = chart.get('confidence', 0.0)
                # Only trust charts we're pretty sure about
                if conf > 0.8:
                    bbox = chart['bounding_box']
                    # Make sure it's not too big or too small (probably a mistake)
                    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    total_area = img_width * img_height
                    if 0.05 < area / total_area < 0.95:  # Between 5% and 95% of image
                        result['blocks'].append({
                            'type': 'chart',
                            'subtype': chart.get('type', 'unknown'),
                            'x1': bbox[0],
                            'y1': bbox[1],
                            'x2': bbox[2],
                            'y2': bbox[3],
                            'bounding_box': bbox,
                            'confidence': conf,
                            'content': 'chart_detected'
                        })
            
            # Look for tables
            tables = elements.get('tables', [])
            for table in tables:
                conf = table.get('confidence', 0.0)
                # Be a bit more lenient with tables - they're harder to spot
                if conf > 0.3:
                    bbox = table['bounding_box']
                    
                    # Make sure it's not covering the whole page (probably wrong)
                    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    total_area = img_width * img_height
                    coverage = area / total_area
                    
                    # Check if this looks like a real table (not too big, not too small)
                    is_at_origin = bbox[0] <= 10 and bbox[1] <= 10  # Starting at top-left corner
                    is_full_coverage = coverage > 0.95  # Taking up almost the whole page
                    
                    if 0.02 < coverage < 0.98 and not (is_at_origin and is_full_coverage):
                        result['blocks'].append({
                            'type': 'table',
                            'x1': bbox[0],
                            'y1': bbox[1],
                            'x2': bbox[2],
                            'y2': bbox[3],
                            'bounding_box': bbox,
                            'confidence': conf,
                            'content': 'table_detected'
                        })
            
            # Look for pictures and photos
            images = elements.get('images', [])
            for img in images:
                conf = img.get('confidence', 0.0)
                if conf > 0.7:  # Pretty sure it's a picture
                    bbox = img['bounding_box']
                    result['blocks'].append({
                        'type': 'image',
                        'x1': bbox[0],
                        'y1': bbox[1],
                        'x2': bbox[2],
                        'y2': bbox[3],
                        'bounding_box': bbox,
                        'confidence': conf,
                        'content': 'image_detected'
                    })
            
            # Look for shapes and geometric figures
            shapes = elements.get('shapes', [])
            for shape in shapes:
                conf = shape.get('confidence', 0.0)
                # Only trust shapes we're very sure about
                if conf > 0.8:
                    bbox = shape['bounding_box']
                    # Check if this shape is actually just text (common mistake)
                    overlaps_text = False
                    for tx1, ty1, tx2, ty2 in text_regions:
                        if self._boxes_overlap(bbox, (tx1, ty1, tx2, ty2)):
                            overlaps_text = True
                            break
                    
                    # Only count it as a shape if it's not overlapping with text
                    if not overlaps_text:
                        result['blocks'].append({
                            'type': 'shape',
                            'subtype': shape.get('type', 'unknown'),
                            'x1': bbox[0],
                            'y1': bbox[1],
                            'x2': bbox[2],
                            'y2': bbox[3],
                            'bounding_box': bbox,
                            'confidence': conf,
                            'content': 'shape_detected'
                        })
        
        return result
    
    def _parse_pdf_with_positions(self, pdf_path: str) -> Dict[str, Any]:
        """
        Read a PDF by converting each page to a picture first
        
        Args:
            pdf_path: The PDF file to read
            
        Returns:
            Everything we found in all pages with exact locations
        """
        from pdf2image import convert_from_path
        
        result = {
            'file_path': pdf_path,
            'pages': [],
            'blocks': [],
            'page_images': []  # Keep track of the image files we create
        }
        
        # Create a folder to store the page images
        images_dir = Path('output/images')
        images_dir.mkdir(parents=True, exist_ok=True)
        
        # Get the PDF name without the .pdf part
        pdf_filename = Path(pdf_path).stem
        
        # Convert each PDF page to a high-quality image
        images = convert_from_path(pdf_path, dpi=300)
        
        logger.info(f"Converting {len(images)} pages from PDF to images...")
        
        # Look at each page one by one
        for page_num, image in enumerate(images):
            # Save this page as a picture file
            page_image_path = images_dir / f"{pdf_filename}_page_{page_num + 1}.png"
            image.save(page_image_path, 'PNG')
            
            logger.info(f"Saved page {page_num + 1}: {page_image_path}")
            
            # Remember where we saved this page
            result['page_images'].append(str(page_image_path))
            
            # Now read this page like we would any other picture
            page_result = self._parse_image_with_positions(str(page_image_path))
            
            # Tell each thing we found which page it's on
            for block in page_result['blocks']:
                block['page'] = page_num + 1
            
            # Keep track of this page's results
            result['pages'].append({
                'page_number': page_num + 1,
                'blocks': page_result['blocks'],
                'image_path': str(page_image_path)
            })
            result['blocks'].extend(page_result['blocks'])
        
        logger.info(f"All {len(images)} pages converted and saved to {images_dir}/")
        
        return result
    
    def _boxes_overlap(self, box1: tuple, box2: tuple, threshold: float = 0.5) -> bool:
        """
        Check if two rectangles are overlapping enough to matter
        
        Args:
            box1: First rectangle (x1, y1, x2, y2)
            box2: Second rectangle (x1, y1, x2, y2)
            threshold: How much overlap we need to care (default 0.5 = 50%)
            
        Returns:
            True if they overlap enough to be the same thing
        """
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Find where the rectangles intersect
        x1_inter = max(x1_1, x1_2)
        y1_inter = max(y1_1, y1_2)
        x2_inter = min(x2_1, x2_2)
        y2_inter = min(y2_1, y2_2)
        
        # If there's no intersection, they don't overlap
        if x2_inter <= x1_inter or y2_inter <= y1_inter:
            return False
        
        # Calculate how much they overlap
        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        
        # Calculate the area of the smaller rectangle
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        min_area = min(area1, area2)
        
        # Check if the overlap is significant enough
        overlap_ratio = inter_area / min_area if min_area > 0 else 0
        return overlap_ratio >= threshold
    
    def _extract_text_blocks_with_positions(self, image_rgb: np.ndarray) -> List[Dict[str, Any]]:
        """
        Read all the text in an image and tell us where each piece is
        
        Args:
            image_rgb: The image we want to read
            
        Returns:
            All the text we found with exact locations
        """
        text_blocks = []
        
        try:
            # Use our text reading engine to find all words and where they are
            data = pytesseract.image_to_data(image_rgb, output_type=pytesseract.Output.DICT)
            
            current_block = None
            block_id = 0
            
            # Look at each word the engine found
            for i in range(len(data['text'])):
                conf = int(data['conf'][i])
                
                # Only trust words we're pretty sure about
                if conf > 30:
                    x = data['left'][i]
                    y = data['top'][i]
                    w = data['width'][i]
                    h = data['height'][i]
                    text = data['text'][i].strip()
                    
                    if text and len(text) > 0:
                        # Check if this word belongs with the previous one (same line)
                        if current_block is None:
                            # Start a new text block
                            x2 = x + w
                            y2 = y + h
                            current_block = {
                                'block_id': block_id,
                                'type': 'text',
                                'x1': x,
                                'y1': y,
                                'x2': x2,
                                'y2': y2,
                                'bounding_box': [x, y, x2, y2],
                                'text': text,
                                'confidence': conf / 100.0
                            }
                            block_id += 1
                        else:
                            # Check if this word is close to the current block (same line)
                            y_diff = y - current_block['y2']
                            
                            if y_diff < h * 2:  # Close enough to be the same line
                                # Add this word to the current block
                                current_block['text'] += ' ' + text
                                current_block['x2'] = max(current_block['x2'], x + w)
                                current_block['y2'] = max(current_block['y2'], y + h)
                                current_block['bounding_box'][2] = max(current_block['bounding_box'][2], x + w)
                                current_block['bounding_box'][3] = max(current_block['bounding_box'][3], y + h)
                            else:
                                # This word is on a new line, so finish the current block
                                text_blocks.append(current_block)
                                x2 = x + w
                                y2 = y + h
                                current_block = {
                                    'block_id': block_id,
                                    'type': 'text',
                                    'x1': x,
                                    'y1': y,
                                    'x2': x2,
                                    'y2': y2,
                                    'bounding_box': [x, y, x2, y2],
                                    'text': text,
                                    'confidence': conf / 100.0
                                }
                                block_id += 1
            
            # Don't forget the last block
            if current_block is not None:
                text_blocks.append(current_block)
            
            # Clean up the text (fix common OCR mistakes)
            for block in text_blocks:
                block['text'] = self._postprocess_text(block['text'])
            
            # Look for tables hidden in the text
            detected_tables = self._detect_tables_from_text_blocks(text_blocks, image_rgb)
            
        except Exception as e:
            logger.error(f"Oops, couldn't read the text: {e}")
            detected_tables = []
        
        return text_blocks, detected_tables
    
    def _classify_table_vs_chart(self, region: List[List[Dict]], text_content: List[str]) -> str:
        """
        Figure out if this looks more like a table or a chart
        
        Args:
            region: The text blocks we're looking at
            text_content: The actual text from those blocks
            
        Returns:
            'table' or 'chart' based on what it looks like
        """
        all_text = ' '.join(text_content).lower()
        
        # Look for signs that this is a chart
        has_chart_text = any(keyword in all_text for keyword in ['column', 'row'] if 'header' not in all_text)
        has_equals = '=' in all_text
        has_numbers = any(char.isdigit() for char in all_text)
        
        # If it has "Column X" and "Row Y" and equals signs, it's probably a chart
        if has_chart_text and has_equals and not any(word in all_text for word in ['total', 'sum', 'header']):
            return 'chart'
        
        # If it has lots of structured text, it might be a table
        if len(text_content) >= 2 and any(len(t.split()) >= 3 for t in text_content):
            # Check if text looks like column headers or data rows
            word_counts = [len(t.split()) for t in text_content]
            if max(word_counts) >= 3:  # Multiple columns
                return 'table'
        
        # If we're not sure, guess it's a table
        return 'table'
    
    def _detect_tables_from_text_blocks(self, text_blocks: List[Dict], image_rgb: np.ndarray) -> List[Dict]:
        """
        Look for tables hidden in the text (like numbered lists or aligned columns)
        
        Args:
            text_blocks: All the text we found
            image_rgb: The image we're looking at
            
        Returns:
            Any tables we spotted in the text
        """
        tables = []
        
        try:
            # Remember how big the image is
            h, w = image_rgb.shape[:2]
            
            # Sort text blocks from top to bottom
            sorted_blocks = sorted(text_blocks, key=lambda b: b['y1'])
            
            # Look for text that's arranged in rows (like a table)
            rows = []
            current_row = []
            
            for i, block in enumerate(sorted_blocks):
                if not current_row:
                    current_row.append(block)
                else:
                    # Check if this text is on the same line as the previous one
                    last_block = current_row[-1]
                    y_diff = abs(block['y1'] - last_block['y1'])
                    
                    # If it's close enough vertically, it's probably the same row
                    if y_diff < 30:
                        current_row.append(block)
                    else:
                        # This is a new row - finish the current one
                        if len(current_row) >= 2:  # Need at least 2 columns for a table
                            rows.append(current_row)
                        
                        # Check if this looks like a numbered list (1. 2. 3. etc.)
                        text = block.get('text', '').strip()
                        if text and text[0].isdigit() and len(text) > 1:
                            # This might be part of a numbered list/table
                            current_row = [block]
                            # Check if the previous row also had a number
                            if rows and rows[-1] and rows[-1][0].get('text', '').strip() and rows[-1][0].get('text', '').strip()[0].isdigit():
                                # Group consecutive numbered items as potential table
                                rows[-1].append(block)
                                current_row = []
                        else:
                            current_row = [block]
                    
                    # Also check if this single block has multiple words that look like a table
                    text = block.get('text', '')
                    words = text.split()
                    # Look for patterns like "Column 1 Column 2" or "Row 1 Row 2 Row 3"
                    if len(words) >= 3 and any(word.lower() in ['column', 'row'] for word in words[:3]):
                        # This might be a table header/row
                        if len(words) >= 4:  # Multiple items
                            rows.append([block])  # Treat as a single-row table
            
            # Don't forget the last row
            if len(current_row) >= 2:
                rows.append(current_row)
            
            # Also look for numbered lists that might be tables
            numbered_blocks = []
            for block in sorted_blocks:
                text = block.get('text', '').strip()
                if text and text[0].isdigit() and len(text) > 1:
                    # Check if it's followed by text (not just a number)
                    if len(text.split()) > 1:
                        numbered_blocks.append(block)
            
            # If we have 3 or more numbered items, treat as a table
            if len(numbered_blocks) >= 3:
                numbered_row = [[block] for block in numbered_blocks]
                rows.extend(numbered_row)
            
            # Now group similar rows together to make table regions
            if len(rows) >= 1:
                table_regions = []
                
                if len(rows) >= 2:
                    current_region = [rows[0]]
                    for i in range(1, len(rows)):
                        # If this row has a similar number of columns, it's probably part of the same table
                        if abs(len(rows[i]) - len(rows[i-1])) <= 2:
                            current_region.append(rows[i])
                        else:
                            if len(current_region) >= 1:
                                table_regions.append(current_region)
                            current_region = [rows[i]]
                    
                    if len(current_region) >= 1:
                        table_regions.append(current_region)
                else:
                    # Single row, check if it looks like a table
                    table_regions = [rows]
                
                # Create table blocks from regions
                for region_idx, region in enumerate(table_regions):
                    # Calculate the bounding box for this table
                    all_blocks = [block for row in region for block in row]
                    
                    if not all_blocks:
                        continue
                    
                    x1 = min(b['x1'] for b in all_blocks)
                    y1 = min(b['y1'] for b in all_blocks)
                    x2 = max(b['x2'] for b in all_blocks)
                    y2 = max(b['y2'] for b in all_blocks)
                    
                    # Make sure it's not too big or too small
                    coverage = ((x2 - x1) * (y2 - y1)) / (w * h)
                    is_at_origin = x1 <= 10 and y1 <= 10
                    
                    if 0.02 < coverage < 0.50 and not (is_at_origin and coverage > 0.95):
                        # Get the text content to figure out what this is
                        text_content = [block['text'] for row in region for block in row]
                        
                        # Decide if this is a table or a chart
                        element_type = self._classify_table_vs_chart(region, text_content)
                        
                        if element_type == 'chart':
                            # Add as chart
                            tables.append({
                                'type': 'chart',
                                'subtype': 'unknown',
                                'x1': x1,
                                'y1': y1,
                                'x2': x2,
                                'y2': y2,
                                'bounding_box': [x1, y1, x2, y2],
                                'confidence': 0.6,
                                'content': 'chart_detected_text_aligned',
                                'rows': len(region),
                                'columns': len(region[0]) if region else 0
                            })
                        else:
                            # Add as table
                            tables.append({
                                'type': 'table',
                                'x1': x1,
                                'y1': y1,
                                'x2': x2,
                                'y2': y2,
                                'bounding_box': [x1, y1, x2, y2],
                                'confidence': 0.7,
                                'content': 'table_detected_text_aligned',
                                'rows': len(region),
                                'columns': len(region[0]) if region else 0
                            })
        
        except Exception as e:
            logger.error(f"Oops, couldn't find tables in the text: {e}")
        
        return tables
    
    def _postprocess_text(self, text: str) -> str:
        """
        Clean up the text that the OCR engine read (fix common mistakes)
        
        Args:
            text: The raw text from the OCR engine
            
        Returns:
            Cleaned up text that's easier to read
        """
        import re
        
        # Fix words that got squished together (like "Itcontains" -> "It contains")
        text = re.sub(r'([A-Z][a-z]+)([A-Z])', r'\1 \2', text)  # Capital+lowercase followed by capital
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # lowercase followed by uppercase
        
        # Fix common apostrophe mistakes
        text = text.replace("'", "'")
        text = text.replace("tis", "this")
        
        # Fix old-fashioned contractions
        text = re.sub(r"'tis", "this", text, flags=re.IGNORECASE)
        
        # Clean up extra spaces
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _generate_summary(self, blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Count up what we found in the document
        
        Args:
            blocks: All the things we found
            
        Returns:
            A summary of what we found
        """
        summary = {
            'total_blocks': len(blocks),
            'text_blocks': 0,
            'image_blocks': 0,
            'table_blocks': 0,
            'chart_blocks': 0,
            'shape_blocks': 0
        }
        
        # Count each type of thing we found
        for block in blocks:
            block_type = block.get('type', 'unknown')
            if block_type == 'text':
                summary['text_blocks'] += 1
            elif block_type == 'image':
                summary['image_blocks'] += 1
            elif block_type == 'table':
                summary['table_blocks'] += 1
            elif block_type == 'chart':
                summary['chart_blocks'] += 1
            elif block_type == 'shape':
                summary['shape_blocks'] += 1
        
        return summary
    
    def save_result(self, result: Dict[str, Any], output_path: str):
        """
        Save our findings to a JSON file
        
        Args:
            result: Everything we found in the document
            output_path: Where to save the file
        """
        def convert_to_native(obj):
            """Convert special number types to regular Python numbers"""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_to_native(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            return obj
        
        # Make sure all numbers are regular Python numbers (not special numpy types)
        result = convert_to_native(result)
        
        # Save everything to a JSON file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Result saved to: {output_path}")
    
    def export_blocks_to_csv(self, result: Dict[str, Any], output_path: str):
        """
        Export our findings to a CSV file (like a spreadsheet)
        
        Args:
            result: Everything we found in the document
            output_path: Where to save the CSV file
        """
        import pandas as pd
        
        blocks = result.get('blocks', [])
        
        if not blocks:
            logger.warning("No blocks to export")
            return
        
        # Create a spreadsheet with all our findings
        df_data = []
        for block in blocks:
            df_data.append({
                'type': block.get('type', 'unknown'),
                'subtype': block.get('subtype', ''),
                'x1': block.get('x1', 0),
                'y1': block.get('y1', 0),
                'x2': block.get('x2', 0),
                'y2': block.get('y2', 0),
                'confidence': block.get('confidence', 0.0),
                'text': block.get('text', '')[:50] if 'text' in block else '',  # First 50 chars
                'page': block.get('page', 1)
            })
        
        df = pd.DataFrame(df_data)
        df.to_csv(output_path, index=False)
        
        logger.info(f"Blocks exported to CSV: {output_path}")


def main():
    """The main function - this is where everything starts"""
    parser = argparse.ArgumentParser(
        description='Smart Document Parser - Reads documents like a human would'
    )
    parser.add_argument('input_file', nargs='?', help='The document you want to read (image or PDF)')
    parser.add_argument('-o', '--output', help='Where to save the results (default: input_file_parsed.json)')
    parser.add_argument('--csv', help='Also create a spreadsheet with the results')
    parser.add_argument('--tesseract-path', help='Where to find the text reading engine')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show more details about what we\'re doing')
    parser.add_argument('--no-gpu', action='store_true', help='Don\'t use the graphics card (use regular CPU)')
    parser.add_argument('--info', action='store_true', help='Show information about your computer and exit')
    
    args = parser.parse_args()
    
    # If they just want to see computer info
    if args.info:
        unified_parser = UnifiedParser(use_gpu=True)
        device_info = unified_parser.get_device_info()
        print("\n" + "="*60)
        print("COMPUTER INFORMATION")
        print("="*60)
        print(json.dumps(device_info, indent=2))
        print("="*60)
        return 0
    
    if not args.input_file:
        parser.error("You need to tell me which file to read (unless you're using --info)")
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Set up our document reader
        use_gpu = not args.no_gpu
        unified_parser = UnifiedParser(tesseract_path=args.tesseract_path, use_gpu=use_gpu)
        
        # Tell them what kind of computer we're using
        device_info = unified_parser.get_device_info()
        print(f"\n🔧 Computer Setup:")
        print(f"   Device: {device_info['device'].upper()}")
        if device_info.get('cuda_available'):
            print(f"   Graphics Card: {device_info['gpu_name']}")
            print(f"   CUDA Version: {device_info['cuda_version']}")
        print()
        
        # Read the document
        logger.info(f"Reading document: {args.input_file}")
        result = unified_parser.parse_with_positions(args.input_file)
        
        # Check if something went wrong
        if 'error' in result:
            logger.error(f"Oops, couldn't read the document: {result['error']}")
            return 1
        
        # Figure out where to save the results
        if args.output:
            output_path = args.output
        else:
            input_path = Path(args.input_file)
            output_path = input_path.with_suffix('.json').name + '_parsed.json'
        
        # Save everything we found
        unified_parser.save_result(result, output_path)
        
        # Create a spreadsheet if they want one
        if args.csv:
            unified_parser.export_blocks_to_csv(result, args.csv)
        
        # Show them what we found
        summary = result.get('summary', {})
        print("\n🔍 What We Found:")
        print(f"   Total things: {summary.get('total_blocks', 0)}")
        print(f"   Text pieces: {summary.get('text_blocks', 0)}")
        print(f"   Pictures: {summary.get('image_blocks', 0)}")
        print(f"   Tables: {summary.get('table_blocks', 0)}")
        print(f"   Charts: {summary.get('chart_blocks', 0)}")
        print(f"   Shapes: {summary.get('shape_blocks', 0)}")
        
        print(f"\n  Results saved to: {output_path}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Something went wrong: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
