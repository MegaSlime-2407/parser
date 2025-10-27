

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Union, Optional, Any
import logging

import cv2
import numpy as np
import pytesseract
from PIL import Image
import fitz  
from pdf2image import convert_from_path

import layoutparser as lp
from layoutparser.elements import Rectangle, TextBlock, Layout

from graphics_handler import GraphicsHandler

try:
    from marker import convert_single_pdf
    from marker.models import load_all_models
    MARKER_AVAILABLE = True
except ImportError:
    MARKER_AVAILABLE = False

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DocumentParser:
    
    
    def __init__(self, tesseract_path: Optional[str] = None, marker_models_path: Optional[str] = None):
        
        self.tesseract_path = tesseract_path
        self.marker_models_path = marker_models_path
        
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        
        self.layout_model = None
        try:
            has_detectron = hasattr(lp, 'is_detectron2_available') and lp.is_detectron2_available()
            has_paddle = hasattr(lp, 'is_paddle_available') and lp.is_paddle_available()
            
            if has_detectron:
                try:
                    self.layout_model = lp.Detectron2LayoutModel(
                        'lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config',
                        extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8],
                        label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}
                    )
                except Exception:
                    self.layout_model = None
            elif has_paddle:
                try:
                    self.layout_model = lp.PaddleDetectionLayoutModel(
                        config_path="lp://PubLayNet/ppyolov2_r50vd_dcn_365e_publaynet/config.yaml",
                        threshold=0.8,
                        label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}
                    )
                except Exception:
                    self.layout_model = None

        except Exception:
            self.layout_model = None
        
        self.marker_models = None
        if MARKER_AVAILABLE and marker_models_path:
            try:
                self.marker_models = load_all_models(marker_models_path)
                logger.info("Marker models loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load Marker models: {e}")
        
        self.graphics_handler = GraphicsHandler(tesseract_path=tesseract_path)
    
    def extract_text_from_image(self, image_path: str) -> str:
        
        try:
            image = Image.open(image_path)
            text = pytesseract.image_to_string(image, lang='eng')
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting text from image {image_path}: {e}")
            return ""
    
    def analyze_layout(self, image_path: str) -> Layout:
        
        try:
            if self.layout_model is None:
                logger.warning("LayoutParser model not available, returning empty layout")
                return Layout()
                
            image = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            layout = self.layout_model.detect(image_rgb)
            return layout
        except Exception as e:
            logger.error(f"Error analyzing layout for {image_path}: {e}")
            return Layout()
    
    def extract_text_with_layout(self, image_path: str) -> Dict[str, Any]:
        
        try:

            image = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            visual_elements = self.graphics_handler.detect_visual_elements(image_path)
            
            if self.layout_model is None:
                logger.warning("LayoutParser model not available, using basic OCR")

                text = pytesseract.image_to_string(image_rgb, lang='eng')
                
                regions = []
                if 'visual_elements' in visual_elements:
                    elements = visual_elements['visual_elements']
                    
                    for img in elements.get('images', []):
                        regions.append({
                            'id': f"img_{img['id']}",
                            'type': 'image',
                            'coordinates': [int(x) for x in img['bounding_box']],
                            'text': '',
                            'confidence': float(img.get('confidence', 0.0)),
                            'metadata': {
                                'area': float(img.get('area', 0)),
                                'aspect_ratio': float(img.get('aspect_ratio', 0)),
                                'color_variance': float(img.get('color_variance', 0))
                            }
                        })
                    
                    for chart in elements.get('charts', []):
                        regions.append({
                            'id': f"chart_{chart['id']}",
                            'type': 'chart',
                            'coordinates': [int(x) for x in chart['bounding_box']],
                            'text': '',
                            'confidence': float(chart.get('confidence', 0.0)),
                            'metadata': {
                                'chart_type': chart.get('type', 'unknown'),
                                'line_count': int(chart.get('line_count', 0))
                            }
                        })
                    
                    for table in elements.get('tables', []):
                        regions.append({
                            'id': f"table_{table['id']}",
                            'type': 'table',
                            'coordinates': [int(x) for x in table['bounding_box']],
                            'text': '',
                            'confidence': float(table.get('confidence', 0.0)),
                            'metadata': {
                                'table_type': table.get('type', 'unknown'),
                                'cell_count': int(table.get('cell_count', 0))
                            }
                        })
                    
                    for shape in elements.get('shapes', []):
                        regions.append({
                            'id': f"shape_{shape['id']}",
                            'type': 'shape',
                            'coordinates': [int(x) for x in shape['bounding_box']],
                            'text': '',
                            'confidence': float(shape.get('confidence', 0.0)),
                            'metadata': {
                                'shape_type': shape.get('type', 'unknown'),
                                'area': float(shape.get('area', 0))
                            }
                        })
                
                return {
                    'full_text': text.strip(),
                    'regions': regions,
                    'metadata': {
                        'image_path': image_path,
                        'total_regions': len(regions),
                        'extraction_method': 'tesseract_graphics',
                        'visual_elements': visual_elements.get('visual_elements', {})
                    }
                }
            
            layout = self.layout_model.detect(image_rgb)
            
            extracted_data = {
                'full_text': '',
                'regions': [],
                'metadata': {
                    'image_path': image_path,
                    'total_regions': len(layout),
                    'extraction_method': 'layoutparser_tesseract_graphics',
                    'visual_elements': visual_elements.get('visual_elements', {})
                }
            }
            
            for i, block in enumerate(layout):
                x1, y1, x2, y2 = block.coordinates
                
                cropped_image = image_rgb[y1:y2, x1:x2]
                
                region_text = pytesseract.image_to_string(cropped_image, lang='eng')
                
                region_data = {
                    'id': i,
                    'type': block.type,
                    'coordinates': [int(x1), int(y1), int(x2), int(y2)],
                    'text': region_text.strip(),
                    'confidence': getattr(block, 'score', 0.0)
                }
                
                extracted_data['regions'].append(region_data)
                extracted_data['full_text'] += region_text + '\n'
            
            if 'visual_elements' in visual_elements:
                elements = visual_elements['visual_elements']
                
                for img in elements.get('images', []):
                    extracted_data['regions'].append({
                        'id': f"img_{img['id']}",
                        'type': 'image',
                        'coordinates': [int(x) for x in img['bounding_box']],
                        'text': '',
                        'confidence': float(img.get('confidence', 0.0)),
                        'metadata': {
                            'area': float(img.get('area', 0)),
                            'aspect_ratio': float(img.get('aspect_ratio', 0)),
                            'color_variance': float(img.get('color_variance', 0))
                        }
                    })
                
                for chart in elements.get('charts', []):
                    extracted_data['regions'].append({
                        'id': f"chart_{chart['id']}",
                        'type': 'chart',
                        'coordinates': [int(x) for x in chart['bounding_box']],
                        'text': '',
                        'confidence': float(chart.get('confidence', 0.0)),
                        'metadata': {
                            'chart_type': chart.get('type', 'unknown'),
                            'line_count': int(chart.get('line_count', 0))
                        }
                    })
                
                for table in elements.get('tables', []):
                    extracted_data['regions'].append({
                        'id': f"table_{table['id']}",
                        'type': 'table',
                        'coordinates': [int(x) for x in table['bounding_box']],
                        'text': '',
                        'confidence': float(table.get('confidence', 0.0)),
                        'metadata': {
                            'table_type': table.get('type', 'unknown'),
                            'cell_count': int(table.get('cell_count', 0))
                        }
                    })
                
                for shape in elements.get('shapes', []):
                    extracted_data['regions'].append({
                        'id': f"shape_{shape['id']}",
                        'type': 'shape',
                        'coordinates': [int(x) for x in shape['bounding_box']],
                        'text': '',
                        'confidence': float(shape.get('confidence', 0.0)),
                        'metadata': {
                            'shape_type': shape.get('type', 'unknown'),
                            'area': float(shape.get('area', 0))
                        }
                    })
            
            extracted_data['metadata']['total_regions'] = len(extracted_data['regions'])
            
            return extracted_data
            
        except Exception as e:
            logger.error(f"Error extracting text with layout from {image_path}: {e}")
            return {'full_text': '', 'regions': [], 'metadata': {'error': str(e)}}
    
    def process_pdf_with_marker(self, pdf_path: str) -> Dict[str, Any]:
        
        if not MARKER_AVAILABLE or not self.marker_models:
            return self.process_pdf_fallback(pdf_path)
        
        try:
            result = convert_single_pdf(pdf_path, self.marker_models)
            
            return {
                'full_text': result.get('text', ''),
                'markdown': result.get('markdown', ''),
                'metadata': {
                    'pdf_path': pdf_path,
                    'extraction_method': 'marker',
                    'pages_processed': result.get('pages', 0)
                }
            }
        except Exception as e:
            logger.error(f"Error processing PDF with Marker: {e}")
            return self.process_pdf_fallback(pdf_path)
    
    def process_pdf_fallback(self, pdf_path: str) -> Dict[str, Any]:
        
        try:
            images = convert_from_path(pdf_path, dpi=300)
            
            extracted_data = {
                'full_text': '',
                'pages': [],
                'metadata': {
                    'pdf_path': pdf_path,
                    'extraction_method': 'pdf2image_tesseract',
                    'total_pages': len(images)
                }
            }
            
            for page_num, image in enumerate(images):
                temp_image_path = f"/tmp/temp_page_{page_num}.png"
                image.save(temp_image_path)
                
                page_data = self.extract_text_with_layout(temp_image_path)
                page_data['page_number'] = page_num + 1
                
                extracted_data['pages'].append(page_data)
                extracted_data['full_text'] += page_data['full_text'] + '\n'
                
                os.remove(temp_image_path)
            
            return extracted_data
            
        except Exception as e:
            logger.error(f"Error processing PDF fallback: {e}")
            return {'full_text': '', 'pages': [], 'metadata': {'error': str(e)}}
    
    def process_document(self, file_path: str, output_format: str = 'json') -> Union[str, Dict[str, Any]]:
        
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_extension = file_path.suffix.lower()
        
        if file_extension == '.pdf':
            if MARKER_AVAILABLE and self.marker_models:
                result = self.process_pdf_with_marker(str(file_path))
            else:
                result = self.process_pdf_fallback(str(file_path))
        elif file_extension in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
            result = self.extract_text_with_layout(str(file_path))
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        if output_format.lower() == 'markdown':
            return self._format_as_markdown(result)
        else:
            return result
    
    def _format_as_markdown(self, data: Dict[str, Any]) -> str:
        
        markdown_content = []
        
        markdown_content.append("# Document Content\n")
        
        if 'metadata' in data:
            markdown_content.append("## Metadata\n")
            for key, value in data['metadata'].items():
                markdown_content.append(f"- **{key}**: {value}\n")
            markdown_content.append("\n")
        
        if 'pages' in data:
            for page in data['pages']:
                markdown_content.append(f"## Page {page.get('page_number', 'Unknown')}\n")
                
                if 'regions' in page:
                    for region in page['regions']:
                        if region['text'].strip():
                            markdown_content.append(f"### {region['type']}\n")
                            markdown_content.append(region['text'] + "\n")
                else:
                    markdown_content.append(page.get('full_text', '') + "\n")
        else:
            if 'regions' in data:
                for region in data['regions']:
                    if region['text'].strip():
                        markdown_content.append(f"## {region['type']}\n")
                        markdown_content.append(region['text'] + "\n")
            else:
                markdown_content.append(data.get('full_text', ''))
        
        return '\n'.join(markdown_content)
    
    def analyze_visual_content(self, file_path: str) -> Dict[str, Any]:
        
        try:
            file_path = Path(file_path)
            file_extension = file_path.suffix.lower()
            
            if file_extension in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
                return self.graphics_handler.generate_visual_summary(str(file_path))
            
            elif file_extension == '.pdf':
                try:
                    images = convert_from_path(str(file_path), dpi=300)
                    visual_analysis = {
                        'file_path': str(file_path),
                        'pages': [],
                        'overall_analysis': {
                            'total_pages': len(images),
                            'visual_elements': {
                                'total_charts': 0,
                                'total_tables': 0,
                                'total_images': 0,
                                'total_shapes': 0
                            }
                        }
                    }
                    
                    for page_num, image in enumerate(images):
                        temp_image_path = f"/tmp/temp_page_{page_num}.png"
                        image.save(temp_image_path)
                        
                        page_analysis = self.graphics_handler.generate_visual_summary(temp_image_path)
                        visual_analysis['pages'].append({
                            'page_number': page_num + 1,
                            'analysis': page_analysis
                        })
                        
                        if 'visual_elements' in page_analysis:
                            stats = page_analysis['visual_elements']
                            visual_analysis['overall_analysis']['visual_elements']['total_charts'] += len(stats.get('charts', []))
                            visual_analysis['overall_analysis']['visual_elements']['total_tables'] += len(stats.get('tables', []))
                            visual_analysis['overall_analysis']['visual_elements']['total_images'] += len(stats.get('images', []))
                            visual_analysis['overall_analysis']['visual_elements']['total_shapes'] += len(stats.get('shapes', []))
                        
                        os.remove(temp_image_path)
                    
                    return visual_analysis
                    
                except Exception as e:
                    logger.error(f"Error analyzing PDF visual content: {e}")
                    return {'error': str(e)}
            
            else:
                raise ValueError(f"Unsupported file format for visual analysis: {file_extension}")
                
        except Exception as e:
            logger.error(f"Error analyzing visual content: {e}")
            return {'error': str(e)}

def main():
    
    parser = argparse.ArgumentParser(description='Document Parser using LayoutParser + Tesseract + Marker')
    parser.add_argument('input_file', help='Path to input document (PDF, image, etc.)')
    parser.add_argument('-o', '--output', help='Output file path (optional)')
    parser.add_argument('-f', '--format', choices=['json', 'markdown'], default='json',
                       help='Output format (default: json)')
    parser.add_argument('--tesseract-path', help='Path to tesseract executable')
    parser.add_argument('--marker-models', help='Path to marker models directory')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        doc_parser = DocumentParser(
            tesseract_path=args.tesseract_path,
            marker_models_path=args.marker_models
        )
        
        logger.info(f"Processing document: {args.input_file}")
        result = doc_parser.process_document(args.input_file, args.format)
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                if args.format == 'json':
                    json.dump(result, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
                else:
                    f.write(result)
            logger.info(f"Output saved to: {args.output}")
        else:
            if args.format == 'json':
                print(json.dumps(result, indent=2, ensure_ascii=False, cls=NumpyEncoder))
            else:
                print(result)
                
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
