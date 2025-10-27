

import os
import json
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
import base64
import io

from skimage import measure, segmentation, color
from skimage.feature import hog
from skimage.filters import threshold_otsu, gaussian
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import pytesseract
from pytesseract import Output

import layoutparser as lp
from layoutparser.elements import Rectangle, TextBlock, Layout

import pandas as pd
from scipy import ndimage
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist

logger = logging.getLogger(__name__)

class GraphicsHandler:
    
    
    def __init__(self, tesseract_path: Optional[str] = None):
        
        self.tesseract_path = tesseract_path
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
    
    def detect_visual_elements(self, image_path: str) -> Dict[str, Any]:
        
        try:
            image = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            elements = {
                'charts': self._detect_charts(image_rgb),
                'tables': self._detect_tables(image_rgb),
                'text_blocks': self._detect_text_blocks(image_rgb),
                'images': self._detect_images(image_rgb),
                'shapes': self._detect_shapes(gray),
                'colors': self._analyze_colors(image_rgb),
                'layout': self._analyze_layout(image_rgb)
            }
            
            return {
                'image_path': image_path,
                'visual_elements': elements,
                'metadata': {
                    'image_size': image.shape[:2],
                    'channels': image.shape[2] if len(image.shape) > 2 else 1,
                    'detection_method': 'computer_vision'
                }
            }
            
        except Exception as e:
            logger.error(f"Error detecting visual elements: {e}")
            return {'error': str(e)}
    
    def _detect_charts(self, image: np.ndarray) -> List[Dict[str, Any]]:
        
        charts = []
        
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            edges = cv2.Canny(gray, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
            
            if lines is not None:
                line_groups = self._group_lines(lines)
                
                for i, group in enumerate(line_groups):
                    if len(group) > 3:
                        bbox = self._get_bounding_box(group)
                        charts.append({
                            'id': i,
                            'type': 'line_chart',
                            'bounding_box': [int(x) for x in bbox],
                            'line_count': len(group),
                            'confidence': min(len(group) / 10, 1.0)
                        })
            

            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for i, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if 100 < area < 10000:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h
                    
                    if 0.1 < aspect_ratio < 10:
                        charts.append({
                            'id': len(charts),
                            'type': 'bar_chart',
                            'bounding_box': [int(x), int(y), int(x+w), int(y+h)],
                            'area': float(area),
                            'aspect_ratio': float(aspect_ratio),
                            'confidence': min(area / 1000, 1.0)
                        })
            
        except Exception as e:
            logger.error(f"Error detecting charts: {e}")
        
        return charts
    
    def _detect_tables(self, image: np.ndarray) -> List[Dict[str, Any]]:
        
        tables = []
        
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            

            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
            

            horizontal_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel)

            vertical_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, vertical_kernel)
            

            table_mask = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0.0)
            

            _, table_mask = cv2.threshold(table_mask, 127, 255, cv2.THRESH_BINARY)
            

            contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            

            img_height, img_width = image.shape[:2]
            
            for i, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if area > 500:
                    x, y, w, h = cv2.boundingRect(contour)
                    

                    coverage_x = w / img_width
                    coverage_y = h / img_height
                    

                    if coverage_x > 0.90 and coverage_y > 0.90:
                        continue
                    

                    table_region = image[y:y+h, x:x+w]
                    cell_count = self._count_table_cells(table_region)
                    

                    if cell_count >= 2:

                        conf = min(cell_count / 10.0, 1.0)
                        tables.append({
                            'id': i,
                            'bounding_box': [int(x), int(y), int(x+w), int(y+h)],
                            'area': float(area),
                            'cell_count': int(cell_count),
                            'confidence': conf
                        })
        
        except Exception as e:
            logger.error(f"Error detecting tables: {e}")
        
        return tables
    
    def _detect_text_blocks(self, image: np.ndarray) -> List[Dict[str, Any]]:
        
        text_blocks = []
        
        try:

            data = pytesseract.image_to_data(image, output_type=Output.DICT)
            
            for i in range(len(data['text'])):
                if int(data['conf'][i]) > 30:
                    x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                    text = data['text'][i].strip()
                    
                    if text and len(text) > 2:
                        text_blocks.append({
                            'id': i,
                            'text': text,
                            'bounding_box': [int(x), int(y), int(x+w), int(y+h)],
                            'confidence': int(data['conf'][i]) / 100.0,
                            'font_size': int(h)
                        })
        
        except Exception as e:
            logger.error(f"Error detecting text blocks: {e}")
        
        return text_blocks
    
    def _detect_images(self, image: np.ndarray) -> List[Dict[str, Any]]:
        
        images = []
        
        try:
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            color_variance = np.var(image, axis=2)
            
            high_variance_threshold = np.percentile(color_variance, 90)
            high_variance_regions = color_variance > high_variance_threshold
            
            try:
                ocr_data = pytesseract.image_to_data(gray, output_type=Output.DICT)
                text_regions = []
                
                for i in range(len(ocr_data['text'])):
                    if int(ocr_data['conf'][i]) > 30:
                        x = ocr_data['left'][i]
                        y = ocr_data['top'][i]
                        w = ocr_data['width'][i]
                        h = ocr_data['height'][i]
                        text_regions.append((x, y, x+w, y+h))
                

                text_mask = np.zeros(gray.shape, dtype=np.uint8)
                for region in text_regions:
                    cv2.rectangle(text_mask, (region[0], region[1]), (region[2], region[3]), 255, -1)
                
                
                kernel = np.ones((20, 20), np.uint8)
                text_mask = cv2.dilate(text_mask, kernel, iterations=1)
                
            except Exception:
                text_mask = np.zeros(gray.shape, dtype=np.uint8)
            
            edges = cv2.Canny(gray, 50, 150)
            edge_density = cv2.filter2D((edges > 0).astype(np.float32), -1, np.ones((30, 30)) / 900)
            high_edge_regions = edge_density > 0.1
            
            combined_mask = high_variance_regions & (text_mask == 0) & high_edge_regions
            
            contours, _ = cv2.findContours(
                combined_mask.astype(np.uint8), 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            for i, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if area > 1000:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h
                    
                    if 0.2 < aspect_ratio < 5.0 and w > 100 and h > 100:
                        roi = image[y:y+h, x:x+w]
                        roi_variance = np.var(roi, axis=2)
                        avg_variance = np.mean(roi_variance)
                        
                        if avg_variance > high_variance_threshold * 0.8:
                            images.append({
                                'id': i,
                                'bounding_box': [int(x), int(y), int(x+w), int(y+h)],
                                'area': float(area),
                                'aspect_ratio': float(aspect_ratio),
                                'confidence': min(area / 10000, 1.0),
                                'color_variance': float(avg_variance)
                            })
        
        except Exception as e:
            logger.error(f"Error detecting images: {e}")
        
        return images
    
    def _detect_shapes(self, gray: np.ndarray) -> List[Dict[str, Any]]:
        
        shapes = []
        
        try:

            edges = cv2.Canny(gray, 50, 150)
            

            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for i, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if area > 100:

                    epsilon = 0.02 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    

                    vertices = len(approx)
                    shape_type = self._classify_shape(vertices)
                    
                    if shape_type != 'unknown':
                        x, y, w, h = cv2.boundingRect(contour)
                        shapes.append({
                            'id': i,
                            'type': shape_type,
                            'bounding_box': [int(x), int(y), int(x+w), int(y+h)],
                            'area': float(area),
                            'vertices': vertices,
                            'confidence': min(area / 1000, 1.0)
                        })
        
        except Exception as e:
            logger.error(f"Error detecting shapes: {e}")
        
        return shapes
    
    def _analyze_colors(self, image: np.ndarray) -> Dict[str, Any]:
        
        try:

            pixels = image.reshape(-1, 3)
            

            from sklearn.cluster import KMeans
            

            sample_size = min(1000, len(pixels))
            sample_pixels = pixels[np.random.choice(len(pixels), sample_size, replace=False)]
            

            kmeans = KMeans(n_clusters=5, random_state=42)
            kmeans.fit(sample_pixels)
            

            dominant_colors = kmeans.cluster_centers_.astype(int)
            color_counts = np.bincount(kmeans.labels_)
            color_percentages = color_counts / len(kmeans.labels_) * 100
            

            mean_color = np.mean(pixels, axis=0)
            color_variance = np.var(pixels, axis=0)
            
            return {
                'dominant_colors': dominant_colors.tolist(),
                'color_percentages': color_percentages.tolist(),
                'mean_color': mean_color.tolist(),
                'color_variance': color_variance.tolist(),
                'brightness': float(np.mean(mean_color)),
                'saturation': float(np.mean(color_variance))
            }
        
        except Exception as e:
            logger.error(f"Error analyzing colors: {e}")
            return {'error': str(e)}
    
    def _analyze_layout(self, image: np.ndarray) -> Dict[str, Any]:
        
        try:
            height, width = image.shape[:2]
            

            grid_size = 3
            regions = []
            
            for i in range(grid_size):
                for j in range(grid_size):
                    y1, y2 = i * height // grid_size, (i + 1) * height // grid_size
                    x1, x2 = j * width // grid_size, (j + 1) * width // grid_size
                    region = image[y1:y2, x1:x2]
                    

                    region_brightness = np.mean(region)
                    region_contrast = np.std(region)
                    
                    regions.append({
                        'position': (i, j),
                        'brightness': float(region_brightness),
                        'contrast': float(region_contrast),
                        'coordinates': [int(x1), int(y1), int(x2), int(y2)]
                    })
            

            center_region = regions[4]
            edge_regions = [regions[i] for i in [0, 1, 2, 3, 5, 6, 7, 8]]
            
            center_brightness = center_region['brightness']
            edge_brightness = np.mean([r['brightness'] for r in edge_regions])
            
            return {
                'grid_regions': regions,
                'center_focus': bool(center_brightness > edge_brightness),
                'brightness_contrast': float(center_brightness - edge_brightness),
                'layout_balance': float(self._calculate_layout_balance(regions)),
                'image_dimensions': [int(width), int(height)],
                'aspect_ratio': float(width / height)
            }
        
        except Exception as e:
            logger.error(f"Error analyzing layout: {e}")
            return {'error': str(e)}
    
    def extract_chart_data(self, image_path: str, chart_bbox: List[int]) -> Dict[str, Any]:
        
        try:

            image = cv2.imread(image_path)
            x1, y1, x2, y2 = chart_bbox
            chart_image = image[y1:y2, x1:x2]
            

            gray = cv2.cvtColor(chart_image, cv2.COLOR_BGR2GRAY)
            

            axes = self._detect_chart_axes(gray)
            data_points = self._extract_data_points(chart_image, axes)
            

            labels = self._extract_chart_labels(chart_image)
            
            return {
                'chart_type': self._classify_chart_type(chart_image),
                'axes': axes,
                'data_points': data_points,
                'labels': labels,
                'bounding_box': chart_bbox
            }
        
        except Exception as e:
            logger.error(f"Error extracting chart data: {e}")
            return {'error': str(e)}
    
    def generate_visual_summary(self, image_path: str) -> Dict[str, Any]:
        
        try:

            elements = self.detect_visual_elements(image_path)
            

            summary = {
                'file_path': image_path,
                'analysis_timestamp': str(pd.Timestamp.now()),
                'visual_elements': elements,
                'statistics': {
                    'total_charts': len(elements.get('charts', [])),
                    'total_tables': len(elements.get('tables', [])),
                    'total_text_blocks': len(elements.get('text_blocks', [])),
                    'total_images': len(elements.get('images', [])),
                    'total_shapes': len(elements.get('shapes', [])),
                    'dominant_colors': len(elements.get('colors', {}).get('dominant_colors', [])),
                    'layout_balance': elements.get('layout', {}).get('layout_balance', 0)
                },
                'content_type': self._classify_content_type(elements),
                'accessibility_score': self._calculate_accessibility_score(elements)
            }
            
            return summary
        
        except Exception as e:
            logger.error(f"Error generating visual summary: {e}")
            return {'error': str(e)}
    

    def _group_lines(self, lines):
        
        if lines is None or len(lines) == 0:
            return []
        
        groups = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            added_to_group = False
            
            for group in groups:
                for existing_line in group:
                    ex1, ey1, ex2, ey2 = existing_line[0]

                    if (abs(x1 - ex1) < 20 and abs(y1 - ey1) < 20) or \
                       (abs(x2 - ex2) < 20 and abs(y2 - ey2) < 20):
                        group.append(line)
                        added_to_group = True
                        break
                if added_to_group:
                    break
            
            if not added_to_group:
                groups.append([line])
        
        return groups
    
    def _get_bounding_box(self, lines):
        
        if not lines:
            return [0, 0, 0, 0]
        
        all_points = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            all_points.extend([(x1, y1), (x2, y2)])
        
        xs = [p[0] for p in all_points]
        ys = [p[1] for p in all_points]
        
        return [min(xs), min(ys), max(xs), max(ys)]
    
    def _count_table_cells(self, table_region):
        
        try:
            gray = cv2.cvtColor(table_region, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            return len([c for c in contours if cv2.contourArea(c) > 50])
        except:
            return 0
    
    def _classify_shape(self, vertices):
        
        if vertices == 3:
            return 'triangle'
        elif vertices == 4:
            return 'rectangle'
        elif vertices == 5:
            return 'pentagon'
        elif vertices == 6:
            return 'hexagon'
        elif vertices > 8:
            return 'circle'
        else:
            return 'polygon'
    
    def _detect_chart_axes(self, gray):
        

        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, minLineLength=20, maxLineGap=10)
        
        axes = {'x_axis': [], 'y_axis': []}
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                
                if abs(angle) < 15:
                    axes['x_axis'].append(line[0])
                elif abs(angle - 90) < 15:
                    axes['y_axis'].append(line[0])
        
        return axes
    
    def _extract_data_points(self, chart_image, axes):
        

        gray = cv2.cvtColor(chart_image, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.01, minDistance=10)
        
        data_points = []
        if corners is not None:
            for corner in corners:
                x, y = corner.ravel()
                data_points.append([int(x), int(y)])
        
        return data_points
    
    def _extract_chart_labels(self, chart_image):
        
        try:
            data = pytesseract.image_to_data(chart_image, output_type=Output.DICT)
            labels = []
            
            for i in range(len(data['text'])):
                if int(data['conf'][i]) > 30:
                    text = data['text'][i].strip()
                    if text:
                        labels.append({
                            'text': text,
                            'position': [data['left'][i], data['top'][i]],
                            'confidence': int(data['conf'][i]) / 100.0
                        })
            
            return labels
        except:
            return []
    
    def _classify_chart_type(self, chart_image):
        

        gray = cv2.cvtColor(chart_image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        

        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, minLineLength=20, maxLineGap=10)
        
        if lines is not None:
            horizontal_lines = 0
            vertical_lines = 0
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                
                if abs(angle) < 15:
                    horizontal_lines += 1
                elif abs(angle - 90) < 15:
                    vertical_lines += 1
            
            if horizontal_lines > vertical_lines:
                return 'bar_chart'
            elif vertical_lines > horizontal_lines:
                return 'line_chart'
            else:
                return 'mixed_chart'
        
        return 'unknown'
    
    def _classify_content_type(self, elements):
        
        chart_count = len(elements.get('charts', []))
        table_count = len(elements.get('tables', []))
        text_count = len(elements.get('text_blocks', []))
        image_count = len(elements.get('images', []))
        
        if chart_count > table_count and chart_count > text_count:
            return 'infographic'
        elif table_count > chart_count and table_count > text_count:
            return 'data_table'
        elif text_count > max(chart_count, table_count, image_count):
            return 'text_document'
        elif image_count > max(chart_count, table_count, text_count):
            return 'image_gallery'
        else:
            return 'mixed_content'
    
    def _calculate_accessibility_score(self, elements):
        
        score = 100
        

        text_blocks = len(elements.get('text_blocks', []))
        if text_blocks == 0:
            score -= 30
        elif text_blocks < 3:
            score -= 15
        

        colors = elements.get('colors', {})
        if 'brightness' in colors and colors['brightness'] < 50:
            score -= 20
        

        charts = elements.get('charts', [])
        labels = elements.get('text_blocks', [])
        if charts and not labels:
            score -= 25
        
        return max(0, score)
    
    def _calculate_layout_balance(self, regions):
        
        if not regions:
            return 0.0
        

        brightness_values = [r['brightness'] for r in regions]
        balance_score = 100 - (float(np.std(brightness_values)) * 10)
        
        return max(0.0, min(100.0, balance_score))

def main():
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Graphics and Visual Content Handler')
    parser.add_argument('input_file', help='Path to image file')
    parser.add_argument('-o', '--output', help='Output file path (optional)')
    parser.add_argument('--tesseract-path', help='Path to tesseract executable')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:

        handler = GraphicsHandler(tesseract_path=args.tesseract_path)
        

        logger.info(f"Analyzing visual content: {args.input_file}")
        summary = handler.generate_visual_summary(args.input_file)
        

        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            logger.info(f"Visual analysis saved to: {args.output}")
        else:
            print(json.dumps(summary, indent=2, ensure_ascii=False))
                
    except Exception as e:
        logger.error(f"Error analyzing visual content: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
