"""
PubTabNet Data Loader and Preparation

Handles loading and preprocessing of PubTabNet dataset for reconstruction validation.
"""

import json
import os
import random
from typing import Dict, List, Tuple, Optional, Iterator
from dataclasses import dataclass
from pathlib import Path
import re


@dataclass
class PubTabNetSample:
    """Single PubTabNet sample with image and annotation data"""
    image_path: str
    html_annotation: str
    cell_bboxes: List[List[float]]  # Cell bounding boxes
    image_width: int
    image_height: int
    table_id: str
    split: str  # 'train', 'val', 'test'
    metadata: Dict


@dataclass
class PubTabNetDataset:
    """PubTabNet dataset container"""
    samples: List[PubTabNetSample]
    split: str
    total_samples: int


class PubTabNetDataLoader:
    """Data loader for PubTabNet dataset"""
    
    def __init__(self, dataset_root: str = None):
        """
        Initialize PubTabNet data loader
        
        Args:
            dataset_root: Root directory of PubTabNet dataset
        """
        self.dataset_root = dataset_root or self._get_default_dataset_path()
        self.supported_splits = ['train', 'val', 'test']
        
    def _get_default_dataset_path(self) -> str:
        """Get default dataset path (can be overridden)"""
        # Common PubTabNet dataset locations
        possible_paths = [
            "./data/pubtabnet",
            "../data/pubtabnet", 
            "~/datasets/pubtabnet",
            "/data/pubtabnet"
        ]
        
        for path in possible_paths:
            expanded_path = os.path.expanduser(path)
            if os.path.exists(expanded_path):
                return expanded_path
        
        # Return a default path for setup
        return "./data/pubtabnet"
    
    def load_split(self, split: str, max_samples: Optional[int] = None, 
                   random_seed: int = 42) -> PubTabNetDataset:
        """
        Load a specific split of the PubTabNet dataset
        
        Args:
            split: Dataset split ('train', 'val', 'test')
            max_samples: Maximum number of samples to load (None for all)
            random_seed: Random seed for reproducible sampling
            
        Returns:
            PubTabNetDataset object
        """
        if split not in self.supported_splits:
            raise ValueError(f"Unsupported split: {split}. Must be one of {self.supported_splits}")
        
        # Set random seed for reproducible sampling
        random.seed(random_seed)
        
        # Load annotation file
        annotation_file = os.path.join(self.dataset_root, f"PubTabNet_{split}.jsonl")
        
        if not os.path.exists(annotation_file):
            # If real dataset not available, create synthetic samples for testing
            print(f"WARNING: PubTabNet {split} dataset not found at {annotation_file}")
            print("Creating synthetic samples for testing...")
            return self._create_synthetic_dataset(split, max_samples or 100)
        
        samples = []
        with open(annotation_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                if max_samples and len(samples) >= max_samples:
                    break
                
                try:
                    sample_data = json.loads(line.strip())
                    sample = self._parse_sample(sample_data, split)
                    if sample:
                        samples.append(sample)
                except Exception as e:
                    print(f"WARNING: Error parsing line {line_num}: {e}")
                    continue
        
        return PubTabNetDataset(
            samples=samples,
            split=split,
            total_samples=len(samples)
        )
    
    def _parse_sample(self, sample_data: Dict, split: str) -> Optional[PubTabNetSample]:
        """Parse a single PubTabNet sample"""
        try:
            # Extract basic information
            table_id = sample_data.get('filename', f"sample_{len(sample_data)}")
            html_annotation = sample_data.get('html', '')
            
            # Extract image dimensions
            image_width = sample_data.get('img_width', 1000)
            image_height = sample_data.get('img_height', 1000)
            
            # Extract cell bounding boxes if available
            cell_bboxes = sample_data.get('cell_bboxes', [])
            
            # Construct image path
            image_path = os.path.join(self.dataset_root, split, f"{table_id}.png")
            
            # Extract metadata
            metadata = {
                'original_data': sample_data,
                'has_cell_bboxes': len(cell_bboxes) > 0,
                'html_length': len(html_annotation)
            }
            
            return PubTabNetSample(
                image_path=image_path,
                html_annotation=html_annotation,
                cell_bboxes=cell_bboxes,
                image_width=image_width,
                image_height=image_height,
                table_id=table_id,
                split=split,
                metadata=metadata
            )
            
        except Exception as e:
            print(f"WARNING: Error parsing sample: {e}")
            return None
    
    def _create_synthetic_dataset(self, split: str, num_samples: int) -> PubTabNetDataset:
        """Create synthetic PubTabNet samples for testing when real data is not available"""
        print(f"Creating {num_samples} synthetic {split} samples...")
        
        samples = []
        for i in range(num_samples):
            # Generate synthetic HTML annotation
            html_annotation = self._generate_synthetic_html(i)
            
            # Generate synthetic cell bboxes
            cell_bboxes = self._generate_synthetic_bboxes(i)
            
            sample = PubTabNetSample(
                image_path=f"synthetic_{split}_{i}.png",
                html_annotation=html_annotation,
                cell_bboxes=cell_bboxes,
                image_width=800,
                image_height=600,
                table_id=f"synthetic_{split}_{i}",
                split=split,
                metadata={
                    'synthetic': True,
                    'generated_index': i,
                    'has_cell_bboxes': True,
                    'html_length': len(html_annotation)
                }
            )
            samples.append(sample)
        
        return PubTabNetDataset(
            samples=samples,
            split=split,
            total_samples=len(samples)
        )
    
    def _generate_synthetic_html(self, index: int) -> str:
        """Generate synthetic HTML table annotation"""
        # Different table types for variety
        table_types = [
            # Simple 2x3 table
            """
            <table>
                <tr>
                    <td>Header 1</td>
                    <td>Header 2</td>
                    <td>Header 3</td>
                </tr>
                <tr>
                    <td>Data 1</td>
                    <td>Data 2</td>
                    <td>Data 3</td>
                </tr>
            </table>
            """,
            # Table with colspan
            """
            <table>
                <tr>
                    <th colspan="2">Product Information</th>
                    <th>Price</th>
                </tr>
                <tr>
                    <td>Laptop</td>
                    <td>Gaming</td>
                    <td>$999</td>
                </tr>
                <tr>
                    <td>Mouse</td>
                    <td>Wireless</td>
                    <td>$25</td>
                </tr>
            </table>
            """,
            # Table with rowspan
            """
            <table>
                <tr>
                    <th>Category</th>
                    <th>Item</th>
                    <th>Price</th>
                </tr>
                <tr>
                    <td rowspan="2">Electronics</td>
                    <td>Laptop</td>
                    <td>$999</td>
                </tr>
                <tr>
                    <td>Mouse</td>
                    <td>$25</td>
                </tr>
            </table>
            """,
            # Complex table with both rowspan and colspan
            """
            <table>
                <tr>
                    <th colspan="3">Sales Report</th>
                </tr>
                <tr>
                    <th>Product</th>
                    <th>Q1</th>
                    <th>Q2</th>
                </tr>
                <tr>
                    <td rowspan="2">Electronics</td>
                    <td>100</td>
                    <td>120</td>
                </tr>
                <tr>
                    <td>80</td>
                    <td>90</td>
                </tr>
            </table>
            """
        ]
        
        # Select table type based on index
        table_type = table_types[index % len(table_types)]
        
        # Add some variation to content
        variations = [
            ("Header 1", f"Header A{index}"),
            ("Header 2", f"Header B{index}"),
            ("Data 1", f"Data A{index}"),
            ("Data 2", f"Data B{index}"),
            ("Product Information", f"Product Info {index}"),
            ("Sales Report", f"Sales Report {index}")
        ]
        
        html = table_type
        for old, new in variations:
            html = html.replace(old, new)
        
        return html.strip()
    
    def _generate_synthetic_bboxes(self, index: int) -> List[List[float]]:
        """Generate synthetic cell bounding boxes"""
        # Base table dimensions
        table_x, table_y = 100, 100
        cell_width, cell_height = 150, 50
        
        # Different bbox patterns based on index
        patterns = [
            # Simple 2x3 grid
            [
                [table_x, table_y, table_x + cell_width, table_y + cell_height],
                [table_x + cell_width, table_y, table_x + 2*cell_width, table_y + cell_height],
                [table_x + 2*cell_width, table_y, table_x + 3*cell_width, table_y + cell_height],
                [table_x, table_y + cell_height, table_x + cell_width, table_y + 2*cell_height],
                [table_x + cell_width, table_y + cell_height, table_x + 2*cell_width, table_y + 2*cell_height],
                [table_x + 2*cell_width, table_y + cell_height, table_x + 3*cell_width, table_y + 2*cell_height]
            ],
            # Table with colspan (wider header)
            [
                [table_x, table_y, table_x + 2*cell_width, table_y + cell_height],
                [table_x + 2*cell_width, table_y, table_x + 3*cell_width, table_y + cell_height],
                [table_x, table_y + cell_height, table_x + cell_width, table_y + 2*cell_height],
                [table_x + cell_width, table_y + cell_height, table_x + 2*cell_width, table_y + 2*cell_height],
                [table_x + 2*cell_width, table_y + cell_height, table_x + 3*cell_width, table_y + 2*cell_height],
                [table_x, table_y + 2*cell_height, table_x + cell_width, table_y + 3*cell_height],
                [table_x + cell_width, table_y + 2*cell_height, table_x + 2*cell_width, table_y + 3*cell_height],
                [table_x + 2*cell_width, table_y + 2*cell_height, table_x + 3*cell_width, table_y + 3*cell_height]
            ]
        ]
        
        pattern = patterns[index % len(patterns)]
        
        # Add some random variation
        variation = (index % 10) * 5  # Small variation based on index
        return [[coord + variation for coord in bbox] for bbox in pattern]
    
    def get_sample_iterator(self, dataset: PubTabNetDataset, 
                          batch_size: int = 1) -> Iterator[List[PubTabNetSample]]:
        """Get iterator for batching samples"""
        samples = dataset.samples
        for i in range(0, len(samples), batch_size):
            yield samples[i:i + batch_size]
    
    def validate_dataset(self, dataset: PubTabNetDataset) -> Dict[str, any]:
        """Validate dataset integrity and provide statistics"""
        stats = {
            'total_samples': len(dataset.samples),
            'samples_with_html': 0,
            'samples_with_bboxes': 0,
            'synthetic_samples': 0,
            'avg_html_length': 0,
            'avg_bbox_count': 0,
            'errors': []
        }
        
        html_lengths = []
        bbox_counts = []
        
        for sample in dataset.samples:
            # Count samples with HTML
            if sample.html_annotation and sample.html_annotation.strip():
                stats['samples_with_html'] += 1
                html_lengths.append(len(sample.html_annotation))
            
            # Count samples with bboxes
            if sample.cell_bboxes:
                stats['samples_with_bboxes'] += 1
                bbox_counts.append(len(sample.cell_bboxes))
            
            # Count synthetic samples
            if sample.metadata.get('synthetic', False):
                stats['synthetic_samples'] += 1
            
            # Validate HTML structure
            if sample.html_annotation and '<table>' not in sample.html_annotation:
                stats['errors'].append(f"Sample {sample.table_id}: No table tag found")
        
        # Calculate averages
        stats['avg_html_length'] = sum(html_lengths) / len(html_lengths) if html_lengths else 0
        stats['avg_bbox_count'] = sum(bbox_counts) / len(bbox_counts) if bbox_counts else 0
        
        return stats


def main():
    """Test the PubTabNet data loader"""
    print("📊 Testing PubTabNet Data Loader")
    print("=" * 50)
    
    # Initialize loader
    loader = PubTabNetDataLoader()
    
    # Test loading different splits
    for split in ['train', 'val', 'test']:
        print(f"\n🔍 Loading {split} split...")
        try:
            dataset = loader.load_split(split, max_samples=5)
            print(f"✅ Loaded {len(dataset.samples)} samples")
            
            # Validate dataset
            stats = loader.validate_dataset(dataset)
            print(f"   📈 Dataset stats: {stats}")
            
            # Show sample details
            if dataset.samples:
                sample = dataset.samples[0]
                print(f"   📝 Sample: {sample.table_id}")
                print(f"   🖼️ Image: {sample.image_path}")
                print(f"   📏 Dimensions: {sample.image_width}x{sample.image_height}")
                print(f"   📄 HTML length: {len(sample.html_annotation)}")
                print(f"   📦 Bboxes: {len(sample.cell_bboxes)}")
                
        except Exception as e:
            print(f"❌ Error loading {split}: {e}")
    
    print("\n🎉 Data loader test completed!")


if __name__ == "__main__":
    main()
