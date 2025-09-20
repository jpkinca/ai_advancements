#!/usr/bin/env python3
"""
VLM Dataset Builder

This module builds paired image-text datasets for Vision-Language Model training,
combining chart images with descriptive labels for contrastive learning.

Features:
- Image-text pairing with metadata
- Dataset versioning and persistence
- Automatic train/val/test splits
- Hard negative mining for better training
- Integration with chart/label generators
- HuggingFace dataset format support
"""

import os
import json
import hashlib
import shutil
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VLMDatasetBuilder:
    """
    Builds and manages VLM training datasets from chart images and text labels
    """

    def __init__(self,
                 dataset_dir: str = 'vlm/datasets',
                 images_dir: str = 'vlm/charts',
                 labels_dir: str = 'vlm/labels'):
        """
        Initialize the dataset builder

        Args:
            dataset_dir: Directory to store built datasets
            images_dir: Directory containing chart images
            labels_dir: Directory containing label files
        """
        self.dataset_dir = Path(dataset_dir)
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)

        # Create directories
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"[INIT] VLM Dataset Builder (datasets: {dataset_dir})")

    def build_dataset(self,
                     name: str,
                     image_label_pairs: List[Tuple[str, Dict[str, Any]]],
                     version: str = 'v1.0',
                     val_split: float = 0.15,
                     test_split: float = 0.1,
                     hard_negatives: bool = True) -> str:
        """
        Build a complete VLM dataset from image-label pairs

        Args:
            name: Dataset name
            image_label_pairs: List of (image_path, label_dict) tuples
            version: Dataset version
            val_split: Validation split ratio
            test_split: Test split ratio
            hard_negatives: Whether to include hard negative samples

        Returns:
            Path to the built dataset directory
        """
        try:
            # Create dataset directory
            dataset_path = self.dataset_dir / f"{name}_{version}"
            dataset_path.mkdir(exist_ok=True)

            # Prepare data
            dataset_data = self._prepare_dataset_data(image_label_pairs)

            # Add hard negatives if requested
            if hard_negatives:
                dataset_data = self._add_hard_negatives(dataset_data)

            # Create splits
            train_data, val_data, test_data = self._create_splits(
                dataset_data, val_split, test_split
            )

            # Save splits
            splits = {
                'train': train_data,
                'validation': val_data,
                'test': test_data
            }

            for split_name, split_data in splits.items():
                split_path = dataset_path / split_name
                split_path.mkdir(exist_ok=True)

                # Save metadata
                metadata_path = split_path / 'metadata.jsonl'
                self._save_metadata(split_data, metadata_path)

                # Copy images
                self._copy_images(split_data, split_path / 'images')

            # Save dataset info
            dataset_info = {
                'name': name,
                'version': version,
                'created_at': datetime.now().isoformat(),
                'total_samples': len(dataset_data),
                'splits': {
                    'train': len(train_data),
                    'validation': len(val_data),
                    'test': len(test_data)
                },
                'hard_negatives': hard_negatives,
                'schema_version': '1.0'
            }

            with open(dataset_path / 'dataset_info.json', 'w') as f:
                json.dump(dataset_info, f, indent=2)

            logger.info(f"[SUCCESS] Built dataset: {dataset_path}")
            logger.info(f"  Total samples: {len(dataset_data)}")
            logger.info(f"  Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

            return str(dataset_path)

        except Exception as e:
            logger.error(f"[ERROR] Failed to build dataset: {e}")
            raise

    def build_dataset_from_samples(self,
                                 samples: List[Dict[str, Any]],
                                 output_path: str,
                                 version: str = 'v1.0',
                                 val_split: float = 0.15,
                                 test_split: float = 0.1) -> str:
        """
        Build a VLM dataset from prepared training samples

        Args:
            samples: List of training samples with image_path and label data
            output_path: Output directory path for the dataset
            version: Dataset version
            val_split: Validation split ratio
            test_split: Test split ratio

        Returns:
            Path to the built dataset directory
        """
        try:
            # Extract name from output path
            name = Path(output_path).name

            # Convert samples to image_label_pairs format
            image_label_pairs = []
            for sample in samples:
                image_path = sample['image_path']
                label_dict = {
                    'symbol': sample.get('symbol', ''),
                    'timeframe': sample.get('timeframe', ''),
                    'base_description': sample.get('text', sample.get('description', '')),
                    'augmented_description': sample.get('text', sample.get('description', '')),
                    'confidence_score': sample.get('confidence_score', 0.8),
                    'patterns_detected': sample.get('patterns_detected', []),
                    'generated_at': sample.get('generated_at', datetime.now().isoformat()),
                    'needs_review': sample.get('needs_review', False)
                }
                image_label_pairs.append((image_path, label_dict))

            # Use the existing build_dataset method
            return self.build_dataset(
                name=name,
                image_label_pairs=image_label_pairs,
                version=version,
                val_split=val_split,
                test_split=test_split,
                hard_negatives=False  # Skip hard negatives for simplicity
            )

        except Exception as e:
            logger.error(f"[ERROR] Failed to build dataset from samples: {e}")
            raise

    def _prepare_dataset_data(self, image_label_pairs: List[Tuple[str, Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Prepare dataset data from image-label pairs"""
        dataset_data = []

        for image_path, label_dict in image_label_pairs:
            # Verify image exists
            if not os.path.exists(image_path):
                logger.warning(f"[WARNING] Image not found: {image_path}")
                continue

            # Create dataset entry
            entry = {
                'image_path': image_path,
                'image_filename': os.path.basename(image_path),
                'text': label_dict.get('augmented_description', label_dict.get('base_description', '')),
                'metadata': {
                    'symbol': label_dict.get('symbol', ''),
                    'timeframe': label_dict.get('timeframe', ''),
                    'confidence_score': label_dict.get('confidence_score', 0.0),
                    'patterns_detected': label_dict.get('patterns_detected', []),
                    'generated_at': label_dict.get('generated_at', ''),
                    'needs_review': label_dict.get('needs_review', False)
                },
                'id': self._generate_sample_id(image_path, label_dict)
            }

            dataset_data.append(entry)

        return dataset_data

    def _add_hard_negatives(self, dataset_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add hard negative samples to improve training"""
        enhanced_data = dataset_data.copy()

        # Group by symbol for negative mining
        symbol_groups = {}
        for entry in dataset_data:
            symbol = entry['metadata']['symbol']
            if symbol not in symbol_groups:
                symbol_groups[symbol] = []
            symbol_groups[symbol].append(entry)

        # Create hard negatives (same symbol, different patterns)
        hard_negatives = []
        for symbol, entries in symbol_groups.items():
            if len(entries) < 2:
                continue

            for i, entry in enumerate(entries):
                # Find entries with different patterns
                for j, other_entry in enumerate(entries):
                    if i == j:
                        continue

                    # Check if patterns are different
                    patterns_i = set(p['type'] for p in entry['metadata']['patterns_detected'])
                    patterns_j = set(p['type'] for p in other_entry['metadata']['patterns_detected'])

                    if patterns_i != patterns_j:
                        # Create hard negative by swapping text
                        hard_negative = entry.copy()
                        hard_negative['text'] = other_entry['text']
                        hard_negative['is_hard_negative'] = True
                        hard_negative['original_id'] = entry['id']
                        hard_negatives.append(hard_negative)

        # Add up to 20% hard negatives
        max_hard_negatives = int(len(dataset_data) * 0.2)
        hard_negatives = hard_negatives[:max_hard_negatives]

        enhanced_data.extend(hard_negatives)
        logger.info(f"[HARD_NEGATIVES] Added {len(hard_negatives)} hard negative samples")

        return enhanced_data

    def _create_splits(self, dataset_data: List[Dict[str, Any]],
                      val_split: float, test_split: float) -> Tuple[List, List, List]:
        """Create train/validation/test splits"""
        # Shuffle data
        data_df = pd.DataFrame(dataset_data)
        data_df = data_df.sample(frac=1, random_state=42).reset_index(drop=True)

        # Calculate split sizes
        n_total = len(data_df)
        n_test = int(n_total * test_split)
        n_val = int(n_total * val_split)
        n_train = n_total - n_test - n_val

        # Create splits
        train_data = data_df[:n_train].to_dict('records')
        val_data = data_df[n_train:n_train + n_val].to_dict('records')
        test_data = data_df[n_train + n_val:].to_dict('records')

        return train_data, val_data, test_data

    def _generate_sample_id(self, image_path: str, label_dict: Dict[str, Any]) -> str:
        """Generate unique sample ID"""
        content = f"{image_path}_{label_dict.get('symbol', '')}_{label_dict.get('generated_at', '')}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def _save_metadata(self, data: List[Dict[str, Any]], output_path: Path):
        """Save metadata in JSONL format"""
        with open(output_path, 'w') as f:
            for entry in data:
                json.dump(entry, f)
                f.write('\n')

    def _copy_images(self, data: List[Dict[str, Any]], output_dir: Path):
        """Copy images to dataset directory"""
        output_dir.mkdir(exist_ok=True)

        for entry in data:
            src_path = entry['image_path']
            dst_path = output_dir / entry['image_filename']

            if os.path.exists(src_path):
                shutil.copy2(src_path, dst_path)
            else:
                logger.warning(f"[WARNING] Source image not found: {src_path}")

    def load_dataset(self, dataset_path: str) -> Dict[str, Any]:
        """
        Load a previously built dataset

        Args:
            dataset_path: Path to dataset directory

        Returns:
            Dataset information and data
        """
        dataset_path_obj = Path(dataset_path)

        # Load dataset info
        info_path = dataset_path_obj / 'dataset_info.json'
        with open(info_path, 'r') as f:
            dataset_info = json.load(f)

        # Load splits
        splits = {}
        for split_name in ['train', 'validation', 'test']:
            split_path = dataset_path_obj / split_name
            if split_path.exists():
                metadata_path = split_path / 'metadata.jsonl'
                if metadata_path.exists():
                    split_data = []
                    with open(metadata_path, 'r') as f:
                        for line in f:
                            split_data.append(json.loads(line.strip()))
                    splits[split_name] = split_data

        return {
            'info': dataset_info,
            'splits': splits,
            'path': str(dataset_path_obj)
        }

    def export_to_huggingface(self, dataset_path: str, hf_dataset_name: str) -> str:
        """
        Export dataset to HuggingFace format

        Args:
            dataset_path: Local dataset path
            hf_dataset_name: HuggingFace dataset name

        Returns:
            HuggingFace dataset path
        """
        try:
            # Optional import for HuggingFace datasets
            import datasets  # type: ignore
            Dataset = datasets.Dataset  # type: ignore
            DatasetDict = datasets.DatasetDict  # type: ignore

            dataset = self.load_dataset(dataset_path)

            # Convert to HuggingFace format
            hf_splits = {}
            for split_name, split_data in dataset['splits'].items():
                # Convert relative paths to absolute
                for entry in split_data:
                    entry['image_path'] = os.path.join(dataset_path, split_name, 'images', entry['image_filename'])

                hf_splits[split_name] = Dataset.from_list(split_data)

            hf_dataset = DatasetDict(hf_splits)

            # Save locally (can be pushed to Hub later)
            hf_path = f"./hf_datasets/{hf_dataset_name}"
            os.makedirs(os.path.dirname(hf_path), exist_ok=True)
            hf_dataset.save_to_disk(hf_path)

            logger.info(f"[EXPORT] Exported to HuggingFace format: {hf_path}")
            return hf_path

        except ImportError:
            logger.error("[ERROR] datasets library not available for HuggingFace export")
            return ""
        except Exception as e:
            logger.error(f"[ERROR] Failed to export to HuggingFace: {e}")
            return ""

    def get_dataset_stats(self, dataset_path: str) -> Dict[str, Any]:
        """Get comprehensive statistics for a dataset"""
        dataset = self.load_dataset(dataset_path)

        stats = {
            'total_samples': dataset['info']['total_samples'],
            'splits': dataset['info']['splits'],
            'symbols': {},
            'timeframes': {},
            'confidence_distribution': {'high': 0, 'medium': 0, 'low': 0},
            'pattern_types': {},
            'needs_review': 0
        }

        # Analyze all samples
        all_samples = []
        for split_data in dataset['splits'].values():
            all_samples.extend(split_data)

        for sample in all_samples:
            meta = sample['metadata']

            # Symbol distribution
            symbol = meta['symbol']
            stats['symbols'][symbol] = stats['symbols'].get(symbol, 0) + 1

            # Timeframe distribution
            timeframe = meta['timeframe']
            stats['timeframes'][timeframe] = stats['timeframes'].get(timeframe, 0) + 1

            # Confidence distribution
            confidence = meta['confidence_score']
            if confidence >= 0.8:
                stats['confidence_distribution']['high'] += 1
            elif confidence >= 0.6:
                stats['confidence_distribution']['medium'] += 1
            else:
                stats['confidence_distribution']['low'] += 1

            # Pattern types
            for pattern in meta['patterns_detected']:
                ptype = pattern['type']
                stats['pattern_types'][ptype] = stats['pattern_types'].get(ptype, 0) + 1

            # Review needed
            if meta.get('needs_review', False):
                stats['needs_review'] += 1

        return stats

# Utility functions
def create_sample_dataset():
    """Create a sample dataset for testing"""
    from chart_image_generator import ChartImageGenerator, create_sample_data
    from text_label_generator import TextLabelGenerator

    # Initialize components
    chart_gen = ChartImageGenerator()
    label_gen = TextLabelGenerator()
    dataset_builder = VLMDatasetBuilder()

    # Generate sample data
    symbols = ['AAPL', 'GOOGL', 'MSFT']
    timeframes = ['1D']
    overlays = [['sma_20', 'bb']]

    image_label_pairs = []

    for symbol in symbols:
        for timeframe in timeframes:
            # Create sample data
            sample_data = create_sample_data(symbol, 50)

            # Generate chart
            image_path, metadata = chart_gen.generate_chart_image(
                sample_data, symbol, timeframe, overlays[0]
            )

            # Generate label
            label = label_gen.generate_label(sample_data, symbol, timeframe, metadata)

            image_label_pairs.append((image_path, label))

    # Build dataset
    dataset_path = dataset_builder.build_dataset(
        name='sample_trading_charts',
        image_label_pairs=image_label_pairs,
        version='v1.0'
    )

    # Get stats
    stats = dataset_builder.get_dataset_stats(dataset_path)

    print(f"Created sample dataset: {dataset_path}")
    print("Dataset Statistics:")
    print(json.dumps(stats, indent=2))

    return dataset_path

if __name__ == "__main__":
    create_sample_dataset()