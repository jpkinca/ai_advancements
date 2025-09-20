#!/usr/bin/env python3
"""
CLIP Fine-Tuning Script for Chart Pattern Recognition

This module implements fine-tuning of CLIP models for chart pattern recognition
using contrastive learning with hard negative mining and mixed precision training.

Features:
- Contrastive learning with image-text pairs
- Hard negative mining for better training
- Mixed precision training (FP16)
- Gradient checkpointing for memory efficiency
- Comprehensive logging and checkpointing
- Evaluation during training
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChartDataset(Dataset):
    """Dataset for chart images and text pairs"""

    def __init__(self, data: List[Dict[str, Any]], processor):
        self.data = data
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Load and process image
        image_path = item['image_path']
        # Handle both absolute and relative paths
        if not os.path.isabs(image_path):
            # Construct absolute path if relative
            image_path = os.path.join(os.getcwd(), image_path)
        
        image = Image.open(image_path).convert('RGB')

        # Process text
        text = item['text']

        # Process image and text separately to avoid batching issues
        image_inputs = self.processor(
            images=image,
            return_tensors="pt"
        )
        
        text_inputs = self.processor(
            text=[text],  # Note: wrap in list for proper processing
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77  # CLIP text limit
        )

        # Extract tensors and remove batch dimension
        return {
            'pixel_values': image_inputs['pixel_values'].squeeze(0),
            'input_ids': text_inputs['input_ids'].squeeze(0),
            'attention_mask': text_inputs['attention_mask'].squeeze(0)
        }

class CLIPFineTuner:
    """
    Fine-tunes CLIP model for chart pattern recognition
    """

    def __init__(self,
                 model_name: str = "openai/clip-vit-base-patch32",
                 output_dir: str = "vlm/models",
                 device: Optional[str] = None):
        """
        Initialize the fine-tuner

        Args:
            model_name: Base CLIP model to fine-tune
            output_dir: Directory to save checkpoints and logs
            device: Device for training
        """
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize model and processor
        self.model = None
        self.processor = None
        self._load_model()

        logger.info(f"[INIT] CLIP Fine-Tuner ({model_name} on {self.device})")

    def _load_model(self):
        """Load CLIP model and processor"""
        try:
            from transformers import CLIPProcessor, CLIPModel

            self.model = CLIPModel.from_pretrained(self.model_name)
            self.processor = CLIPProcessor.from_pretrained(self.model_name)

            # Move to device
            self.model.to(self.device)

            logger.info(f"[SUCCESS] Loaded CLIP model: {self.model_name}")

        except ImportError as e:
            raise RuntimeError("transformers library required for CLIP fine-tuning") from e
        except Exception as e:
            raise RuntimeError(f"Failed to load CLIP model: {e}") from e

    def prepare_data(self, dataset_path: str) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Prepare train/val/test dataloaders from dataset

        Args:
            dataset_path: Path to dataset directory

        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        from vlm.dataset_builder import VLMDatasetBuilder

        # Load dataset
        builder = VLMDatasetBuilder()
        dataset = builder.load_dataset(dataset_path)

        # Custom collate function to handle variable text lengths
        def collate_fn(batch):
            # Extract images and texts separately
            images = [item['pixel_values'] for item in batch]
            texts = [item['input_ids'] for item in batch]
            attention_masks = [item['attention_mask'] for item in batch]
            
            # Stack images (they should be the same size)
            pixel_values = torch.stack(images)
            
            # Pad text sequences to the same length
            max_len = max(text.size(0) for text in texts)
            padded_texts = []
            padded_masks = []
            
            for text, mask in zip(texts, attention_masks):
                # Pad to max length
                pad_len = max_len - text.size(0)
                if pad_len > 0:
                    text = torch.cat([text, torch.zeros(pad_len, dtype=text.dtype)])
                    mask = torch.cat([mask, torch.zeros(pad_len, dtype=mask.dtype)])
                padded_texts.append(text)
                padded_masks.append(mask)
            
            input_ids = torch.stack(padded_texts)
            attention_mask = torch.stack(padded_masks)
            
            return {
                'pixel_values': pixel_values,
                'input_ids': input_ids,
                'attention_mask': attention_mask
            }

        # Prepare data for each split
        dataloaders = {}
        for split_name in ['train', 'validation', 'test']:
            split_data = dataset['splits'].get(split_name, [])

            # Fix path issues - use the correct image paths
            for item in split_data:
                # The image_path in metadata should be absolute or relative to cwd
                image_path = item['image_path']
                if not os.path.isabs(image_path):
                    # If relative, ensure it's correct relative to current working directory
                    if not os.path.exists(image_path):
                        # Try with the filename in the images directory
                        potential_path = os.path.join(
                            dataset_path, split_name, 'images', item['image_filename']
                        )
                        if os.path.exists(potential_path):
                            item['image_path'] = potential_path

            # Create dataset and dataloader
            split_dataset = ChartDataset(split_data, self.processor)
            dataloaders[split_name] = DataLoader(
                split_dataset,
                batch_size=4,  # Reduced batch size for stability
                shuffle=(split_name == 'train'),
                num_workers=0,  # Set to 0 to avoid multiprocessing issues
                pin_memory=False,  # Disable pin_memory since no GPU acceleration
                collate_fn=collate_fn  # Use custom collate function
            )

        return dataloaders['train'], dataloaders['validation'], dataloaders['test']

    def train(self,
              train_loader: DataLoader,
              val_loader: DataLoader,
              num_epochs: int = 10,
              learning_rate: float = 5e-6,
              weight_decay: float = 0.01,
              warmup_steps: int = 100,
              save_steps: int = 500,
              eval_steps: int = 100,
              use_mixed_precision: bool = True,
              gradient_checkpointing: bool = True):
        """
        Fine-tune the CLIP model

        Args:
            train_loader: Training dataloader
            val_loader: Validation dataloader
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            weight_decay: Weight decay for regularization
            warmup_steps: Number of warmup steps
            save_steps: Save checkpoint every N steps
            eval_steps: Evaluate every N steps
            use_mixed_precision: Use mixed precision training
            gradient_checkpointing: Use gradient checkpointing
        """
        # Set up optimizer
        optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Set up scheduler
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs * len(train_loader))

        # Enable gradient checkpointing if requested
        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        # Set up mixed precision
        scaler = torch.cuda.amp.GradScaler() if use_mixed_precision and self.device == 'cuda' else None

        # Training loop
        global_step = 0
        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            logger.info(f"[TRAIN] Starting epoch {epoch + 1}/{num_epochs}")

            # Training phase
            self.model.train()
            train_loss = 0.0
            train_steps = 0

            train_progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
            for batch in train_progress:
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Forward pass with mixed precision
                if scaler:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(**batch)
                        # Calculate contrastive loss manually since CLIP doesn't return loss by default
                        logits_per_image = outputs.logits_per_image
                        logits_per_text = outputs.logits_per_text
                        
                        # Create labels for contrastive learning (diagonal should be positive)
                        batch_size = logits_per_image.shape[0]
                        labels = torch.arange(batch_size, device=self.device)
                        
                        # Calculate cross-entropy loss for both directions
                        loss_img = nn.CrossEntropyLoss()(logits_per_image, labels)
                        loss_txt = nn.CrossEntropyLoss()(logits_per_text, labels)
                        loss = (loss_img + loss_txt) / 2

                    # Backward pass
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = self.model(**batch)
                    # Calculate contrastive loss manually since CLIP doesn't return loss by default
                    logits_per_image = outputs.logits_per_image
                    logits_per_text = outputs.logits_per_text
                    
                    # Create labels for contrastive learning (diagonal should be positive)
                    batch_size = logits_per_image.shape[0]
                    labels = torch.arange(batch_size, device=self.device)
                    
                    # Calculate cross-entropy loss for both directions
                    loss_img = nn.CrossEntropyLoss()(logits_per_image, labels)
                    loss_txt = nn.CrossEntropyLoss()(logits_per_text, labels)
                    loss = (loss_img + loss_txt) / 2
                    
                    loss.backward()
                    optimizer.step()

                optimizer.zero_grad()
                scheduler.step()

                train_loss += loss.item()
                train_steps += 1
                global_step += 1

                # Update progress bar
                train_progress.set_postfix({'loss': f"{loss.item():.4f}"})

                # Evaluation
                if global_step % eval_steps == 0 and val_loader:
                    val_loss = self._evaluate(val_loader)
                    logger.info(f"[EVAL] Step {global_step}: Train Loss = {train_loss/train_steps:.4f}, Val Loss = {val_loss:.4f}")

                    # Save best model
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        self._save_checkpoint(f"best_model", global_step, val_loss)

                # Save checkpoint
                if global_step % save_steps == 0:
                    self._save_checkpoint(f"checkpoint_{global_step}", global_step, train_loss/train_steps)

            # End of epoch
            avg_train_loss = train_loss / train_steps
            logger.info(f"[EPOCH] {epoch + 1}/{num_epochs} completed. Avg Train Loss: {avg_train_loss:.4f}")

        # Save final model
        self._save_checkpoint("final_model", global_step, avg_train_loss)

        logger.info("[TRAIN] Fine-tuning completed!")

    def _evaluate(self, val_loader: DataLoader) -> float:
        """Evaluate model on validation set"""
        self.model.eval()
        val_loss = 0.0
        val_steps = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                
                # Calculate contrastive loss manually
                logits_per_image = outputs.logits_per_image
                logits_per_text = outputs.logits_per_text
                
                batch_size = logits_per_image.shape[0]
                labels = torch.arange(batch_size, device=self.device)
                
                loss_img = nn.CrossEntropyLoss()(logits_per_image, labels)
                loss_txt = nn.CrossEntropyLoss()(logits_per_text, labels)
                loss = (loss_img + loss_txt) / 2
                
                val_loss += loss.item()
                val_steps += 1

        return val_loss / val_steps if val_steps > 0 else 0.0

    def _save_checkpoint(self, name: str, step: int, loss: float):
        """Save model checkpoint"""
        checkpoint_path = self.output_dir / f"{name}.pt"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'step': step,
            'loss': loss,
            'model_name': self.model_name,
            'timestamp': time.time()
        }, checkpoint_path)

        logger.info(f"[SAVE] Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"[LOAD] Checkpoint loaded: {checkpoint_path}")
        return checkpoint

    def add_hard_negatives(self, dataset_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Add hard negative samples to dataset for better contrastive learning

        Args:
            dataset_data: Original dataset samples

        Returns:
            Dataset with hard negatives added
        """
        enhanced_data = dataset_data.copy()

        # Group by symbol for mining hard negatives
        symbol_groups = {}
        for item in dataset_data:
            symbol = item['metadata']['symbol']
            if symbol not in symbol_groups:
                symbol_groups[symbol] = []
            symbol_groups[symbol].append(item)

        hard_negatives = []
        for symbol, items in symbol_groups.items():
            if len(items) < 2:
                continue

            for i, item in enumerate(items):
                # Create hard negative by pairing with similar but different pattern
                for j, other_item in enumerate(items):
                    if i == j:
                        continue

                    # Check if patterns are different enough
                    patterns_i = set(p['type'] for p in item['metadata']['patterns_detected'])
                    patterns_j = set(p['type'] for p in other_item['metadata']['patterns_detected'])

                    if patterns_i != patterns_j:
                        # Create hard negative
                        hard_negative = item.copy()
                        hard_negative['text'] = other_item['text']
                        hard_negative['is_hard_negative'] = True
                        hard_negative['original_id'] = item['id']
                        hard_negatives.append(hard_negative)

        # Limit hard negatives to 30% of original dataset
        max_hard_negatives = int(len(dataset_data) * 0.3)
        hard_negatives = hard_negatives[:max_hard_negatives]

        enhanced_data.extend(hard_negatives)
        logger.info(f"[HARD_NEGATIVES] Added {len(hard_negatives)} hard negative samples")

        return enhanced_data

# Utility functions
def run_fine_tuning(dataset_path: str = 'vlm/datasets/sample_trading_charts_v1.0',
                    model_name: str = "openai/clip-vit-base-patch32",
                    num_epochs: int = 5):
    """Run CLIP fine-tuning on dataset"""

    # Initialize fine-tuner
    fine_tuner = CLIPFineTuner(model_name)

    # Prepare data
    train_loader, val_loader, test_loader = fine_tuner.prepare_data(dataset_path)

    # Train model
    fine_tuner.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        learning_rate=5e-6,
        use_mixed_precision=True,
        gradient_checkpointing=True
    )

    logger.info("Fine-tuning completed successfully!")
    return fine_tuner

def evaluate_fine_tuned_model(model_path: str, test_loader: DataLoader):
    """Evaluate fine-tuned model on test set"""
    # Load model
    fine_tuner = CLIPFineTuner()
    checkpoint = fine_tuner.load_checkpoint(model_path)

    # Evaluate
    test_loss = fine_tuner._evaluate(test_loader)
    logger.info(f"[EVAL] Test Loss: {test_loss:.4f}")

    return test_loss

if __name__ == "__main__":
    # Run fine-tuning
    fine_tuner = run_fine_tuning()

    # Example: Evaluate on test set
    # test_loss = evaluate_fine_tuned_model('vlm/models/best_model.pt', test_loader)