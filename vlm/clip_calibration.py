#!/usr/bin/env python3
"""
CLIP Model Calibration Module

This module implements calibration techniques for CLIP models to improve
confidence scores and prediction reliability for chart pattern recognition.

Features:
- Temperature scaling for logit calibration
- Platt scaling for binary classification
- Expected calibration error (ECE) computation
- Reliability diagrams
- Cross-validation for calibration parameters
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TemperatureScaler(nn.Module):
    """
    Temperature scaling for calibrating CLIP logits
    """

    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(temperature))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply temperature scaling to logits"""
        return logits / self.temperature

    def fit(self, logits: torch.Tensor, labels: torch.Tensor,
            lr: float = 0.01, max_iter: int = 1000):
        """
        Fit temperature parameter using validation data

        Args:
            logits: Model logits (before softmax)
            labels: True labels
            lr: Learning rate for optimization
            max_iter: Maximum iterations
        """
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            scaled_logits = self.forward(logits)
            loss = F.cross_entropy(scaled_logits, labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        logger.info(f"[CALIBRATION] Temperature fitted: {self.temperature.item():.4f}")

class PlattScaler(nn.Module):
    """
    Platt scaling for binary classification calibration
    """

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply Platt scaling to logits"""
        return self.linear(logits.unsqueeze(-1)).squeeze(-1)

    def fit(self, logits: torch.Tensor, labels: torch.Tensor,
            lr: float = 0.01, epochs: int = 100):
        """
        Fit Platt scaling parameters

        Args:
            logits: Model logits
            labels: Binary labels
            lr: Learning rate
            epochs: Number of training epochs
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss()

        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.forward(logits)
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()

        logger.info("[CALIBRATION] Platt scaling fitted")

class CLIPCalibrator:
    """
    Calibrates CLIP model predictions for better confidence scores
    """

    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize calibrator

        Args:
            model_path: Path to fine-tuned CLIP model
            device: Device for computation
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.processor = None
        self.temperature_scaler = None
        self.platt_scaler = None

        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path: str):
        """Load fine-tuned CLIP model"""
        try:
            from transformers import CLIPProcessor, CLIPModel
            from vlm.clip_fine_tune import CLIPFineTuner

            # Load base model
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

            # Load fine-tuned weights
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()

            logger.info(f"[LOAD] Fine-tuned CLIP model loaded from {model_path}")

        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}") from e

    def extract_logits_and_labels(self, dataloader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract logits and labels from dataloader for calibration

        Args:
            dataloader: DataLoader with validation/test data

        Returns:
            Tuple of (logits, labels)
        """
        all_logits = []
        all_labels = []

        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Extracting logits"):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Get model outputs
                outputs = self.model(**batch)

                # For CLIP, we need to compute logits manually
                # This assumes we're doing image-text matching
                image_embeds = outputs.image_embeds
                text_embeds = outputs.text_embeds

                # Compute similarity logits (each image with each text in batch)
                batch_size = len(image_embeds)
                logits = torch.matmul(image_embeds, text_embeds.t()) * self.model.logit_scale.exp()

                # For calibration, we need ground truth labels
                # This assumes diagonal matching (image i matches text i)
                labels = torch.arange(batch_size, device=self.device)

                # Store diagonal elements as the matching scores for calibration
                # For temperature scaling with cross_entropy, we need 2D logits [negative_class, positive_class]
                diagonal_scores = torch.diag(logits)
                # Create binary logits: [negative_score, positive_score]
                binary_logits = torch.stack([-diagonal_scores, diagonal_scores], dim=1)
                # Create binary labels: 1 for matched pairs (positive class)
                binary_labels = torch.ones(batch_size, device=self.device, dtype=torch.long)

                all_logits.append(binary_logits)
                all_labels.append(binary_labels)

        # Concatenate all batches - now all tensors have consistent shapes
        logits = torch.cat(all_logits, dim=0)
        labels = torch.cat(all_labels, dim=0)

        return logits, labels

    def calibrate_temperature(self, val_dataloader: DataLoader,
                            lr: float = 0.01, max_iter: int = 1000):
        """
        Calibrate model using temperature scaling

        Args:
            val_dataloader: Validation dataloader
            lr: Learning rate
            max_iter: Maximum iterations
        """
        logger.info("[CALIBRATION] Starting temperature scaling...")

        # Extract logits and labels
        logits, labels = self.extract_logits_and_labels(val_dataloader)

        # Initialize temperature scaler
        self.temperature_scaler = TemperatureScaler()

        # Fit temperature
        self.temperature_scaler.fit(logits, labels, lr=lr, max_iter=max_iter)

        logger.info("[CALIBRATION] Temperature scaling completed")

    def calibrate_platt(self, val_dataloader: DataLoader,
                       lr: float = 0.01, epochs: int = 100):
        """
        Calibrate model using Platt scaling for binary classification

        Args:
            val_dataloader: Validation dataloader
            lr: Learning rate
            epochs: Number of epochs
        """
        logger.info("[CALIBRATION] Starting Platt scaling...")

        # Extract logits and labels
        logits, labels = self.extract_logits_and_labels(val_dataloader)

        # Convert to binary classification (match vs no-match)
        # For simplicity, consider diagonal elements as positive
        binary_logits = torch.diag(logits).unsqueeze(-1)
        binary_labels = torch.ones_like(binary_logits.squeeze())

        # Initialize Platt scaler
        self.platt_scaler = PlattScaler()

        # Fit Platt scaling
        self.platt_scaler.fit(binary_logits, binary_labels, lr=lr, epochs=epochs)

        logger.info("[CALIBRATION] Platt scaling completed")

    def predict_calibrated(self, image: Any, texts: List[str],
                          calibration_method: str = "temperature") -> Dict[str, Any]:
        """
        Make calibrated predictions

        Args:
            image: Input image (PIL Image or path)
            texts: List of text descriptions
            calibration_method: "temperature" or "platt"

        Returns:
            Dictionary with predictions and confidence scores
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Process inputs
        if isinstance(image, str):
            from PIL import Image
            image = Image.open(image).convert('RGB')

        inputs = self.processor(
            text=texts,
            images=[image] * len(texts),
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)

            # Compute similarity logits
            image_embeds = outputs.image_embeds
            text_embeds = outputs.text_embeds
            logits = torch.matmul(image_embeds, text_embeds.t()) * self.model.logit_scale.exp()

            # Apply calibration
            if calibration_method == "temperature" and self.temperature_scaler:
                calibrated_logits = self.temperature_scaler(logits)
            elif calibration_method == "platt" and self.platt_scaler:
                # For Platt, we need to calibrate each logit
                calibrated_logits = torch.zeros_like(logits)
                for i in range(logits.shape[0]):
                    for j in range(logits.shape[1]):
                        calibrated_logits[i, j] = self.platt_scaler(logits[i, j])
            else:
                calibrated_logits = logits

            # Convert to probabilities
            probs = F.softmax(calibrated_logits, dim=-1)

            # Get predictions
            predictions = torch.argmax(probs, dim=-1)
            confidences = torch.max(probs, dim=-1)[0]

        return {
            'predictions': predictions.cpu().numpy(),
            'confidences': confidences.cpu().numpy(),
            'probabilities': probs.cpu().numpy(),
            'texts': texts
        }

    def compute_ece(self, logits: torch.Tensor, labels: torch.Tensor,
                   n_bins: int = 10) -> float:
        """
        Compute Expected Calibration Error (ECE)

        Args:
            logits: Model logits
            labels: True labels
            n_bins: Number of bins for calibration

        Returns:
            Expected calibration error
        """
        probs = F.softmax(logits, dim=-1)
        confidences, predictions = torch.max(probs, dim=-1)

        # Convert to numpy for easier binning
        confidences = confidences.cpu().numpy()
        predictions = predictions.cpu().numpy()
        labels = labels.cpu().numpy()

        # Compute accuracy for each bin
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0

        for i in range(n_bins):
            bin_start = bin_boundaries[i]
            bin_end = bin_boundaries[i + 1]

            # Find samples in this confidence bin
            bin_mask = (confidences >= bin_start) & (confidences < bin_end)
            if np.sum(bin_mask) == 0:
                continue

            # Compute accuracy and average confidence in this bin
            bin_accuracy = accuracy_score(labels[bin_mask], predictions[bin_mask])
            bin_confidence = np.mean(confidences[bin_mask])
            bin_size = np.sum(bin_mask)

            # Add to ECE
            ece += (bin_size / len(confidences)) * abs(bin_accuracy - bin_confidence)

        return ece

    def plot_reliability_diagram(self, logits: torch.Tensor, labels: torch.Tensor,
                               save_path: Optional[str] = None, n_bins: int = 10):
        """
        Plot reliability diagram

        Args:
            logits: Model logits
            labels: True labels
            save_path: Path to save plot
            n_bins: Number of bins
        """
        probs = F.softmax(logits, dim=-1)
        confidences, predictions = torch.max(probs, dim=-1)

        # Convert to numpy
        confidences = confidences.cpu().numpy()
        predictions = predictions.cpu().numpy()
        labels = labels.cpu().numpy()

        # Compute bin statistics
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_accuracies = []
        bin_confidences = []
        bin_sizes = []

        for i in range(n_bins):
            bin_start = bin_boundaries[i]
            bin_end = bin_boundaries[i + 1]

            bin_mask = (confidences >= bin_start) & (confidences < bin_end)
            if np.sum(bin_mask) == 0:
                bin_accuracies.append(0)
                bin_confidences.append((bin_start + bin_end) / 2)
                bin_sizes.append(0)
                continue

            bin_accuracy = accuracy_score(labels[bin_mask], predictions[bin_mask])
            bin_confidence = np.mean(confidences[bin_mask])

            bin_accuracies.append(bin_accuracy)
            bin_confidences.append(bin_confidence)
            bin_sizes.append(np.sum(bin_mask))

        # Plot
        plt.figure(figsize=(8, 6))
        plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
        plt.plot(bin_confidences, bin_accuracies, 'bo-', label='Model calibration')

        # Add point sizes based on bin size
        for i, (conf, acc, size) in enumerate(zip(bin_confidences, bin_accuracies, bin_sizes)):
            plt.scatter(conf, acc, s=size*10, alpha=0.5)

        plt.xlabel('Confidence')
        plt.ylabel('Accuracy')
        plt.title('Reliability Diagram')
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"[PLOT] Reliability diagram saved to {save_path}")

        plt.show()

    def cross_validate_calibration(self, dataloader: DataLoader,
                                 method: str = "temperature",
                                 n_splits: int = 5) -> Dict[str, List[float]]:
        """
        Cross-validate calibration parameters

        Args:
            dataloader: DataLoader with data
            method: Calibration method ("temperature" or "platt")
            n_splits: Number of cross-validation splits

        Returns:
            Dictionary with ECE scores for each fold
        """
        # Extract all data
        logits, labels = self.extract_logits_and_labels(dataloader)

        # Convert to numpy for sklearn
        logits_np = logits.cpu().numpy()
        labels_np = labels.cpu().numpy()

        # Stratified k-fold
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        ece_scores = []
        calibrated_ece_scores = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(logits_np, labels_np)):
            logger.info(f"[CV] Fold {fold + 1}/{n_splits}")

            # Split data
            train_logits = torch.tensor(logits_np[train_idx], device=self.device)
            train_labels = torch.tensor(labels_np[train_idx], device=self.device)
            val_logits = torch.tensor(logits_np[val_idx], device=self.device)
            val_labels = torch.tensor(labels_np[val_idx], device=self.device)

            # Compute ECE before calibration
            ece_before = self.compute_ece(val_logits, val_labels)
            ece_scores.append(ece_before)

            # Calibrate on training fold
            if method == "temperature":
                temp_scaler = TemperatureScaler()
                temp_scaler.fit(train_logits, train_labels)
                calibrated_val_logits = temp_scaler(val_logits)
            elif method == "platt":
                platt_scaler = PlattScaler()
                platt_scaler.fit(train_logits, train_labels)
                calibrated_val_logits = platt_scaler(val_logits)
            else:
                calibrated_val_logits = val_logits

            # Compute ECE after calibration
            ece_after = self.compute_ece(calibrated_val_logits, val_labels)
            calibrated_ece_scores.append(ece_after)

            logger.info(f"[CALIBRATE] ECE after calibration: {ece_after:.4f}")
            
        return {
            'ece_before_calibration': ece_scores,
            'ece_after_calibration': calibrated_ece_scores
        }

# Utility functions
def calibrate_clip_model(model_path: str, val_dataloader: DataLoader,
                        method: str = "temperature") -> CLIPCalibrator:
    """
    Calibrate a fine-tuned CLIP model

    Args:
        model_path: Path to fine-tuned model
        val_dataloader: Validation dataloader
        method: Calibration method

    Returns:
        Calibrated CLIPCalibrator instance
    """
    calibrator = CLIPCalibrator(model_path)

    if method == "temperature":
        calibrator.calibrate_temperature(val_dataloader)
    elif method == "platt":
        calibrator.calibrate_platt(val_dataloader)
    else:
        raise ValueError(f"Unknown calibration method: {method}")

    return calibrator

def evaluate_calibration(calibrator: CLIPCalibrator, test_dataloader: DataLoader,
                        save_plot: bool = True) -> Dict[str, float]:
    """
    Evaluate calibration performance

    Args:
        calibrator: Calibrated model
        test_dataloader: Test dataloader
        save_plot: Whether to save reliability diagram

    Returns:
        Dictionary with evaluation metrics
    """
    # Extract test data
    logits, labels = calibrator.extract_logits_and_labels(test_dataloader)

    # Compute ECE before calibration
    ece_before = calibrator.compute_ece(logits, labels)

    # Apply calibration
    if calibrator.temperature_scaler:
        calibrated_logits = calibrator.temperature_scaler(logits)
        method = "temperature"
    elif calibrator.platt_scaler:
        calibrated_logits = calibrator.platt_scaler(logits)
        method = "platt"
    else:
        calibrated_logits = logits
        method = "none"

    # Compute ECE after calibration
    ece_after = calibrator.compute_ece(calibrated_logits, labels)

    # Plot reliability diagram
    if save_plot:
        plot_path = f"vlm/calibration/reliability_diagram_{method}.png"
        calibrator.plot_reliability_diagram(calibrated_logits, labels, plot_path)

    results = {
        'ece_before': ece_before,
        'ece_after': ece_after,
        'ece_improvement': ece_before - ece_after,
        'calibration_method': method
    }

    logger.info(f"[EVAL] Calibration Results:")
    logger.info(f"  ECE Before: {ece_before:.4f}")
    logger.info(f"  ECE After: {ece_after:.4f}")
    logger.info(f"  Improvement: {results['ece_improvement']:.4f}")

    return results

if __name__ == "__main__":
    # Example usage
    from vlm.clip_fine_tune import CLIPFineTuner

    # Load fine-tuned model
    model_path = "vlm/models/best_model.pt"
    calibrator = calibrate_clip_model(model_path, None, method="temperature")  # Pass actual dataloader

    # Evaluate calibration
    # results = evaluate_calibration(calibrator, test_dataloader)