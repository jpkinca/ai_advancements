"""
Compressed Sensing for Trading

This module implements compressed sensing techniques for market analysis:
- Sparse representation of market features
- L1 regularization for feature selection
- High-frequency pattern compression
- Anomaly detection in price data
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from scipy import sparse, optimize
from sklearn.linear_model import Lasso, LassoCV, ElasticNet
from sklearn.decomposition import SparsePCA, DictionaryLearning
from sklearn.preprocessing import StandardScaler
from decimal import Decimal
from dataclasses import dataclass
from datetime import datetime
import logging

from ..core.data_structures import MarketData, TradingSignal
from ..core.base_classes import BaseTradingModel

logger = logging.getLogger(__name__)

@dataclass
class CompressedSensingConfig:
    """Configuration for compressed sensing analysis."""
    sparsity_level: float = 0.1  # Target sparsity level (0-1)
    l1_alpha: float = 0.01  # L1 regularization strength
    n_components: int = 50  # Number of dictionary components
    max_iter: int = 1000  # Maximum iterations for optimization
    tolerance: float = 1e-6  # Convergence tolerance
    anomaly_threshold: float = 2.0  # Anomaly detection threshold (in std deviations)

@dataclass 
class SparseFeature:
    """Represents a sparse feature extracted from market data."""
    feature_id: int
    weight: float
    description: str
    frequency_band: Optional[Tuple[float, float]] = None
    importance_score: float = 0.0

class MarketFeatureExtractor:
    """Extract features from market data for compressed sensing."""
    
    def __init__(self):
        self.feature_names = []
        self.scaler = StandardScaler()
        
        logger.info("[SUCCESS] Market Feature Extractor initialized")
    
    def extract_features(self, market_data: List[MarketData], window_size: int = 50) -> np.ndarray:
        """Extract comprehensive features from market data."""
        if len(market_data)  2 else 0.0,  # Skewness
                self._safe_kurtosis(returns),  # Kurtosis
                np.min(returns),   # Min return
                np.max(returns),   # Max return
                np.percentile(returns, 25),  # 25th percentile
                np.percentile(returns, 75),  # 75th percentile
            ])
            
            # Technical indicators
            window_features.extend(self._calculate_technical_indicators(window_prices, window_volumes))
            
            # Frequency domain features
            window_features.extend(self._calculate_frequency_features(window_prices))
            
            # High-frequency patterns
            window_features.extend(self._calculate_pattern_features(window_prices, window_highs, window_lows))
            
            features_list.append(window_features)
        
        # Convert to numpy array
        features_array = np.array(features_list)
        
        # Handle NaN and infinite values
        features_array = np.nan_to_num(features_array, nan=0.0, posinf=1e6, neginf=-1e6)
        
        logger.info(f"[SUCCESS] Extracted {features_array.shape[1]} features from {features_array.shape[0]} windows")
        return features_array
    
    def _safe_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis safely."""
        if len(data)  List[float]:
        """Calculate technical indicator features."""
        indicators = []
        
        # Moving averages
        if len(prices) >= 20:
            ma_5 = np.mean(prices[-5:])
            ma_10 = np.mean(prices[-10:])
            ma_20 = np.mean(prices[-20:])
            
            indicators.extend([
                prices[-1] / ma_5 - 1,   # Price relative to 5-day MA
                prices[-1] / ma_10 - 1,  # Price relative to 10-day MA
                prices[-1] / ma_20 - 1,  # Price relative to 20-day MA
                ma_5 / ma_20 - 1,        # Short MA relative to long MA
            ])
        else:
            indicators.extend([0.0, 0.0, 0.0, 0.0])
        
        # RSI (simplified)
        if len(prices) >= 14:
            price_changes = np.diff(prices[-15:])
            gains = price_changes[price_changes > 0]
            losses = -price_changes[price_changes  0 else 0.0
            avg_loss = np.mean(losses) if len(losses) > 0 else 0.0
            
            if avg_loss > 0:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            else:
                rsi = 100.0
            
            indicators.append((rsi - 50) / 50)  # Normalized RSI
        else:
            indicators.append(0.0)
        
        # Volume indicators
        if len(volumes) >= 10:
            volume_ma = np.mean(volumes[-10:])
            volume_ratio = volumes[-1] / max(volume_ma, 1e-6)
            indicators.append(np.log(max(volume_ratio, 1e-6)))  # Log volume ratio
        else:
            indicators.append(0.0)
        
        # Bollinger Bands
        if len(prices) >= 20:
            ma_20 = np.mean(prices[-20:])
            std_20 = np.std(prices[-20:])
            bb_position = (prices[-1] - ma_20) / max(std_20, 1e-6)
            indicators.append(bb_position)
        else:
            indicators.append(0.0)
        
        return indicators
    
    def _calculate_frequency_features(self, prices: np.ndarray) -> List[float]:
        """Calculate frequency domain features."""
        if len(prices) = 0.85 * cumsum_power[-1])[0]
        spectral_rolloff = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else 0.0
        features.append(spectral_rolloff)
        
        # Spectral flux (change in spectrum)
        if hasattr(self, '_prev_spectrum'):
            spectral_flux = np.sum((power_spectrum[:len(freqs)//2] - self._prev_spectrum) ** 2)
            features.append(spectral_flux)
        else:
            features.append(0.0)
        
        self._prev_spectrum = power_spectrum[:len(freqs)//2]
        
        # Energy in frequency bands
        n_bins = len(freqs) // 2
        band_size = n_bins // 5
        
        for i in range(5):
            start_idx = i * band_size
            end_idx = min((i + 1) * band_size, n_bins)
            band_energy = np.sum(power_spectrum[start_idx:end_idx])
            features.append(band_energy)
        
        return features
    
    def _calculate_pattern_features(self, prices: np.ndarray, highs: np.ndarray, lows: np.ndarray) -> List[float]:
        """Calculate high-frequency pattern features."""
        features = []
        
        # Price gaps
        if len(prices) >= 2:
            price_gap = (prices[-1] - prices[-2]) / prices[-2]
            features.append(price_gap)
        else:
            features.append(0.0)
        
        # Intraday range
        if len(highs) >= 1 and len(lows) >= 1:
            intraday_range = (highs[-1] - lows[-1]) / max(prices[-1], 1e-6)
            features.append(intraday_range)
        else:
            features.append(0.0)
        
        # Local extrema count
        if len(prices) >= 5:
            local_maxima = 0
            local_minima = 0
            
            for i in range(2, len(prices) - 2):
                if prices[i] > prices[i-1] and prices[i] > prices[i+1]:
                    local_maxima += 1
                elif prices[i] = 10:
            short_trend = np.polyfit(range(5), prices[-5:], 1)[0]
            long_trend = np.polyfit(range(10), prices[-10:], 1)[0]
            trend_consistency = short_trend * long_trend  # Same direction = positive
            features.append(trend_consistency)
        else:
            features.append(0.0)
        
        return features

class CompressedSensingAnalyzer:
    """Compressed sensing analyzer for market data."""
    
    def __init__(self, config: CompressedSensingConfig = None):
        self.config = config or CompressedSensingConfig()
        self.feature_extractor = MarketFeatureExtractor()
        self.dictionary = None
        self.sparse_coder = None
        self.scaler = StandardScaler()
        
        logger.info("[SUCCESS] Compressed Sensing Analyzer initialized")
    
    def learn_sparse_dictionary(self, market_data: List[MarketData]) -> Dict[str, Any]:
        """Learn sparse dictionary from market data."""
        # Extract features
        features = self.feature_extractor.extract_features(market_data)
        
        if features.size == 0:
            return {}
        
        # Normalize features
        features_normalized = self.scaler.fit_transform(features)
        
        # Learn dictionary using sparse coding
        dict_learner = DictionaryLearning(
            n_components=self.config.n_components,
            alpha=self.config.l1_alpha,
            max_iter=self.config.max_iter,
            tol=self.config.tolerance,
            random_state=42
        )
        
        # Fit dictionary
        sparse_codes = dict_learner.fit_transform(features_normalized)
        self.dictionary = dict_learner.components_
        self.sparse_coder = dict_learner
        
        # Analyze dictionary
        dictionary_analysis = {
            'dictionary_shape': self.dictionary.shape,
            'sparsity_level': np.mean(sparse_codes == 0),
            'reconstruction_error': dict_learner.reconstruction_err_,
            'n_iter': dict_learner.n_iter_,
            'components_analysis': self._analyze_dictionary_components()
        }
        
        logger.info(f"[SUCCESS] Sparse dictionary learned: {self.dictionary.shape} components")
        return dictionary_analysis
    
    def _analyze_dictionary_components(self) -> List[Dict[str, Any]]:
        """Analyze learned dictionary components."""
        if self.dictionary is None:
            return []
        
        components_analysis = []
        
        for i, component in enumerate(self.dictionary):
            # Find most important features in this component
            important_indices = np.argsort(np.abs(component))[-10:]  # Top 10 features
            
            component_info = {
                'component_id': i,
                'sparsity': np.mean(component == 0),
                'energy': np.sum(component ** 2),
                'max_weight': np.max(np.abs(component)),
                'important_features': important_indices.tolist(),
                'feature_weights': component[important_indices].tolist()
            }
            
            components_analysis.append(component_info)
        
        return components_analysis
    
    def encode_sparse_representation(self, market_data: List[MarketData]) -> Dict[str, Any]:
        """Encode market data as sparse representation."""
        if self.sparse_coder is None:
            raise ValueError("Must learn dictionary first using learn_sparse_dictionary()")
        
        # Extract features
        features = self.feature_extractor.extract_features(market_data)
        
        if features.size == 0:
            return {}
        
        # Normalize features
        features_normalized = self.scaler.transform(features)
        
        # Encode as sparse representation
        sparse_codes = self.sparse_coder.transform(features_normalized)
        
        # Analyze sparse codes
        sparse_analysis = {
            'sparse_codes': sparse_codes,
            'sparsity_level': np.mean(sparse_codes == 0),
            'active_components': np.sum(sparse_codes != 0, axis=1),
            'reconstruction_error': self._calculate_reconstruction_error(features_normalized, sparse_codes),
            'dominant_components': self._find_dominant_components(sparse_codes)
        }
        
        logger.info(f"[SUCCESS] Sparse encoding completed: {sparse_codes.shape}")
        return sparse_analysis
    
    def _calculate_reconstruction_error(self, original: np.ndarray, sparse_codes: np.ndarray) -> float:
        """Calculate reconstruction error."""
        if self.dictionary is None:
            return 0.0
        
        reconstructed = sparse_codes @ self.dictionary
        mse = np.mean((original - reconstructed) ** 2)
        return float(mse)
    
    def _find_dominant_components(self, sparse_codes: np.ndarray) -> List[Dict[str, float]]:
        """Find dominant components in sparse representation."""
        # Calculate component usage statistics
        component_usage = np.mean(np.abs(sparse_codes), axis=0)
        component_frequency = np.mean(sparse_codes != 0, axis=0)
        
        dominant_components = []
        
        for i, (usage, frequency) in enumerate(zip(component_usage, component_frequency)):
            if frequency > 0.05:  # Used in at least 5% of samples
                dominant_components.append({
                    'component_id': int(i),
                    'avg_usage': float(usage),
                    'frequency': float(frequency),
                    'importance': float(usage * frequency)
                })
        
        # Sort by importance
        dominant_components.sort(key=lambda x: x['importance'], reverse=True)
        
        return dominant_components[:20]  # Top 20 components

class AnomalyDetector:
    """Detect anomalies using compressed sensing."""
    
    def __init__(self, config: CompressedSensingConfig = None):
        self.config = config or CompressedSensingConfig()
        self.baseline_representation = None
        self.anomaly_threshold = None
        
        logger.info("[SUCCESS] Anomaly Detector initialized")
    
    def fit_baseline(self, normal_data: List[MarketData], cs_analyzer: CompressedSensingAnalyzer):
        """Fit baseline representation on normal market data."""
        # Get sparse representation of normal data
        sparse_analysis = cs_analyzer.encode_sparse_representation(normal_data)
        
        if not sparse_analysis:
            return
        
        sparse_codes = sparse_analysis['sparse_codes']
        
        # Calculate baseline statistics
        self.baseline_representation = {
            'mean_codes': np.mean(sparse_codes, axis=0),
            'std_codes': np.std(sparse_codes, axis=0),
            'covariance': np.cov(sparse_codes.T)
        }
        
        # Set anomaly threshold based on reconstruction errors
        reconstruction_errors = []
        for i in range(len(sparse_codes)):
            error = cs_analyzer._calculate_reconstruction_error(
                sparse_codes[i:i+1], sparse_codes[i:i+1]
            )
            reconstruction_errors.append(error)
        
        mean_error = np.mean(reconstruction_errors)
        std_error = np.std(reconstruction_errors)
        self.anomaly_threshold = mean_error + self.config.anomaly_threshold * std_error
        
        logger.info(f"[SUCCESS] Baseline fitted with threshold: {self.anomaly_threshold:.6f}")
    
    def detect_anomalies(self, market_data: List[MarketData], cs_analyzer: CompressedSensingAnalyzer) -> List[Dict[str, Any]]:
        """Detect anomalies in market data."""
        if self.baseline_representation is None:
            raise ValueError("Must fit baseline first using fit_baseline()")
        
        # Get sparse representation
        sparse_analysis = cs_analyzer.encode_sparse_representation(market_data)
        
        if not sparse_analysis:
            return []
        
        sparse_codes = sparse_analysis['sparse_codes']
        anomalies = []
        
        for i, code in enumerate(sparse_codes):
            # Calculate anomaly score
            anomaly_score = self._calculate_anomaly_score(code)
            
            # Check if anomalous
            if anomaly_score > self.anomaly_threshold:
                anomaly = {
                    'timestamp': market_data[i + 50].timestamp if i + 50  float:
        """Calculate anomaly score for sparse code."""
        if self.baseline_representation is None:
            return 0.0
        
        baseline_mean = self.baseline_representation['mean_codes']
        baseline_std = self.baseline_representation['std_codes']
        
        # Mahalanobis-like distance
        normalized_diff = (sparse_code - baseline_mean) / np.maximum(baseline_std, 1e-6)
        anomaly_score = np.sqrt(np.sum(normalized_diff ** 2))
        
        return anomaly_score
    
    def _find_anomalous_components(self, sparse_code: np.ndarray) -> List[int]:
        """Find which components contribute most to anomaly."""
        if self.baseline_representation is None:
            return []
        
        baseline_mean = self.baseline_representation['mean_codes']
        baseline_std = self.baseline_representation['std_codes']
        
        # Calculate z-scores for each component
        z_scores = np.abs((sparse_code - baseline_mean) / np.maximum(baseline_std, 1e-6))
        
        # Find components with high z-scores
        anomalous_indices = np.where(z_scores > 2.0)[0]  # More than 2 std deviations
        
        return anomalous_indices.tolist()

class CompressedSensingTradingModel(BaseTradingModel):
    """Trading model based on compressed sensing analysis."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.cs_config = CompressedSensingConfig(**self.config.get('compressed_sensing', {}))
        
        self.cs_analyzer = CompressedSensingAnalyzer(self.cs_config)
        self.anomaly_detector = AnomalyDetector(self.cs_config)
        self.is_trained = False
        
        logger.info("[SUCCESS] Compressed Sensing Trading Model initialized")
    
    def prepare_data(self, market_data: List[MarketData]) -> None:
        """Prepare data for compressed sensing analysis."""
        logger.info(f"[SUCCESS] Data prepared: {len(market_data)} data points")
    
    def train(self, validation_data: Optional[List[MarketData]] = None) -> Dict[str, Any]:
        """Train compressed sensing model."""
        if not hasattr(self, '_training_data'):
            raise ValueError("Must call prepare_data() with training data first")
        
        # Learn sparse dictionary
        dictionary_analysis = self.cs_analyzer.learn_sparse_dictionary(self._training_data)
        
        # Fit anomaly detector baseline
        self.anomaly_detector.fit_baseline(self._training_data, self.cs_analyzer)
        
        self.is_trained = True
        
        training_results = {
            'dictionary_analysis': dictionary_analysis,
            'model_type': 'compressed_sensing',
            'training_samples': len(self._training_data)
        }
        
        logger.info("[SUCCESS] Compressed sensing model training completed")
        return training_results
    
    def predict(self, market_data: List[MarketData]) -> List[TradingSignal]:
        """Generate trading signals using compressed sensing analysis."""
        if not self.is_trained:
            # For demonstration, we'll train on the provided data
            self._training_data = market_data[:len(market_data)//2]  # Use first half for training
            self.train()
            prediction_data = market_data[len(market_data)//2:]  # Use second half for prediction
        else:
            prediction_data = market_data
        
        signals = []
        
        # Encode sparse representation
        sparse_analysis = self.cs_analyzer.encode_sparse_representation(prediction_data)
        
        if not sparse_analysis:
            return signals
        
        # Detect anomalies
        anomalies = self.anomaly_detector.detect_anomalies(prediction_data, self.cs_analyzer)
        
        # Generate signals from sparse analysis
        dominant_components = sparse_analysis.get('dominant_components', [])
        
        for component in dominant_components[:3]:  # Top 3 components
            if component['importance'] > 0.1:  # Significant component
                
                signal_type = self._determine_signal_from_component(component, sparse_analysis)
                confidence = min(component['importance'] * 2, 1.0)
                
                if signal_type != "HOLD" and confidence >= 0.3:
                    signal = TradingSignal(
                        symbol=prediction_data[-1].symbol,
                        signal_type=signal_type,
                        confidence=Decimal(str(confidence)),
                        price_target=prediction_data[-1].close,
                        stop_loss=self._calculate_stop_loss(prediction_data[-1].close, signal_type),
                        take_profit=self._calculate_take_profit(prediction_data[-1].close, signal_type),
                        timestamp=prediction_data[-1].timestamp,
                        metadata={
                            'component_id': component['component_id'],
                            'component_importance': component['importance'],
                            'sparsity_level': sparse_analysis['sparsity_level'],
                            'reconstruction_error': sparse_analysis['reconstruction_error'],
                            'model_type': 'Compressed_Sensing'
                        }
                    )
                    
                    signals.append(signal)
        
        # Generate signals from anomalies
        for anomaly in anomalies[-3:]:  # Recent anomalies
            if anomaly['severity'] > 2.0:  # Significant anomaly
                
                # Anomalies often precede reversals
                signal_type = "SELL" if anomaly['severity'] > 3.0 else "BUY"
                confidence = min(anomaly['severity'] / 5.0, 1.0)
                
                if confidence >= 0.4:
                    signal = TradingSignal(
                        symbol=prediction_data[-1].symbol,
                        signal_type=signal_type,
                        confidence=Decimal(str(confidence)),
                        price_target=prediction_data[-1].close,
                        stop_loss=self._calculate_stop_loss(prediction_data[-1].close, signal_type),
                        take_profit=self._calculate_take_profit(prediction_data[-1].close, signal_type),
                        timestamp=prediction_data[-1].timestamp,
                        metadata={
                            'anomaly_score': anomaly['anomaly_score'],
                            'anomaly_severity': anomaly['severity'],
                            'anomalous_components': anomaly['anomalous_components'],
                            'signal_source': 'anomaly_detection',
                            'model_type': 'Compressed_Sensing_Anomaly'
                        }
                    )
                    
                    signals.append(signal)
        
        logger.info(f"[SUCCESS] Generated {len(signals)} compressed sensing signals")
        return signals
    
    def _determine_signal_from_component(self, component: Dict[str, Any], sparse_analysis: Dict[str, Any]) -> str:
        """Determine trading signal from sparse component analysis."""
        importance = component['importance']
        frequency = component['frequency']
        
        # Simple heuristic: high importance + high frequency = strong signal
        if importance > 0.15 and frequency > 0.2:
            return "BUY"
        elif importance > 0.12 and frequency  Decimal:
        """Calculate stop loss for compressed sensing signals."""
        stop_pct = 0.025  # 2.5% stop loss
        
        if signal_type == "BUY":
            return current_price * Decimal(str(1 - stop_pct))
        else:
            return current_price * Decimal(str(1 + stop_pct))
    
    def _calculate_take_profit(self, current_price: Decimal, signal_type: str) -> Decimal:
        """Calculate take profit for compressed sensing signals."""
        profit_pct = 0.05  # 5% take profit
        
        if signal_type == "BUY":
            return current_price * Decimal(str(1 + profit_pct))
        else:
            return current_price * Decimal(str(1 - profit_pct))

def create_compressed_sensing_model(config: Dict[str, Any] = None) -> CompressedSensingTradingModel:
    """Factory function to create compressed sensing trading model."""
    return CompressedSensingTradingModel(config)
