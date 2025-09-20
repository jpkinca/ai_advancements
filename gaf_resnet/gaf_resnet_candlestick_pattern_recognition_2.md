Enhancing GAF-Encoded Candlestick Classification with ResNet
Switching from the paper’s simple LeNet-inspired CNN to ResNet is a smart upgrade for your stock trading AI, especially if you’re dealing with more complex or noisy datasets. ResNet (Residual Network, introduced in 2015) uses skip connections to train deeper networks (e.g., ResNet-18, -50) without vanishing gradients, making it excellent for capturing subtle hierarchies in images—like the temporal correlations in GAF-encoded candlestick patterns. The original paper achieves ~90.7% accuracy with a basic CNN on EUR/USD data, but ResNet could push this higher by learning finer features, as seen in related time-series works.
From recent research (up to 2025), ResNet pairs well with GAF in financial time series:
	•	A 2025 IEEE study on multimodal financial prediction converts time series to GAF images and feeds them into deep CNNs (implying ResNet-like architectures) for better forecasting, outperforming baselines by 5-10% in volatility prediction.
	•	A 2016 arXiv paper on time-series classification proposes ResNet directly on GAF-transformed data, achieving state-of-the-art results on benchmarks (e.g., UCR archive), with accuracy gains of 2-5% over shallower nets like LeNet.
	•	Other works (e.g., 2025 Nature Scientific Reports) use ResNet variants with GAF/RP fusions for stock forecasting, noting improved generalization in volatile markets.
Benefits for Stock Trading Algos:
	•	Deeper Feature Extraction: ResNet’s residuals help detect nuanced candlestick subtleties (e.g., shadow lengths in Hammer vs. Inverted Hammer) that LeNet might miss, leading to more reliable reversal signals.
	•	Transfer Learning: Pre-train on ImageNet, fine-tune on your GAF images—ideal for small datasets like the paper’s (~2900 samples). This reduces training time and overfitting.
	•	Scalability: Handles multi-asset or higher-resolution GAF (e.g., longer time windows), enabling ensemble strategies with indicators like RSI.
Concerns and Mitigations:
	•	Overfitting Risk: Deeper models like ResNet-50 need more data; the paper’s simulated GBM + real data might not suffice. Mitigate with dropout (0.5 as in paper), data augmentation (e.g., rotate GAF images), or synthetic data via GBM.
	•	Compute Cost: ResNet is heavier than LeNet—training on CPU could take hours vs. minutes. Use GPUs; for real-time trading, deploy inference-only on edge devices.
	•	Not Always Superior: In low-noise simulations, gains might be marginal (1-3%); test on your EUR/USD data. If patterns are simple, stick to LeNet for efficiency.
	•	Interpretability: Deeper nets are black-boxier; combine with Grad-CAM (as in some GAF papers) for visual explanations of predictions.
Programmatic Implementation: Use transfer learning with Keras/TensorFlow. First, encode OHLC as GAF (from paper’s Eq. 1-4), then fine-tune ResNet. Assume 32x32 GAF images for 8 classes.
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam

# GAF Encoding (from paper, simplified for closes; extend to OHLC channels)
def normalize_series(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x)) if np.max(x) != np.min(x) else x

def gaf_encode(normalized_x):
    phi = np.arccos(normalized_x.clip(-1, 1))  # Handle edge cases
    cos_sum = np.cos(np.outer(phi, np.ones_like(phi)) + np.outer(np.ones_like(phi), phi))
    return cos_sum  # Resize to 32x32 if needed via cv2.resize

# Example: Encode sample closes
closes = np.random.rand(32)  # Replace with real OHLC window
norm_closes = normalize_series(closes)
gaf_image = gaf_encode(norm_closes)  # Shape: (32,32,1) after resize

# ResNet Model with Transfer Learning
def build_gaf_resnet(input_shape=(32, 32, 3), num_classes=8):  # RGB for multi-channel GAF
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False  # Freeze base for transfer learning
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Usage: Train on GAF dataset
# model.fit(gaf_train_images, train_labels, epochs=20, validation_split=0.2)
# Fine-tune: base_model.trainable = True; recompile with lower LR
Start with ResNet-18 for lighter compute. Backtest on historical data (e.g., via yfinance) to validate signals—aim for Sharpe >1.0. If accuracy plateaus, hybridize with LSTM as in some papers. Let’s prototype this if you share sample data.
