# IBD 50 VLM Training

This project implements Vision-Language Models (VLMs) for algorithmic trading analysis on the IBD 50 stock universe.

## Quick Start

### Prerequisites
- Python 3.8+
- GPU recommended (CUDA-compatible)
- 8GB+ RAM
- Internet connection for data fetching

### Installation
```bash
pip install -r requirements_vlm.txt
```

### Training Commands

#### Quick Test (Recommended First)
Test the pipeline on 5 stocks to verify everything works:
```bash
python vlm_quick_start.py --quick-test
```

#### Full Training
Train on all 50 IBD stocks (2-4 hours):
```bash
python vlm_quick_start.py --full-training
```

#### Custom Number of Stocks
Test on a specific number of stocks:
```bash
python vlm_quick_start.py --custom-stocks 10
```

#### Evaluate Existing Model
Evaluate a previously trained model:
```bash
python vlm_quick_start.py --evaluate-model
```

#### Check Training Status
View current training progress and results:
```bash
python vlm_quick_start.py --status
```

## Project Structure

```
vlm/
├── models/ibd50/           # Trained models
│   ├── clip_models/       # CLIP fine-tuned models
│   ├── fusion_models/     # Multimodal fusion models
│   └── training_summary.json
├── data/ibd50/            # Training datasets
│   └── ibd50_training_dataset_v1.0/
├── chart_image_generator.py    # Chart generation
├── text_label_generator.py     # Text labeling
├── dataset_builder.py          # Dataset creation
├── clip_baseline_eval.py       # Baseline evaluation
├── clip_fine_tune.py          # CLIP fine-tuning
├── clip_calibration.py        # Model calibration
├── multimodal_fusion.py       # Fusion training
├── vlm_inference_service.py   # Inference service
├── vlm_xgboost_integration.py # XGBoost integration
└── Visual Language Model.md   # Documentation
```

## Key Components

### 1. Data Pipeline
- **Historical Data**: Fetches 2 years of daily data for IBD 50 stocks
- **Technical Indicators**: RSI, MACD, Bollinger Bands, Volume analysis
- **Chart Generation**: Creates candlestick charts with mplfinance
- **Text Labeling**: Generates descriptive captions for charts

### 2. Model Training
- **CLIP Fine-tuning**: Adapts vision-language model to trading charts
- **Multimodal Fusion**: Combines visual and textual features
- **Calibration**: Optimizes prediction thresholds

### 3. Integration
- **XGBoost Integration**: Combines VLM features with traditional ML
- **Real-time Inference**: Streaming prediction service
- **ChromaDB Storage**: Vector database for embeddings

## Output Files

- `vlm/models/ibd50/clip_models/best_model.pt` - Best CLIP model
- `vlm/models/ibd50/fusion_models/fusion_model.pt` - Fusion model
- `vlm/data/ibd50/ibd50_training_dataset_v1.0/` - Training dataset
- `vlm/models/ibd50/training_summary.json` - Training results

## Performance Metrics

The models are evaluated on:
- **Accuracy**: Overall prediction accuracy
- **Precision**: Positive prediction quality
- **Recall**: Ability to find all positive cases
- **F1-Score**: Harmonic mean of precision and recall

## Troubleshooting

### Common Issues

1. **Missing Dependencies**
   ```bash
   pip install -r requirements_vlm.txt
   ```

2. **CUDA Out of Memory**
   - Reduce batch size in training scripts
   - Use CPU training: set `device='cpu'`

3. **Data Fetching Errors**
   - Check internet connection
   - Verify yfinance is working: `python -c "import yfinance; print('OK')"`

4. **Model Loading Errors**
   - Ensure model files exist in expected locations
   - Check PyTorch version compatibility

### Logs
Training logs are saved to `vlm_ibd50_training.log`

## Advanced Usage

### Custom Stock Universe
Modify `stock_universes.py` to define custom stock lists.

### Model Configuration
Adjust hyperparameters in the training scripts for different performance targets.

### Integration with Trading Systems
Use `vlm_inference_service.py` for real-time predictions in your trading platform.

## Documentation

For detailed technical documentation, see:
- `vlm/Visual Language Model.md` - Complete implementation guide
- `AI_TRADING_SYSTEM_ROADMAP_AND_PLAN.md` - System architecture
- `README.md` - Project overview

## Support

If you encounter issues:
1. Run the quick test first: `python vlm_quick_start.py --quick-test`
2. Check the logs: `tail -f vlm_ibd50_training.log`
3. Verify your environment: `python vlm_quick_start.py --status`