# Vision-Language Model (VLM) Development Plan

Date: 2025-09-19
Status: Planning / Pre-Implementation
Owner: AI Advancements Engineering

## 1. Objective
Implement a fine-tuned CLIP-based multimodal system for semantic candlestick and structural pattern understanding, zero-shot querying, and integration into the algorithmic trading signal stack.

## 2. KPIs
| Metric | Target (Phase 1) | Stretch (Phase 2) |
|--------|------------------|-------------------|
| Pattern Recall (core set) | >= 0.92 | >= 0.95 |
| Zero-Shot Pattern Precision | >= 0.70 | >= 0.80 |
| Inference Latency (single 224x224) | < 80 ms P95 | < 50 ms P95 |
| Calibration Recall Lift | +5% | +8% |
| Embedding Drift Alert Lead Time | < 1 trading day | < 0.5 trading day |

## 3. Phases Overview
| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Spec + chart renderer + dataset v1 | Pending |
| 2 | Zero-shot baseline + fine-tune CLIP | Pending |
| 3 | Calibration + hybrid fusion | Pending |
| 4 | Inference service + trading integration | Pending |
| 5 | Monitoring + drift + optimization | Pending |
| 6 | Advanced retrieval / active learning | Future |

## 4. Components
- Chart Rendering Pipeline
- Text Label Generation & Augmentation
- Dataset Builder (image + text pairs w/ versioning)
- CLIP Fine-Tuning & Hard Negative Miner
- Calibration Layer (temperature / Platt)
- Hybrid Fusion (VLM + CNN/ResNet + XGBoost features)
- Inference Microservice (FastAPI)
- Ensemble Integration (confidence gating)
- Monitoring & Drift Detection

## 5. Directory Layout (Planned)
```
vlm/
  rendering/
    chart_renderer.py
    style_config.py
  labeling/
    pattern_rules.py
    llm_augment.py
  dataset/
    build_vlm_dataset.py
    dataset_schema.md
  training/
    clip_finetune.py
    hard_negative_buffer.py
    calibration.py
  inference/
    service.py
    prompt_library.json
  fusion/
    hybrid_fuser.py
  monitoring/
    embedding_monitor.py
    drift_metrics.py
  config/
    vlm_config.yaml
```

## 6. Detailed Tasks
1. Requirements & Pattern Taxonomy (Complete specification doc)
2. Deterministic Chart Rendering (candles + overlays + multi-timeframe composites)
3. Rule-Based Pattern Seeding (engulfing, hammer, doji, morning/evening star, inside bar, outside bar)
4. LLM Augmented Description Generation (variation control, dedupe via embedding clustering)
5. Dataset Assembly (train/val/test stratified; version tagging)
6. Zero-Shot Baseline (frozen CLIP; prompt engineering cycle)
7. Fine-Tuning (partial unfreeze; mixed precision; hard negatives)
8. Calibration (temperature scaling on validation logits)
9. Hybrid Fusion Prototype (probability blend and MLP feature concat)
10. Inference Service (FastAPI + batching + model warm start)
11. Trading Integration (ensemble gating w/ XGBoost directional bias)
12. Monitoring (centroid shift, cosine variance, latency histogram, recall spot-check set)
13. Performance Optimization (prompt caching, AMP, ONNX trial, quantization if needed)
14. Documentation Wave 1 (pipeline, API, dataset guide)

## 7. Data & Label Strategy
- Window Sizes: 50–120 bars configurable.
- Timeframes: 1m, 5m, 15m, 1h, 1d.
- Label Confidence Tiers: rule_high, rule_medium, augmented.
- Hard Negatives: visually similar but different labeled patterns (e.g., inverted hammer vs hammer).

## 8. Evaluation Protocol
- Hold-out test set never used in prompt tuning.
- Metrics: per-pattern recall/precision, macro F1, zero-shot success rate.
- Drift Evaluation: weekly embedding distance vs. baseline centroid.

## 9. Hybrid Fusion Design
Two-stage:
1. Late Fusion: weighted sum of calibrated probabilities.
2. Learned Fusion: MLP(head([embedding_vlm, embedding_resnet, engineered_features])) → logits.
Fallback to late fusion if MLP underperforms or overfits.

## 10. Inference API Sketch
```
POST /predict/patterns
{ "symbol": "AAPL", "timeframe": "5m", "bars": [...], "patterns": ["bullish engulfing", "hammer"] }

POST /query/zero_shot
{ "image_base64": "...", "prompts": ["evening star", "inside bar breakout"] }

GET /embeddings?symbol=AAPL&timeframe=5m&lookback=100
```
Responses include: probabilities, calibrated flag, latency_ms, model_version, embedding_id.

## 11. Monitoring Metrics
| Metric | Method | Threshold |
|--------|--------|-----------|
| Embedding Centroid Shift | L2(μ_t, μ_ref)/||μ_ref|| | > 0.15 warn |
| Cosine Variance Increase | Δσ_cosine | > 20% warn |
| Pattern Recall Degradation | Rolling 7d vs. baseline | -5% warn |
| Latency P95 | Logged | > 2× baseline warn |

## 12. Risks & Mitigations
| Risk | Impact | Mitigation |
|------|--------|-----------|
| Label Noise (augmented text) | Reduced alignment | Confidence tiers + filtering by embedding similarity |
| Overfitting small dataset | Poor generalization | Freeze early layers + heavy augmentation + hard negatives |
| Prompt Drift | Inconsistent zero-shot results | Versioned prompt library; regression prompts suite |
| Latency Spikes | Trading timing risk | Batch + warm pools + profiling early |
| Embedding Drift | Silent performance decay | Scheduled drift checks + alert pipeline |

## 13. Tooling
- Hugging Face Transformers & Datasets
- PyTorch AMP
- FastAPI / Uvicorn
- Optional: ONNX Runtime for inference acceleration
- Redis (cache prompts & embeddings) (optional)

## 14. Acceptance Criteria (Phase 1 Completion)
- Dataset v1 constructed with >= 3,000 paired samples across >= 6 patterns + neutral
- Zero-shot baseline recorded prior to fine-tune
- Fine-tuned model improves macro recall by >= 10% absolute over baseline
- Inference API functional (patterns + zero-shot) with latency logs
- Monitoring script produces drift + latency metrics JSON artifacts

## 15. Future Extensions
- Active learning (query uncertain samples)
- Pattern narrative generation
- Multi-modal fusion (news headline context + chart)
- Continual training loop

---
This file will be updated as milestones are achieved.
