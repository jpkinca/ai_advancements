# Vision-Language Model (VLM) vs. GAF+ResNet Candlestick Classifier

Date: 2025-09-19
Status: Reference Comparison

## 1. High-Level Summary
| Aspect | VLM (Fine-Tuned CLIP) | GAF+ResNet |
|--------|----------------------|-----------|
| Modality | Multimodal (image + text) | Unimodal (image derived from time-series) |
| Input Representation | Rendered candlestick chart (PNG) + textual prompt/label | Gramian Angular Field (GAF) matrices from normalized OHLC/V windows |
| Primary Task | Semantic pattern understanding & zero-shot querying | Fixed-class supervised pattern classification |
| Flexibility | High (dynamic prompts, retrieval) | Low–Medium (requires retrain to add classes) |
| Zero-Shot Capability | Yes (prompt-based) | No |
| Embedding Utility | Similarity search, clustering, regime tagging | Limited (internal CNN features only) |
| Labeling Overhead | Requires image–text pairing (augmentable) | Simple class labels |
| Latency (Optimized) | Moderate (<80 ms GPU per chart) | Low (<10–20 ms GPU / faster with small CNN) |
| Interpretability | Prompt alignment + embedding introspection | Grad-CAM on GAF + class probabilities |
| Extensibility | Broad (new prompts, tasks) | Additive (new classes via retraining) |
| Deployment Footprint | Larger (Transformer) | Smaller (ResNet-18 or custom CNN) |

## 2. Core Functional Differences
- VLM aligns visual pattern structure with natural language semantics, enabling adaptive querying (e.g., "detect inside bar compression before breakout").
- GAF+ResNet transforms temporal numerical structure into a correlation-preserving image and classifies among predefined reversal categories.

## 3. Data & Label Strategy
| Dimension | VLM | GAF+ResNet |
|----------|-----|------------|
| Data Source | Historical OHLCV → rendered charts | OHLC(V) → normalized → GAF encode |
| Labels | Descriptive phrases (multi-form) | Discrete pattern IDs (e.g., 0–7) |
| Augmentation | Text paraphrasing, chart style variants | Window jittering, synthetic GBM series |
| Hard Negatives | Semantically similar but mislabeled descriptions | Visually similar other pattern windows |

## 4. Learning Paradigm
| Attribute | VLM | GAF+ResNet |
|----------|-----|------------|
| Objective | Contrastive (image↔text alignment) | Cross-entropy classification |
| Transfer Learning | Pretrained CLIP + partial unfreeze | ImageNet-pretrained ResNet backbone (optional) |
| Zero-Shot Adaptation | By adding prompts | Not supported |

## 5. Strengths & Weaknesses
### VLM Strengths
- Zero/low-shot adaptability for new pattern concepts.
- Embeddings reusable for similarity search ("find past structurally similar reversals").
- Semantic explainability in logs (stores prompt and match confidence).

### VLM Weaknesses
- Heavier computations; more complex pipeline (render + prompt + encode).
- Sensitive to prompt phrasing drift.
- Requires disciplined dataset versioning to avoid label noise.

### GAF+ResNet Strengths
- Simple, deterministic, fast.
- Performs well with limited, well-defined pattern sets.
- GAF encoding preserves temporal correlations in a compact form.

### GAF+ResNet Weaknesses
- No semantic generalization; rigid class boundaries.
- Hard to explore emergent or composite patterns without retraining.
- GAF abstractions may miss contextual cues (volume anomalies, multi-timeframe context) unless added as extra channels.

## 6. Use Case Alignment
| Use Case | Recommended Approach |
|----------|---------------------|
| Production core reversal signal | GAF+ResNet (stable & efficient) |
| Exploratory alpha discovery | VLM |
| Zero-shot narrative / hypothesis testing | VLM |
| Low-resource deployment (edge) | GAF or lightweight CNN |
| Regime tagging & retrieval | VLM |
| Ensemble diversification | Both (fusion layer) |

## 7. Fusion Strategy
1. Parallel Inference: Run both models; collect probabilities + embeddings.
2. Confidence Gating: Require VLM semantic confidence > threshold to confirm ambiguous GAF classifications.
3. Weighted Ensemble: `p_final = w_gaf * p_gaf + w_vlm * p_vlm` (class-aligned mapping or meta-head).
4. Conflict Resolution: Favor high historical precision source for that class.

## 8. Monitoring Considerations
| Metric | VLM | GAF+ResNet |
|--------|-----|-----------|
| Embedding Drift | μ/σ shift in embedding space | N/A (unless feature tapping) |
| Class Distribution Shift | Via predicted prompt matches | Via class probability histogram |
| Latency | Render + encode | Encode only |
| Pattern Recall Stability | Prompt set regression tests | Standard confusion matrix tracking |

## 9. Failure Modes
| Failure | VLM Impact | GAF Impact | Mitigation |
|---------|------------|-----------|-----------|
| Prompt Drift | Misclassification / inconsistent retrieval | N/A | Versioned prompt library + regression suite |
| Label Noise | Embedding misalignment | Class boundary fuzziness | Tiered confidence labels / curation |
| Data Gaps | Poor rendering context | GAF distortion (abrupt scaling) | Integrity checks + window sufficiency guard |
| Regime Shift | Semantic mismatch (new volatility regime) | Pattern frequency changes | Embedding drift alarms + retrain schedule |

## 10. Deployment Recommendations
- Stage 1: Deploy GAF+ResNet for immediate pattern classification baseline.
- Stage 2: Bring up VLM in parallel (research mode) logging shadow predictions.
- Stage 3: Introduce fusion with conservative weighting (e.g., 0.7 GAF / 0.3 VLM) until VLM calibration matures.
- Stage 4: Promote VLM queries for advanced analytics (pattern clustering, narrative tagging).

## 11. Decision Matrix (Quick Reference)
| Question | Choose VLM If | Choose GAF+ResNet If |
|----------|---------------|----------------------|
| Need to query “novel” pattern verbally? | Yes | No |
| GPU budget constrained? | No | Yes |
| Need stable, low-latency fixed labels? | No | Yes |
| Desire retrieval / similarity search? | Yes | Not required |
| Rapid initial deployment? | Lower priority | High priority |

## 12. Next Steps (If Pursued Together)
1. Finalize GAF pattern class definitions & training set.
2. Stand up VLM dataset pipeline (paired render + text) with small pilot (500 samples).
3. Establish shared evaluation harness producing side-by-side metrics.
4. Implement fusion experiment and track lift in precision/recall & trading PnL impact.

## 13. Glossary
- GAF: Gramian Angular Field – transformation encoding temporal correlations.
- CLIP: Contrastive Language-Image Pretraining model (image/text embedding alignment).
- Zero-Shot: Inference on concepts not explicitly present in supervised training labels by using descriptive prompts.
- Hard Negative: Similar-looking sample intentionally labeled differently to sharpen decision boundaries.

---
This comparison will guide architectural decisions and fusion strategy calibration.
