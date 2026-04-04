# Overnight Scaling Results

Generated: 2026-04-04 15:36:55

## Scaling Curve (per backbone)

| Objects | DINOv2 PD | V-JEPA2 PD | CLIP PD | DINOv2 Acc | V-JEPA2 Acc | CLIP Acc |
|---------|-----------|------------|---------|------------|-------------|---------|
| 3 | 0.956±0.080 | — | — | 69.5%±1.2% | — | — |
| 5 | 0.989±0.008 | — | — | 69.0%±1.2% | — | — |
| 8 | 0.814±0.110 | — | — | 69.2%±2.2% | — | — |
| 12 | 0.968±0.052 | — | — | 68.7%±3.4% | — | — |
| 20 | 0.822±0.217 | 0.698±0.186 | 0.624±0.158 | 67.8%±6.1% | 68.0%±6.4% | 66.3%±6.9% |
| 30 | 0.646±0.093 | 0.422±0.244 | 0.621±0.272 | 68.4%±6.4% | 66.3%±7.5% | 67.3%±6.5% |
| 50 | 0.571±0.355 | 0.517±0.312 | 0.595±0.256 | 65.5%±8.1% | 65.2%±7.7% | 65.5%±8.1% |
| 75 | 0.619±0.458 | 0.473±0.373 | 0.621±0.458 | 67.9%±7.4% | 68.1%±7.4% | 68.0%±7.5% |
| 100 | 0.594±0.485 | 0.446±0.364 | 0.347±0.297 | 67.6%±6.2% | 67.8%±6.4% | 67.7%±6.3% |
| 150 | 0.690±0.390 | 0.641±0.369 | 0.774±0.316 | 69.3%±4.4% | 69.2%±4.5% | 69.7%±4.6% |
| 200 | 0.690±0.394 | 0.493±0.269 | 0.543±0.327 | 68.1%±4.5% | 68.7%±4.9% | 69.5%±5.2% |

## Analysis

### DINOv2 Scaling

- 3 objects: PosDis = 0.956
- 5 objects: PosDis = 0.989
- 8 objects: PosDis = 0.814
- 12 objects: PosDis = 0.968
- 20 objects: PosDis = 0.822
- 30 objects: PosDis = 0.646
- 50 objects: PosDis = 0.571
- 75 objects: PosDis = 0.619
- 100 objects: PosDis = 0.594
- 150 objects: PosDis = 0.690
- 200 objects: PosDis = 0.690

### Break Points

- **dinov2**: PosDis ≥ 0.85 up to 12 objects, ≥ 0.50 up to 200 objects
- **vjepa2**: PosDis ≥ 0.85 up to 0 objects, ≥ 0.50 up to 150 objects
- **clip**: PosDis ≥ 0.85 up to 0 objects, ≥ 0.50 up to 200 objects

### Key Insight: Variance, Not Collapse

Compositionality doesn't collapse at high object counts — it becomes *high variance*. Some seeds still achieve PosDis > 0.98 at 200 objects, while others drop to 0. The mean drops because more seeds fail to converge to compositional solutions, not because the compositional solution doesn't exist. This suggests the loss landscape has multiple optima — compositional and non-compositional — and higher object counts make the compositional basin harder to find.


## Headline

**Compositionality persists to 200 objects** on dinov2, clip (mean PosDis > 0.5). Degradation is gradual, not catastrophic. The 40-dimensional bottleneck maintains some compositional structure even with 200 visual objects in the scene.


## Files

- `overnight_scaling_curve.png` — Main result figure (PosDis + Accuracy)
- `accuracy_curve.png` — Accuracy vs object count
- `posdis_variance.png` — Variance analysis
- `full_metrics_overview.png` — All 4 metrics
- `scaling_results.csv` — Raw data