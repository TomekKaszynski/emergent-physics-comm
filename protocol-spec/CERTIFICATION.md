# WMCP Certification Levels

## Overview

Three certification levels verify protocol quality for deployment. Each level builds on the previous.

## Level 1 — Bronze

**Minimum viable deployment.**

| Requirement | Threshold | Test |
|------------|-----------|------|
| Message format | Valid K, L, one-hot symbols | `wmcp validate --level bronze` |
| PosDis | > 0.3 | Computed on holdout objects |
| Accuracy | > 55% | Pairwise comparison on holdout |
| Encoder frozen | Verified | Check no gradient flow |

**Use case:** Development, internal testing, non-critical applications.

## Level 2 — Silver

**Production-ready for controlled environments.**

| Requirement | Threshold | Test |
|------------|-----------|------|
| All Bronze requirements | ✓ | — |
| PosDis | > 0.5 | Triple metrics (PosDis + TopSim + BosDis) |
| Noise robustness | < 5% accuracy drop at σ=0.3 | Gaussian noise injection |
| Onboarding speed | < 100 steps to 90% | New encoder fine-tune |
| Latency | < 10ms on target hardware | Standardized benchmark |
| Holdout validation | Object-level, 20% holdout | Not sample-level |

**Use case:** Production deployment in controlled industrial environments (factory floor, warehouse).

## Level 3 — Gold

**Enterprise-grade deployment with security and monitoring.**

| Requirement | Threshold | Test |
|------------|-----------|------|
| All Silver requirements | ✓ | — |
| PosDis | > 0.7 | Stringent compositionality |
| Noise robustness | < 10% accuracy drop at σ=0.5 | Higher noise tolerance |
| Adversarial detection | Enabled, > 90% detection rate | Entropy-based anomaly detection |
| Monitoring | Active drift detection | ProtocolMonitor enrolled |
| Multi-encoder | ≥ 2 encoder types validated | Cross-architecture compliance |
| Temporal stability | No drift over 10,000 rounds | Phase 114 methodology |
| Message signing | Enabled (when crypto module available) | Integrity verification |

**Use case:** Mission-critical enterprise deployment (autonomous vehicles, surgical robots, defense).

## Certification Process

```bash
# Run certification at desired level
wmcp certify --level bronze --protocol spring.wmcp
wmcp certify --level silver --protocol spring.wmcp
wmcp certify --level gold --protocol spring.wmcp
```

Output: `WMCP_CERTIFICATE.json` containing:
- Protocol identifier
- Certification level
- Test results per requirement
- Timestamp
- Expires: 1 year (recertification required)

## Certification Badge

Certified protocols display:

```
┌─────────────────────┐
│  WMCP CERTIFIED     │
│  ★★★ GOLD           │
│  v0.1.0 | 2026-03   │
└─────────────────────┘
```

## Revenue Model

- **Bronze:** Free (self-service, automated)
- **Silver:** $500/year per protocol instance (automated + support)
- **Gold:** $2,500/year per protocol instance (manual review + SLA + priority support)
- **Enterprise:** Custom pricing for fleet-wide certification
