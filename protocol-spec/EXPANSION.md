# WMCP Expansion Architecture

## Layer 1: Protocol Spec + Enrollment SDK (NOW)

**What:** Formal protocol specification, pip-installable Python package, compliance validation suite, onboarding workflow for new encoders.

**What it requires:** The current codebase — 118 phases of experimental validation, wmcp Python package (23 tests pass), protocol-spec/ documentation.

**Revenue model:** Open-source core. No revenue at this layer. The goal is adoption and external reproduction.

**Dependencies:** None. This layer is complete.

## Layer 2: Monitoring & Observability (Month 3–6)

**What:** Real-time tracking of protocol health in deployment. Message quality monitoring (entropy drift, PosDis degradation over time), latency alerts, anomaly detection for corrupted or adversarial agents, dashboard for fleet-wide communication status.

**What it requires:**
- Streaming metrics pipeline (message entropy, per-position symbol frequency, accuracy on held-out validation pairs)
- Anomaly detection module (Phase 112 showed entropy-based detection works for adversarial agents)
- Time-series storage for message statistics
- Alert thresholds calibrated from Phase 114 (10,000-round stability test showed no drift)

**Revenue model:** SaaS monitoring tier. Free for <10 agents, paid for fleet-scale monitoring. Usage-based pricing on message volume.

**Dependencies:** Layer 1 (deployed protocol instances to monitor).

## Layer 3: Domain Packs (Month 6–12)

**What:** Pre-trained protocol instances for specific robotics verticals. Each domain pack includes: trained projection layers for 2–3 encoder architectures, validation dataset, compliance report, deployment guide.

**Planned domain packs:**
- **Manufacturing inspection** — defect detection, material property assessment (ramp scenario validates surface properties, Phase 117: 82.4% accuracy)
- **Autonomous navigation** — obstacle properties, terrain assessment (requires new training data from driving datasets)
- **Surgical robotics** — tissue property estimation (requires medical imaging features)
- **Warehouse logistics** — object weight/fragility estimation (spring scenario validates mass, Phase 95: 81.8% accuracy)

**What it requires:**
- Domain-specific training datasets (video + ground-truth physical properties)
- Feature extraction from domain-relevant encoders (may need domain-specific fine-tuned encoders)
- Compliance validation per domain (PosDis > 0.5, triple metric convergence)
- Phase 109 showed new domains bootstrap in 22 seconds on spring-sized datasets

**Revenue model:** Per-domain-pack licensing. Annual subscription includes updates and new encoder support. Enterprise tier includes custom domain pack development.

**Dependencies:** Layer 1 (protocol spec), Layer 2 (monitoring for deployed packs).

## Layer 4: World Model OS (Year 1–2)

**What:** Orchestration layer that manages multiple domain protocols simultaneously. A single robot or camera fleet runs multiple domain packs and routes messages to the correct protocol based on context.

**Components:**
- **Protocol router** — classifies incoming scenes by domain and routes to the correct protocol instance (Phase 110 prototype: feature-distance routing)
- **Version manager** — handles protocol versioning, backward compatibility, staged rollouts
- **A/B testing** — compare protocol versions on live traffic
- **Deployment orchestration** — containerized protocol instances, auto-scaling, health checks
- **Multi-domain fusion** — combine predictions from multiple domain protocols for richer scene understanding

**What it requires:**
- Robust domain classification (Phase 110 showed feature-distance routing works; needs hardening)
- Container runtime integration (Docker, Kubernetes)
- Protocol versioning mechanism (WMCP spec Section 9 defines semantic versioning)
- Cross-domain message routing without interference (Phase 100 showed domains are isolated)

**Revenue model:** Platform fee. Usage-based pricing on message volume × number of active domain protocols. Enterprise SLA with guaranteed latency and uptime.

**Dependencies:** Layer 3 (multiple domain packs to route between), Layer 2 (monitoring for routing quality).

## Layer 5: Marketplace (Year 2–3)

**What:** Third-party ecosystem where encoder vendors and domain specialists publish certified components.

**Components:**
- **Encoder marketplace** — vision model vendors publish certified projection layers for their encoders. Certification requires passing the compliance suite (Phase 108: minimum viable at 886K parameters, takes minutes to validate)
- **Domain pack marketplace** — vertical specialists publish domain packs for their industries
- **Compliance certification service** — automated testing that certifies encoder-domain combinations
- **Revenue sharing** — marketplace takes percentage of domain pack sales

**What it requires:**
- Certification infrastructure (automated compliance testing, model card generation)
- HuggingFace Hub integration (Phase 118: model card templates ready)
- Trust and safety (adversarial robustness testing from Phase 112)
- SDK for third-party encoder integration (wmcp.onboarding module exists)

**Revenue model:** Marketplace commission (15–30% of third-party revenue). Certification fee for new encoder listings. Premium placement for featured encoders/domains.

**Dependencies:** Layer 4 (OS to deploy marketplace components on), established user base from Layers 1–3.

## Critical Path

```
Layer 1 (NOW) ──→ Layer 2 (Month 3) ──→ Layer 3 (Month 6) ──→ Layer 4 (Year 1) ──→ Layer 5 (Year 2)
     │                    │                     │                     │
     │                    │                     │                     └── Marketplace economics
     │                    │                     └── Domain expertise + training data
     │                    └── Streaming infrastructure
     └── Research validation + community adoption
```

Each layer depends on the previous. Skipping layers creates technical debt. The expansion is designed so that each layer generates revenue (or adoption) independently — Layer 3 can be profitable without Layer 4 existing.
