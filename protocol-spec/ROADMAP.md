# WMCP Roadmap

## v0.1 — Specification Draft (current)

Protocol specification based on experimental evidence from 1,350+ training runs.

**Delivered:**
- Message format definition (K=3 vocabulary, L positions per agent)
- Encoding procedure (frozen encoder → projection → Gumbel-Softmax)
- Alignment procedure (co-training, not zero-shot)
- Compositionality requirements (PosDis ≥ 0.5, triple metric validation)
- Robustness profile (noise tolerance, latency benchmarks)
- Scalability bounds (1–16 agents, 3 architectures)
- Domain specificity characterization
- Experimental evidence summary

## v0.2 — Reference Implementation

Self-contained PyTorch implementation of the full protocol stack.

**Planned:**
- `wmcp.Sender` — encoder-agnostic sender module
- `wmcp.Receiver` — pairwise comparison receiver
- `wmcp.Protocol` — training loop with temperature annealing, receiver reset, early stopping
- `wmcp.metrics` — PosDis, TopSim, BosDis computation
- Feature extraction utilities for V-JEPA 2, DINOv2, CLIP
- Example training scripts for Physics 101 scenarios
- Unit tests for all modules

**Requirements:** PyTorch ≥ 2.0, numpy, scipy, scikit-learn.

## v0.3 — Onboarding CLI

Command-line tool for adding new encoders to a trained protocol.

**Planned:**
- `wmcp add-encoder --features path/to/features.pt --protocol path/to/protocol.pt`
- Automatic co-training with frozen existing agents
- Compositionality validation and compliance report generation
- `wmcp validate --protocol path/to/protocol.pt` — run full certification suite
- `wmcp info --protocol path/to/protocol.pt` — display protocol metadata

## v0.4 — Certification Test Suite

Automated testing for protocol compliance.

**Planned:**
- Compositionality certification (PosDis, TopSim, BosDis thresholds)
- Noise robustness testing (configurable σ sweep)
- Object-level holdout validation
- Monotonicity verification for mass-symbol mappings
- Regression testing against previous protocol versions
- CI/CD integration (GitHub Actions workflow)

## v1.0 — Production Release

Stable specification with formal versioning and backward compatibility guarantees.

**Requirements before v1.0:**
- Multi-property validation with truly independent attributes (not mass-derived)
- Validation on at least one non-physics domain (e.g., manipulation affordances)
- In-the-wild video validation beyond Physics 101
- Formal convergence analysis or convergence bounds
- Protocol negotiation mechanism (agents discover each other's capabilities)
- Backward compatibility policy: v1.x agents must communicate with v1.0 agents

## Future Directions

### ROS 2 Integration
- Native ROS 2 node wrappers (`wmcp_sender_node`, `wmcp_receiver_node`)
- Standard message types for `/wmcp/messages` topic
- Integration with ROS 2 lifecycle management
- Proof-of-concept completed with async pub-sub (Phase 106: 0.70ms latency, 1,424 comms/s)

### Multi-Domain Routing
- Protocol router that dispatches messages to domain-specific decoders
- Domain identification from message statistics
- Shared encoder, multiple domain-specific projection layers

### Continuous Onboarding
- Online fine-tuning of new agents without stopping the protocol
- Gradual weight interpolation for smooth agent replacement
- Version negotiation between agents running different protocol versions

### Formal Verification
- Information-theoretic bounds on compositionality emergence
- Minimum dataset size for reliable convergence
- Relationship between codebook size K, population size N, and PosDis ceiling
