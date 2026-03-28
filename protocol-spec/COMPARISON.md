# Protocol Comparison

## WMCP vs Existing Standards

| | WMCP | MCP (Anthropic) | VDA 5050 | ECP | Cisco LSTP/CSTP |
|--|------|-----------------|----------|-----|-----------------|
| **Abstraction level** | Representation (latent space) | Application (tool calls) | Task (logistics commands) | Emergent (research) | Network (camera streams) |
| **Communication type** | Discrete compositional symbols | JSON-RPC function calls | JSON task graphs | Continuous/discrete learned | Video stream protocols |
| **Multi-architecture** | Yes (V-JEPA, DINOv2, CLIP validated) | N/A (single LLM) | N/A (AGV controllers) | Research only | Vendor-specific |
| **Latency** | 1.19ms (CPU) | ~100-500ms (API call) | ~100ms (MQTT) | Not benchmarked | ~50ms (streaming) |
| **Compression** | 5,200× vs raw features | N/A | N/A | Varies | Codec-dependent |
| **Compositionality** | Verified (PosDis, TopSim, BosDis) | Not applicable | Not applicable | Sometimes measured | Not applicable |
| **Onboarding** | 50 steps, ~seconds | Prompt engineering | Configuration file | Full retraining | SDK integration |
| **Implementation** | pip-installable, 23 tests | Open standard, SDKs | Open standard | Academic code | Proprietary |
| **Adoption** | Pre-launch (v0.1) | 97M+ downloads | Industry standard (logistics) | Academic citations | Enterprise deployments |
| **Target** | Multi-model robotics | LLM tool integration | AGV fleet management | Research | Camera networks |

## Positioning

**WMCP fills a gap that no existing protocol addresses:** communication between heterogeneous vision foundation models at the representation level.

- **MCP** operates at the application layer — it tells an LLM which tools to call. WMCP operates at the representation layer — it enables vision models to communicate what they see. These are complementary, not competing: an MCP-connected system could use WMCP internally for multi-model perception.

- **VDA 5050** is a logistics industry standard for AGV fleet coordination. It defines task-level commands (go to waypoint, pick up pallet). WMCP operates below this — it enables the perception systems on those AGVs to share physical scene understanding across different camera/model configurations.

- **ECP** (Emergent Communication Protocols) is the academic research tradition that WMCP builds on. The contribution is moving from research demonstrations to a deployable specification with versioning, compliance testing, and production infrastructure.

- **Cisco LSTP/CSTP** handles camera video stream transport. WMCP compresses what cameras see into 6.3-bit discrete messages — 5,200× more bandwidth-efficient than raw features. For distributed camera fleets, WMCP could replace feature-level communication entirely.

## Honest Assessment

WMCP is pre-launch with zero external users. MCP has 97M downloads. VDA 5050 runs real warehouses. The comparison is aspirational, not current.

What WMCP has that others don't:
1. Architecture-agnostic representation-level communication (no other protocol does this)
2. Empirical validation across 3 encoder families on real video
3. Sub-millisecond latency suitable for real-time robotics
4. A formal compositionality verification framework

What WMCP lacks:
1. External adoption
2. Production deployments
3. Enterprise hardening
4. Community beyond the author

The gap between "validated in 128 experiment phases" and "running in production" is significant but bridgeable.
