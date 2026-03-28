# WMCP Security Assessment

## Threat Model

WMCP operates in environments where multiple vision models communicate through discrete messages. The threat model considers:

1. **Adversarial agents** — a compromised model injects malicious messages
2. **Message interception** — an attacker reads messages on the bus
3. **Model poisoning** — a corrupted encoder produces subtly wrong features
4. **Denial of service** — flooding the message bus with invalid messages

## Known Risks and Mitigations

### Risk 1: Pickle Deserialization (CRITICAL)

**Issue:** `torch.load()` uses Python pickle by default, which can execute arbitrary code during deserialization.

**Affected code:** All feature loading calls use `torch.load(path, weights_only=False)`.

**Mitigation:**
- Set `weights_only=True` for all production deployments
- Validate file checksums before loading
- Only load model files from trusted sources
- Future: migrate to safetensors format

**Status:** Documented, not yet fixed in codebase (research code uses `weights_only=False` for compatibility with saved dicts containing non-tensor data).

### Risk 2: No Message Authentication

**Issue:** Messages on the pub-sub bus have no authentication. Any process can publish messages claiming to be any agent.

**Affected code:** `wmcp_ros2/publisher_node.py`, `wmcp/pubsub.py`

**Mitigation:**
- In production, use authenticated ROS 2 topics (SROS2)
- Add HMAC signing to WMCPMessage (planned for v0.2)
- Validate agent_id against enrollment registry

**Status:** No authentication implemented. Planned for v0.2.

### Risk 3: Adversarial Message Injection

**Issue:** A compromised agent can send random or adversarial tokens to degrade consensus accuracy.

**Affected code:** All receiver/decoder components.

**Mitigation:**
- Entropy-based anomaly detection (Phase 112: 100% detection rate)
- Agent quarantine system (Phase 125)
- Note: quarantine has false positives that need tuning

**Status:** Detection implemented. Quarantine needs refinement.

### Risk 4: CLI Argument Injection

**Issue:** CLI commands accept file paths that could reference sensitive locations.

**Affected code:** `wmcp/cli.py`

**Mitigation:**
- Path validation and sandboxing (not yet implemented)
- CLI runs with user permissions only
- No network operations in CLI (all local)

**Status:** Low risk (CLI is local-only). Path validation planned.

### Risk 5: No Input Validation on Features

**Issue:** The protocol accepts arbitrary tensors as "features." Malformed tensors could cause crashes or unexpected behavior.

**Affected code:** `wmcp/protocol.py`, `wmcp/projection.py`

**Mitigation:**
- Add shape validation in `Protocol.encode()`
- Add dtype checking (reject non-float tensors)
- Add NaN/Inf detection

**Status:** Not implemented. Planned for v0.2.

### Risk 6: Information Leakage via Messages

**Issue:** Discrete messages encode physical properties. An attacker who intercepts messages can infer scene properties.

**Assessment:** Messages use K=3 symbols with L=2 positions = 9 possible messages. Information content is 6.3 bits — negligible for sensitive applications. However, the MI matrix shows messages correlate with mass (MI ≈ 0.5 nats), so property values can be partially reconstructed.

**Mitigation:**
- For sensitive deployments, encrypt messages on the wire
- The discrete bottleneck naturally limits information leakage (6.3 bits maximum)

**Status:** Inherent to the protocol design. Encryption is a deployment concern.

## Security Practices

1. **Dependencies:** torch, numpy, scipy only. No network dependencies at runtime.
2. **No telemetry:** The package does not phone home.
3. **No credentials:** No API keys, tokens, or secrets in the codebase.
4. **No hardcoded paths:** All paths are user-specified or relative.
5. **Test suite:** 23 unit tests run on synthetic data (no external data dependencies).

## Reporting Vulnerabilities

Report security issues to: t.kaszynski@proton.me

Please include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact assessment
- Suggested mitigation if available
