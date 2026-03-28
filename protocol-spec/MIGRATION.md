# Migration & Integration Guide

WMCP is additive, not a replacement. It sits alongside existing protocols.

## Integrating with VDA 5050

VDA 5050 handles AGV task-level coordination (waypoints, actions, state). WMCP handles perception-level communication between the vision models on those AGVs.

**Architecture:**
```
VDA 5050 Master ←→ AGV Fleet Controller
                        ↓
              WMCP Protocol Layer
              ↙        ↓        ↘
         AGV-1        AGV-2       AGV-3
        (V-JEPA)     (DINOv2)    (CLIP)
```

**Integration points:**
- VDA 5050 sends task commands ("go to station B, pick up pallet")
- WMCP enables the AGVs' cameras to share physical scene understanding ("this pallet is heavy, surface is slippery")
- VDA 5050 state messages include a `wmcp_context` field with protocol tokens

**Coexistence:** VDA 5050 uses MQTT for transport. WMCP messages can be embedded in VDA 5050 `information` fields or published on a parallel MQTT topic (`/wmcp/messages`).

## Integrating with Anthropic MCP

MCP (Model Context Protocol) enables LLMs to call external tools. WMCP enables vision models to communicate about what they see. They operate at different abstraction levels.

**Architecture:**
```
Claude (LLM) ←—MCP—→ Tools, APIs, Databases
                          ↓
                   WMCP Protocol Layer
                   ↙              ↘
          Camera-1 (V-JEPA)   Camera-2 (DINOv2)
```

**Integration pattern:**
- MCP provides an `observe_scene` tool that queries WMCP agents
- WMCP agents encode their visual observations as discrete tokens
- The MCP response includes: WMCP tokens + decoded property predictions
- The LLM uses the WMCP output as grounded world-state context

**Example MCP tool:**
```json
{
  "name": "wmcp_observe",
  "description": "Query WMCP agents about physical scene properties",
  "input_schema": {
    "type": "object",
    "properties": {
      "scene_id": {"type": "string"},
      "domain": {"type": "string", "enum": ["spring", "ramp", "fall"]}
    }
  }
}
```

## Integrating with ROS 2 DDS

WMCP messages map directly to ROS 2 topics via DDS.

**Topic mapping:**
| WMCP Concept | ROS 2/DDS Equivalent |
|-------------|---------------------|
| Message bus | Topic (`/wmcp/messages`) |
| Agent sender | Publisher node |
| Agent receiver | Subscriber node |
| Message envelope | Custom message type |
| Domain routing | Namespace (`/wmcp/spring/messages`) |

**QoS settings:**
- Reliability: RELIABLE (for training), BEST_EFFORT (for real-time inference)
- Durability: VOLATILE (messages are ephemeral)
- History: KEEP_LAST(1) (only latest message matters)

**Implementation:** See `wmcp_ros2/` for complete ROS 2 node wrappers and message definitions.

## Migration Checklist

For any integration:

1. **Identify the communication layer.** WMCP operates at the perception/representation layer, below task coordination and above raw sensor data.
2. **Map message transport.** WMCP is transport-agnostic. Map to your existing transport (MQTT, DDS, HTTP, WebSocket).
3. **Keep existing protocols.** WMCP adds a capability, it doesn't replace anything.
4. **Train per-domain.** Each physics domain needs its own WMCP instance. Cross-domain transfer doesn't work.
5. **Monitor integration.** Use WMCP monitoring module to track message quality alongside existing observability.
