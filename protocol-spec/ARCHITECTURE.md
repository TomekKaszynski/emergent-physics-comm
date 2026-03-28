# WMCP Architecture

## Communication Loop

The full communication round for a pairwise comparison task between two scenes:

```mermaid
graph TD
    subgraph Scene_A["Scene A"]
        VA[Video/Image Frames]
    end

    subgraph Scene_B["Scene B"]
        VB[Video/Image Frames]
    end

    subgraph Agent_1["Agent 1 (e.g., V-JEPA 2)"]
        E1[Frozen Encoder<br/>1024-dim features]
        P1[Projection Layer<br/>Conv1d → Linear]
        G1[Gumbel-Softmax<br/>K=3 per position]
    end

    subgraph Agent_2["Agent 2 (e.g., DINOv2)"]
        E2[Frozen Encoder<br/>384-dim features]
        P2[Projection Layer<br/>Conv1d → Linear]
        G2[Gumbel-Softmax<br/>K=3 per position]
    end

    subgraph Message_Bus["Discrete Message Bus"]
        MA["Message A<br/>[s₁, s₂, ..., sₗ] per agent"]
        MB["Message B<br/>[s₁, s₂, ..., sₗ] per agent"]
    end

    subgraph Receiver["Receiver (Shared)"]
        R[MLP Decoder<br/>msg_dim × 2 → hidden → 1]
        PRED[Property Comparison<br/>Prediction]
    end

    VA --> E1 --> P1 --> G1
    VB --> E1
    VA --> E2 --> P2 --> G2
    VB --> E2

    G1 -->|"one-hot symbols"| MA
    G2 -->|"one-hot symbols"| MA
    G1 -->|"one-hot symbols"| MB
    G2 -->|"one-hot symbols"| MB

    MA --> R
    MB --> R
    R --> PRED
```

### Data Flow Detail

1. **Feature extraction:** Frozen encoder processes video frames → continuous features (B, T, D).
2. **Temporal encoding:** Conv1d over time dimension → pooled representation (B, 128).
3. **Head projection:** Independent linear heads per message position → logits (B, K) per position.
4. **Discretization:** Gumbel-Softmax (training) or argmax (inference) → one-hot symbol per position.
5. **Message assembly:** Concatenate all agent symbols → joint message vector.
6. **Decoding:** Receiver MLP takes message pair (scene A, scene B) → comparison logit.

## Onboarding Flow

How a new encoder joins an existing trained protocol:

```mermaid
graph LR
    subgraph Existing_Protocol["Existing Trained Protocol"]
        EA[Agent 1 Sender<br/>FROZEN]
        EB[Agent 2 Sender<br/>FROZEN]
        RX[Receiver<br/>FROZEN]
    end

    subgraph New_Agent["New Agent (e.g., CLIP)"]
        EN[New Frozen Encoder]
        PN[New Projection Layer<br/>TRAINABLE]
        GN[Gumbel-Softmax]
    end

    subgraph Training["Co-Training Loop"]
        TASK[Cooperative Task<br/>Pairwise Comparison]
        LOSS[Loss Signal<br/>BCE]
        GRAD[Gradient<br/>→ New Projection Only]
    end

    EN --> PN --> GN
    GN -->|"new messages"| TASK
    EA -->|"existing messages"| TASK
    EB -->|"existing messages"| TASK
    TASK --> RX --> LOSS
    LOSS --> GRAD --> PN

    style EA fill:#ccc,stroke:#999
    style EB fill:#ccc,stroke:#999
    style RX fill:#ccc,stroke:#999
    style PN fill:#afa,stroke:#090
```

### Onboarding Steps

1. Freeze all existing agent senders and the receiver.
2. Initialize new projection layer with random weights.
3. Co-train the new projection layer using the existing cooperative task.
4. The new agent learns to produce messages compatible with the existing protocol.
5. Convergence: 50 training steps to reach 90% of base accuracy (Phase 104, 10/10 seeds).

## Multi-Agent Topology

```mermaid
graph TB
    subgraph Population["N-Agent Population"]
        A1["Agent 1<br/>(V-JEPA 2)"]
        A2["Agent 2<br/>(DINOv2)"]
        A3["Agent 3<br/>(V-JEPA 2)"]
        A4["Agent 4<br/>(DINOv2)"]
    end

    subgraph Views["Temporal View Split"]
        F1["Frames 0-1"]
        F2["Frames 2-3"]
        F3["Frames 4-5"]
        F4["Frames 6-7"]
    end

    subgraph Messages["Joint Message"]
        M1["[s₁, s₂]<br/>Agent 1"]
        M2["[s₃, s₄]<br/>Agent 2"]
        M3["[s₅, s₆]<br/>Agent 3"]
        M4["[s₇, s₈]<br/>Agent 4"]
        JM["Concatenated:<br/>[s₁, s₂, s₃, s₄, s₅, s₆, s₇, s₈]"]
    end

    F1 --> A1 --> M1
    F2 --> A2 --> M2
    F3 --> A3 --> M3
    F4 --> A4 --> M4

    M1 --> JM
    M2 --> JM
    M3 --> JM
    M4 --> JM
```

### Topology Notes

- Each agent observes a **different temporal window** of the same scene.
- Agents produce messages **independently** — no inter-agent communication during encoding.
- The joint message is the **concatenation** of all agent messages.
- Heterogeneous populations alternate architectures: agents 0, 2 use architecture A; agents 1, 3 use architecture B.
- For 3-architecture populations: agents cycle through architectures (A, B, C, A, B, C, ...).
- The receiver sees only the joint message — it has no information about which agent produced which symbol positions.

## Encoder Compatibility Matrix

Based on experimental validation (Phase 96, K=3, 2-agent, 10 seeds):

```
                V-JEPA 2    DINOv2    CLIP ViT-L/14
V-JEPA 2         0.777       0.764       0.737
DINOv2           0.764       0.661       0.657
CLIP ViT-L/14    0.737       0.657       0.547
```

Values are mean PosDis. All pairings achieve PosDis > 0.5 (certification threshold). Pairings with V-JEPA 2 consistently achieve the highest compositionality due to its temporal physics-encoding features.
