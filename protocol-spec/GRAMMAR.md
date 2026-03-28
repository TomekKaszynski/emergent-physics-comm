# WMCP Formal Grammar

EBNF specification for WMCP message format.

## Message Grammar

```ebnf
(* WMCP Message Grammar — EBNF *)

wmcp_envelope    = header , message , checksum ;

header           = version , domain_id , agent_id , sequence , timestamp ;
version          = major , "." , minor , "." , patch ;
major            = non_negative_integer ;
minor            = non_negative_integer ;
patch            = non_negative_integer ;
domain_id        = string ;                (* e.g., "physics_spring" *)
agent_id         = non_negative_integer ;
sequence         = non_negative_integer ;
timestamp        = float ;                 (* UNIX epoch, seconds *)

message          = vocab_size , n_positions , { position } ;
vocab_size       = positive_integer ;      (* K — symbols per position *)
n_positions      = positive_integer ;      (* L — positions per agent *)
position         = symbol ;
symbol           = integer ;              (* value in [0, K-1] *)

checksum         = uint32 ;               (* CRC32 of header + message *)

(* Constraints *)
(* len(positions) = n_positions *)
(* for all p in positions: 0 <= p.symbol < vocab_size *)
(* Default: vocab_size = 3, n_positions = 2 *)
```

## Wire Format (JSON)

```json
{
  "version": "0.1.0",
  "domain": "physics_spring",
  "agent_id": 0,
  "seq": 42,
  "ts": 1711612800.0,
  "K": 3,
  "L": 2,
  "tokens": [1, 2],
  "encoder": "vjepa2"
}
```

## Binary Wire Format (Compact)

For bandwidth-constrained environments (drones, underwater):

```
Byte 0:     version_major (uint8)
Byte 1:     version_minor (uint8)
Byte 2:     agent_id (uint8)
Byte 3:     K (uint8, vocab_size)
Byte 4:     L (uint8, n_positions)
Byte 5..5+L: tokens (uint8 each)
Byte 5+L..9+L: timestamp (float32, big-endian)
Byte 9+L..13+L: checksum (uint32, CRC32)
```

Total size for K=3, L=2: **15 bytes** per message.

## Multi-Agent Joint Message

```ebnf
joint_message = { agent_message } ;
agent_message = agent_id , message ;

(* Total positions in joint message = N_agents * L *)
(* Total capacity = N_agents * L * log2(K) bits *)
```

## Validation Rules

1. `len(tokens) == L` — message length must match declared positions
2. `all(0 <= t < K for t in tokens)` — symbols must be in vocabulary
3. `version.major` must match between sender and receiver
4. `checksum` must match CRC32 of header + message bytes
5. `timestamp` must be within acceptable clock skew (default: 5 seconds)
