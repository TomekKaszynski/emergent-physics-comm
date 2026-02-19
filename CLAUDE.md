# Visual World Model Project

Multi-agent visual world model. Two agents observe 3D physics sim from different views, communicate through learned bottleneck, predict future states via JEPA.

## Files
- `physics_sim.py` — Physics environments + data generation
- `world_model.py` — Neural networks (slot attention, JEPA, communication)
- `run_all.py` — Experiment functions (run_phaseXX)
- `EXPERIMENTS.md` — Chronological experiment log and current state
- `RESEARCH_SYNTHESIS.md` — Literature review and implementation plans
- `results/` — Visualizations (phaseXX_description.png) and checkpoints (phaseXX_model.pt)

## Session Start
Read these in order:
1. This file (CLAUDE.md) — invariants and rules
2. EXPERIMENTS.md — what happened, current state, what to do next
3. RESEARCH_SYNTHESIS.md — only if task needs literature context

## Commands
- Train: `PYTHONUNBUFFERED=1 python3 -c "from run_all import run_phaseXX; run_phaseXX()"`
- Always use PYTHONUNBUFFERED=1 for live output
- All print statements must use flush=True

## Hardware
- M3 MacBook Pro, MPS acceleration, float32 only (never float16)
- PYTORCH_ENABLE_MPS_FALLBACK=1
- num_workers=0 or 2, disable pin_memory
- torch.mps.empty_cache() every ~100 steps for long runs

## Never Silently Change
hidden_dim, num_layers, num_slots, learning_rate, batch_size, slot_dim, bottleneck_dim, encoder stride, decoder architecture, loss function, optimizer. Create new configs instead of modifying existing ones.

## Workflow
- Use Plan Mode for architecture or hyperparameter changes
- Before editing: summarize planned changes and why
- After edits: list ALL changes made
- One task per session. Summarize findings before ending.
- Update EXPERIMENTS.md at end of every session
- `git commit` before and after each experiment
- Use descriptive commit messages: "Phase 26g: constant LR, port ref fixes"
- If something breaks: `git diff` to see what changed, `git stash` or `git revert` to recover

## Run Discipline
- Define exit criteria BEFORE launching any run:
  - What proves success? What proves failure? Max runtime?
- Build exit criteria INTO the training loop (auto-stop on success/failure)
- Check criteria at regular intervals. Stop when answer is clear.
- Don't run to completion "just to see."
- Print status every 10 epochs: epoch, loss, key metrics, elapsed time
- Log every run to EXPERIMENTS.md: phase, config change, metrics, verdict