
    # ══════════════════════════════════════════════════════════
    # STAGE 4: Perceive with split observation (A: 0-40, B: 40-80)
    # ══════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}", flush=True)
    print(f"STAGE 4: Split perception (A: 0-40, B: 40-80)", flush=True)
    print(f"{'=' * 60}", flush=True)
    t4 = time.time()

    del states_t_raw, states_tp1_raw, slots_t, slots_tp1, actions_t
    del tr_slots_t, tr_slots_tp1, tr_actions
    del all_slots_flat, all_pos_flat, gt_pos
    del tr_dec_slots, tr_dec_pos, vl_dec_slots, vl_dec_pos
    if device.type == 'mps':
        torch.mps.empty_cache()

    half = n_frames // 2  # 40

    # Perceive each sequence from both agents' perspectives
    train_feats_A = []  # Agent A sees frames 0-40
    train_feats_B = []  # Agent B sees frames 40-80
    for seq_i in range(n_train_seq):
        feats_a, _ = perceive_sequence(
            all_objects_init[seq_i], n_frames, seed_offset=10000 + seq_i,
            obs_starts=[0] * n_obj, obs_ends=[half] * n_obj)
        feats_b, _ = perceive_sequence(
            all_objects_init[seq_i], n_frames, seed_offset=10000 + seq_i,
            obs_starts=[half] * n_obj)
        train_feats_A.append(feats_a)
        train_feats_B.append(feats_b)
        if (seq_i + 1) % 500 == 0:
            print(f"│  Perceived {seq_i+1}/{n_train_seq} (both views)", flush=True)
    print(f"└─ Stage 4 done [{time.time()-t4:.0f}s]", flush=True)

    # ══════════════════════════════════════════════════════════
    # STAGE 5: Train Communication (shared sender/receiver)
    # ══════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}", flush=True)
    print(f"STAGE 5: Train shared communication (both views)", flush=True)
    print(f"{'=' * 60}", flush=True)
    t5 = time.time()

    n_feat = 8  # 7 physics + 1 obs_confidence

    # ── Mass pathway (Gumbel, 8-dim input) ──
    class MassSender(nn.Module):
        def __init__(self):
            super().__init__()
            self.head = nn.Sequential(
                nn.Linear(n_feat, 64), nn.ReLU(),
                nn.Linear(64, 32), nn.ReLU(),
                nn.Linear(32, vocab_size))
        def forward(self, x, tau=1.0, hard=False):
            logits = self.head(x)
            if hard:
                return F.one_hot(logits.argmax(-1), vocab_size).float()
            return F.gumbel_softmax(logits, tau=tau, hard=True)

    class MassReceiver(nn.Module):
        """Count-agnostic: per-object scoring."""
        def __init__(self, embed_dim=32, hidden=64):
            super().__init__()
            self.embed = nn.Linear(vocab_size, embed_dim)
            self.head = nn.Sequential(
                nn.Linear(embed_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        def forward(self, tokens):
            emb = F.relu(self.embed(tokens))
            scores = self.head(emb).squeeze(-1)
            return scores

    # ── Elasticity pathway (FSQ, 8-dim input) ──
    class ElastSender(nn.Module):
        def __init__(self):
            super().__init__()
            self.head = nn.Sequential(
                nn.Linear(n_feat, 64), nn.ReLU(),
                nn.Linear(64, 32), nn.ReLU(),
                nn.Linear(32, 1))
        def forward(self, x, hard=False, **kwargs):
            raw = self.head(x)
            scaled = torch.sigmoid(raw.squeeze(-1)) * (vocab_size - 1)
            quantized = torch.round(scaled).clamp(0, vocab_size - 1)
            if hard:
                return quantized
            return scaled + (quantized - scaled).detach()

    class ElastReceiver(nn.Module):
        """Count-agnostic: per-object classification."""
        def __init__(self, embed_dim=32, hidden=64):
            super().__init__()
            self.embed = nn.Linear(1, embed_dim)
            self.drop = nn.Dropout(0.3)
            self.head = nn.Sequential(
                nn.Linear(embed_dim, hidden), nn.ReLU(),
                nn.Dropout(0.3), nn.Linear(hidden, 1))
        def forward(self, tokens):
            emb = F.relu(self.embed(tokens.unsqueeze(-1)))
            emb = self.drop(emb)
            logits = self.head(emb).squeeze(-1)
            return logits

    torch.manual_seed(42)
    mass_sender = MassSender().to(device)
    mass_receiver = MassReceiver().to(device)
    elast_sender = ElastSender().to(device)
    elast_receiver = ElastReceiver().to(device)

    ms_p = sum(p.numel() for p in mass_sender.parameters())
    mr_p = sum(p.numel() for p in mass_receiver.parameters())
    es_p = sum(p.numel() for p in elast_sender.parameters())
    er_p = sum(p.numel() for p in elast_receiver.parameters())
    print(f"│  Mass pathway: {ms_p}+{mr_p} = {ms_p+mr_p} params", flush=True)
    print(f"│  Elast pathway: {es_p}+{er_p} = {es_p+er_p} params", flush=True)

    # Combine both views into training data (2x samples)
    # Each sequence contributes two observations: A's view and B's view
    all_feats_combined = []
    all_heavy_combined = []
    all_elast_combined = []
    for seq_i in range(n_train_seq):
        all_feats_combined.append(train_feats_A[seq_i])
        all_feats_combined.append(train_feats_B[seq_i])
        all_heavy_combined.append(all_heavy_idx[seq_i])
        all_heavy_combined.append(all_heavy_idx[seq_i])
        all_elast_combined.append(all_elasticities[seq_i])
        all_elast_combined.append(all_elasticities[seq_i])

    n_combined = len(all_feats_combined)
    n_comm_train = int(0.8 * n_combined)
    train_feats_tensor = torch.tensor(all_feats_combined, dtype=torch.float32).to(device)
    train_heavy_tensor = torch.tensor(all_heavy_combined, dtype=torch.long).to(device)
    train_elast_tensor = torch.tensor(
        [[1.0 if e == 1.0 else 0.0 for e in el] for el in all_elast_combined],
        dtype=torch.float32).to(device)

    tr_feats = train_feats_tensor[:n_comm_train]
    tr_heavy = train_heavy_tensor[:n_comm_train]
    tr_elast = train_elast_tensor[:n_comm_train]
    vl_feats = train_feats_tensor[n_comm_train:]
    vl_heavy = train_heavy_tensor[n_comm_train:]
    vl_elast = train_elast_tensor[n_comm_train:]

    comm_batch = 64

    # ── Train mass pathway (400 epochs) ──
    print(f"│  Training mass pathway (400 epochs, τ 2.0→0.5)", flush=True)
    mass_params = list(mass_sender.parameters()) + list(mass_receiver.parameters())
    mass_opt = torch.optim.Adam(mass_params, lr=3e-4)
    best_mass_acc = 0.0
    best_mass_state = None

    for epoch in range(1, 401):
        mass_sender.train(); mass_receiver.train()
        tau = 2.0 - (2.0 - 0.5) * (epoch - 1) / 399
        ep_perm = torch.randperm(n_comm_train, device=device)
        for start in range(0, n_comm_train, comm_batch):
            idx = ep_perm[start:start + comm_batch]
            B = len(idx)
            feats_batch = tr_feats[idx]
            heavy_batch = tr_heavy[idx]
            tokens = torch.zeros(B, n_obj, vocab_size, device=device)
            for oi in range(n_obj):
                tokens[:, oi, :] = mass_sender(feats_batch[:, oi, :], tau=tau)
            logits = mass_receiver(tokens)
            loss = F.cross_entropy(logits, heavy_batch)
            mass_opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mass_params, 1.0)
            mass_opt.step()
        if epoch % 40 == 0 or epoch == 1:
            mass_sender.eval(); mass_receiver.eval()
            with torch.no_grad():
                vl_tokens = torch.zeros(len(vl_heavy), n_obj, vocab_size, device=device)
                for oi in range(n_obj):
                    vl_tokens[:, oi, :] = mass_sender(vl_feats[:, oi, :], hard=True)
                vl_logits = mass_receiver(vl_tokens)
                ma = (vl_logits.argmax(-1) == vl_heavy).float().mean().item()
            if ma > best_mass_acc:
                best_mass_acc = ma
                best_mass_state = {
                    'sender': {k: v.cpu().clone() for k, v in mass_sender.state_dict().items()},
                    'receiver': {k: v.cpu().clone() for k, v in mass_receiver.state_dict().items()},
                }
            print(f"│    Mass Epoch {epoch:3d}: τ={tau:.2f}, acc={ma:.3f} "
                  f"(best={best_mass_acc:.3f})", flush=True)

    mass_sender.load_state_dict(best_mass_state['sender'])
    mass_receiver.load_state_dict(best_mass_state['receiver'])
    mass_sender.to(device).eval(); mass_receiver.to(device).eval()
    print(f"│  Mass done: best={best_mass_acc*100:.1f}%", flush=True)

    # ── Train elasticity pathway (400 epochs, FSQ, reinit every 100) ──
    print(f"│  Training elasticity pathway (400 epochs, FSQ, reinit every 100)", flush=True)

    def make_elast_opt():
        return torch.optim.Adam([
            {'params': elast_sender.parameters(), 'lr': 3e-4, 'weight_decay': 0.0},
            {'params': elast_receiver.parameters(), 'lr': 3e-4, 'weight_decay': 0.01},
        ])

    elast_opt = make_elast_opt()
    best_elast_acc = 0.0
    best_elast_state = None

    for epoch in range(1, 401):
        if epoch > 1 and (epoch - 1) % 100 == 0:
            print(f"│    Reinit ElastReceiver at epoch {epoch}", flush=True)
            elast_receiver = ElastReceiver().to(device)
            elast_opt = make_elast_opt()

        elast_sender.train(); elast_receiver.train()
        ep_perm = torch.randperm(n_comm_train, device=device)
        for start in range(0, n_comm_train, comm_batch):
            idx = ep_perm[start:start + comm_batch]
            B = len(idx)
            feats_batch = tr_feats[idx]
            elast_batch = tr_elast[idx]
            tokens = torch.zeros(B, n_obj, device=device)
            for oi in range(n_obj):
                tokens[:, oi] = elast_sender(feats_batch[:, oi, :])
            logits = elast_receiver(tokens)
            loss = F.binary_cross_entropy_with_logits(logits, elast_batch)
            elast_opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(elast_sender.parameters()) + list(elast_receiver.parameters()), 1.0)
            elast_opt.step()
        if epoch % 40 == 0 or epoch == 1:
            elast_sender.eval(); elast_receiver.eval()
            with torch.no_grad():
                vl_tokens = torch.zeros(len(vl_elast), n_obj, device=device)
                for oi in range(n_obj):
                    vl_tokens[:, oi] = elast_sender(vl_feats[:, oi, :], hard=True)
                vl_logits = elast_receiver(vl_tokens)
                ea = ((vl_logits > 0).float() == vl_elast).float().mean().item()
            if ea > best_elast_acc:
                best_elast_acc = ea
                best_elast_state = {
                    'sender': {k: v.cpu().clone() for k, v in elast_sender.state_dict().items()},
                    'receiver': {k: v.cpu().clone() for k, v in elast_receiver.state_dict().items()},
                }
            print(f"│    Elast Epoch {epoch:3d}: acc={ea:.3f} "
                  f"(best={best_elast_acc:.3f})", flush=True)

    elast_sender.load_state_dict(best_elast_state['sender'])
    elast_receiver.load_state_dict(best_elast_state['receiver'])
    elast_sender.to(device).eval(); elast_receiver.to(device).eval()
    print(f"│  Elast done: best={best_elast_acc*100:.1f}%", flush=True)
    print(f"│  Final: mass={best_mass_acc*100:.1f}%, elast={best_elast_acc*100:.1f}%", flush=True)
    print(f"└─ Stage 5 done [{time.time()-t5:.0f}s]", flush=True)

    # ══════════════════════════════════════════════════════════
    # STAGE 6: Test scenarios with split observation
    # ══════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}", flush=True)
    print(f"STAGE 6: Generate 200 test scenarios (split observation)", flush=True)
    print(f"{'=' * 60}", flush=True)
    t6 = time.time()

    del train_feats_tensor, tr_feats, tr_heavy, tr_elast, vl_feats, vl_heavy, vl_elast
    if device.type == 'mps':
        torch.mps.empty_cache()

    cpu = torch.device('cpu')
    jepa.to(cpu).eval()
    pos_decoder.to(cpu).eval()
    mass_sender.to(cpu).eval()
    mass_receiver.to(cpu).eval()
    elast_sender.to(cpu).eval()
    elast_receiver.to(cpu).eval()
    state_proj = state_proj.to(cpu)

    n_test_seq = 200
    random.seed(999)
    np.random.seed(999)

    test_scenarios = []
    for si in range(n_test_seq):
        masses = [1.0, 1.0, 1.0]
        heavy_idx = random.randint(0, n_obj - 1)
        masses[heavy_idx] = 3.0
        elasticities = [random.choice([0.5, 1.0]) for _ in range(n_obj)]

        objects = []
        for oi in range(n_obj):
            r = random.randint(6, 9)
            for _attempt in range(100):
                cx = random.uniform(r + 3, S - r - 4)
                cy = random.uniform(r + 3, S - r - 4)
                ok = True
                for prev in objects:
                    if math.hypot(cx - prev['cx'], cy - prev['cy']) < r + prev['r'] + 3:
                        ok = False; break
                if ok: break
            vx = random.uniform(-5.0, 5.0)
            vy = random.uniform(-5.0, 5.0)
            objects.append({'cx': cx, 'cy': cy, 'vx': vx, 'vy': vy,
                            'r': r, 'mass': masses[oi], 'elasticity': elasticities[oi]})

        # Agent A: frames 0-40, Agent B: frames 40-80
        feats_a, _ = perceive_sequence(objects, n_frames, seed_offset=50000 + si,
                                        obs_starts=[0]*n_obj, obs_ends=[half]*n_obj)
        feats_b, _ = perceive_sequence(objects, n_frames, seed_offset=50000 + si,
                                        obs_starts=[half]*n_obj)

        test_scenarios.append({
            'gt_heavy': heavy_idx,
            'elasticities': elasticities,
            'objects_init': copy.deepcopy(objects),
            'masses': masses,
            'feats_A': feats_a,
            'feats_B': feats_b,
        })
        if (si + 1) % 50 == 0:
            print(f"│  Processed {si+1}/{n_test_seq}", flush=True)
    print(f"└─ Stage 6 done [{time.time()-t6:.0f}s]", flush=True)

    # ══════════════════════════════════════════════════════════
    # STAGE 7: Coordinated Planning + Evaluation
    # ══════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}", flush=True)
    print(f"STAGE 7: Coordinated Planning + Evaluation", flush=True)
    print(f"{'=' * 60}", flush=True)
    t7 = time.time()

    random.seed(777)
    np.random.seed(777)
    test_targets = []
    for si in range(n_test_seq):
        test_targets.append(np.array([
            random.uniform(0.2, 0.8), random.uniform(0.2, 0.8)], dtype=np.float32))

    # --- Single-agent CEM (baseline: only B acts) ---
    def cem_single(current_state, target_obj_idx, target_pos_norm):
        cur_slots = state_proj(
            torch.tensor(current_state, dtype=torch.float32).unsqueeze(0))
        target_t = torch.tensor(target_pos_norm, dtype=torch.float32)
        mu = torch.zeros(2)
        sigma = torch.ones(2) * 0.3
        for round_i in range(n_rounds):
            forces = mu + sigma * torch.randn(n_candidates, 2)
            forces = forces.clamp(-force_range, force_range)
            with torch.no_grad():
                cand_actions = torch.zeros(n_candidates, n_obj + 2)
                cand_actions[:, target_obj_idx] = 1.0
                cand_actions[:, n_obj] = forces[:, 0]
                cand_actions[:, n_obj + 1] = forces[:, 1]
                pred_slots = jepa(cur_slots.expand(n_candidates, -1, -1).clone(),
                                  cand_actions)
                target_slots = pred_slots[:, target_obj_idx, :]
                pred_pos = pos_decoder(target_slots)
                scores = -((pred_pos - target_t) ** 2).sum(dim=-1)
            elite_idx = torch.topk(scores, n_elite).indices
            elite_forces = forces[elite_idx]
            mu = elite_forces.mean(dim=0)
            sigma = elite_forces.std(dim=0).clamp(min=0.01)
        return mu[0].item(), mu[1].item()

    def run_single_agent(objects_init, target_obj_idx, target_pos):
        objs = copy.deepcopy(objects_init)
        for step in range(K):
            current_state = objects_to_state(objs)
            fx, fy = cem_single(current_state, target_obj_idx, target_pos)
            objs[target_obj_idx]['vx'] += fx * vmax
            objs[target_obj_idx]['vy'] += fy * vmax
            physics_step(objs)
        gt_heavy = None
        for oi in range(n_obj):
            if objs[oi]['mass'] == 3.0:
                gt_heavy = oi
        final_pos = np.array([objs[gt_heavy]['cx'] / S, objs[gt_heavy]['cy'] / S])
        return np.linalg.norm((final_pos - target_pos) * S)

    # --- Joint CEM (both agents act, different objects) ---
    def cem_joint(current_state, a_obj, b_obj, target_pos_norm, gt_heavy_idx):
        """CEM for joint action: A pushes a_obj, B pushes b_obj.
        Uses sequential JEPA: A's action then B's action per step."""
        cur_slots = state_proj(
            torch.tensor(current_state, dtype=torch.float32).unsqueeze(0))
        target_t = torch.tensor(target_pos_norm, dtype=torch.float32)
        n_cand = 64  # per assignment
        n_el = 8
        # 4-dim: [a_fx, a_fy, b_fx, b_fy]
        mu = torch.zeros(4)
        sigma = torch.ones(4) * 0.3
        for round_i in range(n_rounds):
            forces = mu + sigma * torch.randn(n_cand, 4)
            forces = forces.clamp(-force_range, force_range)
            with torch.no_grad():
                # Build actions for A and B
                action_a = torch.zeros(n_cand, n_obj + 2)
                action_a[:, a_obj] = 1.0
                action_a[:, n_obj] = forces[:, 0]
                action_a[:, n_obj + 1] = forces[:, 1]
                action_b = torch.zeros(n_cand, n_obj + 2)
                action_b[:, b_obj] = 1.0
                action_b[:, n_obj] = forces[:, 2]
                action_b[:, n_obj + 1] = forces[:, 3]
                # Sequential JEPA: A's action then B's action
                pred = cur_slots.expand(n_cand, -1, -1).clone()
                pred = jepa(pred, action_a)
                pred = jepa(pred, action_b)
                # Evaluate: heavy object position
                heavy_slots = pred[:, gt_heavy_idx, :]
                pred_pos = pos_decoder(heavy_slots)
                scores = -((pred_pos - target_t) ** 2).sum(dim=-1)
            elite_idx = torch.topk(scores, n_el).indices
            elite_forces = forces[elite_idx]
            mu = elite_forces.mean(dim=0)
            sigma = elite_forces.std(dim=0).clamp(min=0.01)
        return mu[0].item(), mu[1].item(), mu[2].item(), mu[3].item()

    def run_coordinated(objects_init, comm_heavy, target_pos):
        """Both agents act. Enumerate (a_obj, b_obj) pairs, pick best."""
        best_dist = float('inf')
        best_result = None
        current_state = objects_to_state(objects_init)
        # Try all 6 valid (a, b) assignments
        for a_obj in range(n_obj):
            for b_obj in range(n_obj):
                if a_obj == b_obj:
                    continue
                a_fx, a_fy, b_fx, b_fy = cem_joint(
                    current_state, a_obj, b_obj, target_pos, comm_heavy)
                # Simulate with GT physics
                objs = copy.deepcopy(objects_init)
                for step in range(K):
                    objs[a_obj]['vx'] += a_fx * vmax
                    objs[a_obj]['vy'] += a_fy * vmax
                    objs[b_obj]['vx'] += b_fx * vmax
                    objs[b_obj]['vy'] += b_fy * vmax
                    physics_step(objs)
                gt_heavy = None
                for oi in range(n_obj):
                    if objs[oi]['mass'] == 3.0:
                        gt_heavy = oi
                final_pos = np.array([objs[gt_heavy]['cx'] / S,
                                      objs[gt_heavy]['cy'] / S])
                dist = np.linalg.norm((final_pos - target_pos) * S)
                if dist < best_dist:
                    best_dist = dist
                    best_result = (a_obj, b_obj, a_fx, a_fy, b_fx, b_fy)
        return best_dist, best_result

    # --- Evaluate ---
    mass_A_correct = 0  # A's own view
    mass_B_correct = 0  # B's own view
    mass_fused_correct = 0  # fused (A_own + B_tokens, B_own + A_tokens)
    elast_A_correct = 0
    elast_B_correct = 0
    agree_count = 0

    coordinated_dists = []
    single_dists = []
    oracle_coord_dists = []
    oracle_single_dists = []

    for si in range(n_test_seq):
        sc = test_scenarios[si]
        gt_heavy = sc['gt_heavy']
        target_pos = test_targets[si]

        feat_a = torch.tensor(sc['feats_A'], dtype=torch.float32)
        feat_b = torch.tensor(sc['feats_B'], dtype=torch.float32)

        with torch.no_grad():
            # Agent A sends tokens (from A's view)
            m_tok_a = torch.zeros(1, n_obj, vocab_size)
            e_tok_a = torch.zeros(1, n_obj)
            for oi in range(n_obj):
                m_tok_a[0, oi, :] = mass_sender(feat_a[oi:oi+1], hard=True)
                e_tok_a[0, oi] = elast_sender(feat_a[oi:oi+1], hard=True)

            # Agent B sends tokens (from B's view)
            m_tok_b = torch.zeros(1, n_obj, vocab_size)
            e_tok_b = torch.zeros(1, n_obj)
            for oi in range(n_obj):
                m_tok_b[0, oi, :] = mass_sender(feat_b[oi:oi+1], hard=True)
                e_tok_b[0, oi] = elast_sender(feat_b[oi:oi+1], hard=True)

            # Each agent decodes other's tokens + its own
            # A's estimate: own view + B's tokens
            a_own_mass = mass_receiver(m_tok_a)  # [1, n_obj]
            a_recv_mass = mass_receiver(m_tok_b)  # B's tokens decoded by A
            a_fused_mass = (a_own_mass + a_recv_mass) / 2
            a_heavy = a_fused_mass[0].argmax().item()

            # B's estimate: own view + A's tokens
            b_own_mass = mass_receiver(m_tok_b)
            b_recv_mass = mass_receiver(m_tok_a)
            b_fused_mass = (b_own_mass + b_recv_mass) / 2
            b_heavy = b_fused_mass[0].argmax().item()

            # Individual accuracy (from own view only)
            a_heavy_own = a_own_mass[0].argmax().item()
            b_heavy_own = b_own_mass[0].argmax().item()
            mass_A_correct += int(a_heavy_own == gt_heavy)
            mass_B_correct += int(b_heavy_own == gt_heavy)
            mass_fused_correct += int(a_heavy == gt_heavy)

            # Agreement
            agree = a_heavy == b_heavy
            agree_count += int(agree)

            # Elast accuracy (per-agent)
            gt_elast = [1.0 if e == 1.0 else 0.0 for e in sc['elasticities']]
            a_elast_logits = elast_receiver(e_tok_a)
            b_elast_logits = elast_receiver(e_tok_b)
            a_elast_ok = all((a_elast_logits[0, oi] > 0).float().item() == gt_elast[oi]
                             for oi in range(n_obj))
            b_elast_ok = all((b_elast_logits[0, oi] > 0).float().item() == gt_elast[oi]
                             for oi in range(n_obj))
            elast_A_correct += int(a_elast_ok)
            elast_B_correct += int(b_elast_ok)

        # Use fused estimate for planning (agents agree via fusion)
        comm_heavy = a_heavy  # fused estimate (same for both agents in expectation)

        # 1. Coordinated planning (both agents act)
        coord_dist, _ = run_coordinated(sc['objects_init'], comm_heavy, target_pos)
        coordinated_dists.append(coord_dist)

        # 2. Single-agent planning (only B acts, like Phase 41m)
        single_dist = run_single_agent(sc['objects_init'], comm_heavy, target_pos)
        single_dists.append(single_dist)

        # 3. Oracle coordinated (GT heavy)
        oracle_dist, _ = run_coordinated(sc['objects_init'], gt_heavy, target_pos)
        oracle_coord_dists.append(oracle_dist)

        # 4. Oracle single (GT heavy)
        oracle_single_dist = run_single_agent(sc['objects_init'], gt_heavy, target_pos)
        oracle_single_dists.append(oracle_single_dist)

        if (si + 1) % 50 == 0:
            ma_f = mass_fused_correct / (si + 1) * 100
            agr = agree_count / (si + 1) * 100
            co_s = sum(1 for d in coordinated_dists if d < success_thresh) / len(coordinated_dists) * 100
            si_s = sum(1 for d in single_dists if d < success_thresh) / len(single_dists) * 100
            print(f"│    {si+1}/{n_test_seq}: fused_mass={ma_f:.0f}%, agree={agr:.0f}%, "
                  f"coord={co_s:.0f}%, single={si_s:.0f}%", flush=True)

    mass_A_acc = mass_A_correct / n_test_seq * 100
    mass_B_acc = mass_B_correct / n_test_seq * 100
    mass_fused_acc = mass_fused_correct / n_test_seq * 100
    elast_A_acc = elast_A_correct / n_test_seq * 100
    elast_B_acc = elast_B_correct / n_test_seq * 100
    agree_pct = agree_count / n_test_seq * 100
    coord_success = sum(1 for d in coordinated_dists if d < success_thresh) / n_test_seq * 100
    single_success = sum(1 for d in single_dists if d < success_thresh) / n_test_seq * 100
    oracle_coord_success = sum(1 for d in oracle_coord_dists if d < success_thresh) / n_test_seq * 100
    oracle_single_success = sum(1 for d in oracle_single_dists if d < success_thresh) / n_test_seq * 100

    print(f"\n│  Communication:", flush=True)
    print(f"│    A-only mass:  {mass_A_acc:.1f}% (frames 0-40)", flush=True)
    print(f"│    B-only mass:  {mass_B_acc:.1f}% (frames 40-80)", flush=True)
    print(f"│    Fused mass:   {mass_fused_acc:.1f}% (own + received)", flush=True)
    print(f"│    Agreement:    {agree_pct:.1f}%", flush=True)
    print(f"│    A elast:      {elast_A_acc:.1f}%", flush=True)
    print(f"│    B elast:      {elast_B_acc:.1f}%", flush=True)
    print(f"│  Planning:", flush=True)
    print(f"│    Coordinated:  {coord_success:.1f}%", flush=True)
    print(f"│    Single-agent: {single_success:.1f}%", flush=True)
    print(f"│    Oracle coord: {oracle_coord_success:.1f}%", flush=True)
    print(f"│    Oracle single:{oracle_single_success:.1f}%", flush=True)
    coord_advantage = coord_success - single_success
    print(f"│    Coordination advantage: {coord_advantage:+.1f}pp", flush=True)
    print(f"└─ Stage 7 done [{time.time()-t7:.0f}s]", flush=True)

    # ══════════════════════════════════════════════════════════
    # STAGE 8: Visualization
    # ══════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}", flush=True)
    print(f"STAGE 8: Visualization", flush=True)
    print(f"{'=' * 60}", flush=True)
    elapsed = time.time() - t0

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. Communication accuracy by agent
    ax = axes[0, 0]
    agents = ['A (0-40)', 'B (40-80)', 'Fused']
    mass_accs = [mass_A_acc, mass_B_acc, mass_fused_acc]
    elast_accs = [elast_A_acc, elast_B_acc, (elast_A_acc + elast_B_acc) / 2]
    x = np.arange(3)
    w = 0.35
    bars1 = ax.bar(x - w/2, mass_accs, w, label='Mass', color='#2196F3',
                   edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + w/2, elast_accs, w, label='Elasticity', color='#FF9800',
                   edgecolor='black', linewidth=0.5)
    for bar, val in zip(bars1, mass_accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.0f}%', ha='center', fontsize=9, color='#2196F3')
    for bar, val in zip(bars2, elast_accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.0f}%', ha='center', fontsize=9, color='#FF9800')
    ax.set_xticks(x)
    ax.set_xticklabels(agents)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Communication by Agent View')
    ax.set_ylim(0, 105)
    ax.legend()

    # 2. Planning: coordinated vs single-agent
    ax = axes[0, 1]
    planners = ['Coord\n(comm)', 'Single\n(comm)', 'Coord\n(oracle)', 'Single\n(oracle)']
    successes = [coord_success, single_success, oracle_coord_success, oracle_single_success]
    colors = ['#2196F3', '#FF9800', '#4CAF50', '#8BC34A']
    bars = ax.bar(planners, successes, color=colors, edgecolor='black', linewidth=0.5)
    for bar, val in zip(bars, successes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', fontsize=9, fontweight='bold')
    ax.set_ylabel('Success Rate (%)')
    ax.set_title(f'Planning Success (<{success_thresh:.0f}px)')
    ax.set_ylim(0, 105)

    # 3. CDF of distances
    ax = axes[0, 2]
    for dists, label, color in [(coordinated_dists, 'Coordinated', '#2196F3'),
                                 (single_dists, 'Single-agent', '#FF9800'),
                                 (oracle_coord_dists, 'Oracle coord', '#4CAF50'),
                                 (oracle_single_dists, 'Oracle single', '#8BC34A')]:
        sorted_d = np.sort(dists)
        cdf = np.arange(1, len(sorted_d) + 1) / len(sorted_d)
        ax.plot(sorted_d, cdf * 100, label=label, color=color, linewidth=2)
    ax.axvline(x=success_thresh, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Distance to target (px)')
    ax.set_ylabel('Cumulative %')
    ax.set_title('CDF: Coordinated vs Single-Agent')
    ax.legend(fontsize=8)
    ax.set_xlim(0, 40)

    # 4. Agreement and fusion benefit
    ax = axes[1, 0]
    metrics = ['A-only\nmass', 'B-only\nmass', 'Fused\nmass', 'Agreement']
    vals = [mass_A_acc, mass_B_acc, mass_fused_acc, agree_pct]
    colors_m = ['#90CAF9', '#FFCC80', '#2196F3', '#9C27B0']
    bars = ax.bar(metrics, vals, color=colors_m, edgecolor='black', linewidth=0.5)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.0f}%', ha='center', fontsize=9, fontweight='bold')
    ax.set_ylabel('%')
    ax.set_title('Multi-Agent Fusion Benefit')
    ax.set_ylim(0, 105)

    # 5. Distance distributions (histograms)
    ax = axes[1, 1]
    ax.hist(coordinated_dists, bins=25, alpha=0.6, label=f'Coord (med={np.median(coordinated_dists):.1f})',
            color='#2196F3')
    ax.hist(single_dists, bins=25, alpha=0.6, label=f'Single (med={np.median(single_dists):.1f})',
            color='#FF9800')
    ax.axvline(x=success_thresh, color='red', linestyle='--', label=f'{success_thresh:.0f}px threshold')
    ax.set_xlabel('Distance to target (px)')
    ax.set_ylabel('Count')
    ax.set_title('Distance Distributions')
    ax.legend(fontsize=8)

    # 6. Summary
    ax = axes[1, 2]
    ax.axis('off')
    summary = (
        f"Phase 44: Multi-Agent Coordination\n\n"
        f"Communication:\n"
        f"  A mass:   {mass_A_acc:.0f}%  B mass:   {mass_B_acc:.0f}%\n"
        f"  Fused:    {mass_fused_acc:.0f}%  Agreement: {agree_pct:.0f}%\n"
        f"  A elast:  {elast_A_acc:.0f}%  B elast:  {elast_B_acc:.0f}%\n\n"
        f"Planning:\n"
        f"  Coordinated:   {coord_success:.1f}%\n"
        f"  Single-agent:  {single_success:.1f}%\n"
        f"  Coord advantage: {coord_advantage:+.1f}pp\n\n"
        f"  Oracle coord:  {oracle_coord_success:.1f}%\n"
        f"  Oracle single: {oracle_single_success:.1f}%\n\n"
        f"Total time: {elapsed:.0f}s"
    )
    ax.text(0.05, 0.5, summary, transform=ax.transAxes, fontsize=11,
            fontfamily='monospace', verticalalignment='center')

    fig.suptitle(f'Phase 44: Multi-Agent Coordinated Action\n'
                 f'coord={coord_success:.0f}% single={single_success:.0f}% '
                 f'(advantage: {coord_advantage:+.0f}pp) | '
                 f'fused mass={mass_fused_acc:.0f}% agree={agree_pct:.0f}%',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/phase44_coordination.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n│  Saved results/phase44_coordination.png", flush=True)

    print(f"\n{'=' * 70}", flush=True)
    if coord_success > single_success + 10:
        verdict = "SUCCESS"
    elif coord_success > single_success + 5:
        verdict = "PARTIAL"
    else:
        verdict = "FAIL"
    print(f"VERDICT: {verdict}", flush=True)
    print(f"\n  Communication:", flush=True)
    print(f"    A mass (0-40):  {mass_A_acc:.1f}%", flush=True)
    print(f"    B mass (40-80): {mass_B_acc:.1f}%", flush=True)
    print(f"    Fused mass:     {mass_fused_acc:.1f}%", flush=True)
    print(f"    Agreement:      {agree_pct:.1f}%", flush=True)
    print(f"    A elast:        {elast_A_acc:.1f}%", flush=True)
    print(f"    B elast:        {elast_B_acc:.1f}%", flush=True)
    print(f"\n  Planning:", flush=True)
    print(f"    Coordinated:    {coord_success:.1f}%", flush=True)
    print(f"    Single-agent:   {single_success:.1f}%", flush=True)
    print(f"    Advantage:      {coord_advantage:+.1f}pp (target: >+10pp)", flush=True)
    print(f"    Oracle coord:   {oracle_coord_success:.1f}%", flush=True)
    print(f"    Oracle single:  {oracle_single_success:.1f}%", flush=True)
    print(f"\n  Total time: {elapsed:.0f}s", flush=True)
    print(f"{'=' * 70}", flush=True)
