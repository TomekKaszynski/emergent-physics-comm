"""
Physics Simulator: 2D bouncing balls in a box.
Supports single ball, multi-ball with elastic collisions,
and rendering to pixel frames (front, top-down, side views).
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Ball:
    x: float
    y: float
    vx: float
    vy: float
    radius: float = 0.05
    mass: float = 1.0


@dataclass
class SimConfig:
    width: float = 2.0
    height: float = 2.0
    gravity: float = 9.81
    friction: float = 0.0
    restitution: float = 1.0
    dt: float = 0.02


class PhysicsSimulator:
    """Simple 2D physics with gravity, wall bounces, and ball-ball collisions."""

    def __init__(self, config: SimConfig = None):
        self.config = config or SimConfig()

    def step(self, balls: List[Ball]) -> List[Ball]:
        dt = self.config.dt
        new_balls = []
        for b in balls:
            vx = b.vx * (1 - self.config.friction)
            vy = b.vy * (1 - self.config.friction) - self.config.gravity * dt
            x = b.x + vx * dt
            y = b.y + vy * dt
            r = b.radius
            if y - r < 0:
                y = r + (r - y)
                vy = abs(vy) * self.config.restitution
            if y + r > self.config.height:
                y = self.config.height - r - (y + r - self.config.height)
                vy = -abs(vy) * self.config.restitution
            if x - r < 0:
                x = r + (r - x)
                vx = abs(vx) * self.config.restitution
            if x + r > self.config.width:
                x = self.config.width - r - (x + r - self.config.width)
                vx = -abs(vx) * self.config.restitution
            new_balls.append(Ball(x, y, vx, vy, b.radius, b.mass))
        if len(new_balls) > 1:
            new_balls = self._resolve_collisions(new_balls)
        return new_balls

    def _resolve_collisions(self, balls: List[Ball]) -> List[Ball]:
        for i in range(len(balls)):
            for j in range(i + 1, len(balls)):
                bi, bj = balls[i], balls[j]
                dx = bj.x - bi.x
                dy = bj.y - bi.y
                dist = np.sqrt(dx**2 + dy**2)
                min_dist = bi.radius + bj.radius
                if dist < min_dist and dist > 1e-10:
                    nx, ny = dx / dist, dy / dist
                    dvx = bi.vx - bj.vx
                    dvy = bi.vy - bj.vy
                    dvn = dvx * nx + dvy * ny
                    if dvn > 0:
                        m1, m2 = bi.mass, bj.mass
                        impulse = (2 * dvn) / (m1 + m2)
                        bi.vx -= impulse * m2 * nx
                        bi.vy -= impulse * m2 * ny
                        bj.vx += impulse * m1 * nx
                        bj.vy += impulse * m1 * ny
                        overlap = min_dist - dist
                        bi.x -= overlap * nx * 0.5
                        bi.y -= overlap * ny * 0.5
                        bj.x += overlap * nx * 0.5
                        bj.y += overlap * ny * 0.5
        return balls

    def simulate(self, balls: List[Ball], n_steps: int) -> np.ndarray:
        n_balls = len(balls)
        trajectory = np.zeros((n_steps + 1, n_balls * 4))
        for i, b in enumerate(balls):
            trajectory[0, i*4:(i+1)*4] = [b.x, b.y, b.vx, b.vy]
        current = [Ball(b.x, b.y, b.vx, b.vy, b.radius, b.mass) for b in balls]
        for t in range(n_steps):
            current = self.step(current)
            for i, b in enumerate(current):
                trajectory[t+1, i*4:(i+1)*4] = [b.x, b.y, b.vx, b.vy]
        return trajectory

    def _draw_balls(self, frame, balls, resolution, colors, px_fn, py_fn):
        for idx, b in enumerate(balls):
            color = colors[idx % len(colors)]
            px = int(px_fn(b) * resolution)
            py = int(py_fn(b) * resolution)
            pr = max(2, int(b.radius / self.config.width * resolution))
            for dy in range(-pr, pr + 1):
                for dx in range(-pr, pr + 1):
                    if dx**2 + dy**2 <= pr**2:
                        ix, iy = px + dx, py + dy
                        if 0 <= ix < resolution and 0 <= iy < resolution:
                            frame[iy, ix] = color
        frame[0, :] = [0, 0, 0]; frame[-1, :] = [0, 0, 0]
        frame[:, 0] = [0, 0, 0]; frame[:, -1] = [0, 0, 0]
        return frame

    def render_frame(self, balls, resolution=64):
        colors = [[.2,.4,.9],[.9,.3,.2],[.2,.8,.3],[.9,.7,.1],[.7,.2,.8]]
        frame = np.ones((resolution, resolution, 3))
        return self._draw_balls(frame, balls, resolution, colors,
            lambda b: b.x / self.config.width,
            lambda b: 1 - b.y / self.config.height)

    def render_topdown(self, balls, resolution=64):
        colors = [[.2,.4,.9],[.9,.3,.2],[.2,.8,.3],[.9,.7,.1],[.7,.2,.8]]
        frame = np.ones((resolution, resolution, 3))
        n = max(len(balls), 1)
        return self._draw_balls(frame, balls, resolution, colors,
            lambda b: b.x / self.config.width,
            lambda b: (balls.index(b) + 0.5) / n if b in balls else 0.5)

    def render_side(self, balls, resolution=64):
        colors = [[.2,.4,.9],[.9,.3,.2],[.2,.8,.3],[.9,.7,.1],[.7,.2,.8]]
        frame = np.ones((resolution, resolution, 3))
        n = max(len(balls), 1)
        return self._draw_balls(frame, balls, resolution, colors,
            lambda b: (balls.index(b) + 0.5) / n if b in balls else 0.5,
            lambda b: 1 - b.y / self.config.height)

    def render_trajectory(self, balls, n_steps, resolution=64):
        frames = np.zeros((n_steps + 1, resolution, resolution, 3))
        current = [Ball(b.x, b.y, b.vx, b.vy, b.radius, b.mass) for b in balls]
        frames[0] = self.render_frame(current, resolution)
        for t in range(n_steps):
            current = self.step(current)
            frames[t + 1] = self.render_frame(current, resolution)
        return frames


def generate_random_balls(n_balls, config=None, seed=None):
    if seed is not None:
        np.random.seed(seed)
    cfg = config or SimConfig()
    balls = []
    for _ in range(n_balls):
        r = 0.05
        balls.append(Ball(
            x=np.random.uniform(r + 0.1, cfg.width - r - 0.1),
            y=np.random.uniform(r + 0.1, cfg.height - r - 0.1),
            vx=np.random.uniform(-3, 3),
            vy=np.random.uniform(-3, 3),
            radius=r, mass=1.0))
    return balls


def generate_dataset(n_trajectories, n_steps, n_balls=1, config=None, seed=42):
    np.random.seed(seed)
    cfg = config or SimConfig()
    sim = PhysicsSimulator(cfg)
    all_x, all_y = [], []
    for _ in range(n_trajectories):
        balls = generate_random_balls(n_balls, cfg)
        traj = sim.simulate(balls, n_steps)
        all_x.append(traj[:-1])
        all_y.append(traj[1:])
    return np.concatenate(all_x).astype(np.float32), np.concatenate(all_y).astype(np.float32)


def get_occluded_state(full_state, n_balls, agent='A'):
    """
    Occlude balls based on x-position. Agent A sees left half (x < 1.0),
    Agent B sees right half (x >= 1.0). Unseen balls get zeroed out.
    full_state: [x1,y1,vx1,vy1, x2,y2,vx2,vy2, ...] for n_balls
    Returns: same-shape array with zeros for invisible balls.
    """
    state = np.array(full_state, dtype=np.float32).copy()
    for i in range(n_balls):
        x = state[i * 4]
        if agent == 'A' and x >= 1.0:
            state[i*4:(i+1)*4] = 0.0
        elif agent == 'B' and x < 1.0:
            state[i*4:(i+1)*4] = 0.0
    return state


def generate_occlusion_dataset(n_trajectories, n_steps, n_balls=3, seed=42):
    """
    Generate trajectories with occlusion for multi-agent communication.
    Returns: full_states, agent_a_states, agent_b_states, next_full_states, handoff_flags
    handoff_flags[i] is True if any ball crosses x=1.0 midline at step i.
    """
    np.random.seed(seed)
    cfg = SimConfig()
    sim = PhysicsSimulator(cfg)

    full_cur, occluded_a, occluded_b, full_nxt, handoffs = [], [], [], [], []

    for _ in range(n_trajectories):
        balls = generate_random_balls(n_balls, cfg)
        current = [Ball(b.x, b.y, b.vx, b.vy, b.radius, b.mass) for b in balls]
        for _ in range(n_steps):
            state = []
            for b in current:
                state.extend([b.x, b.y, b.vx, b.vy])
            state = np.array(state, dtype=np.float32)

            nxt_balls = sim.step(current)
            nxt_state = []
            for b in nxt_balls:
                nxt_state.extend([b.x, b.y, b.vx, b.vy])
            nxt_state = np.array(nxt_state, dtype=np.float32)

            # Detect handoff: any ball crosses x=1.0
            handoff = False
            for i in range(n_balls):
                x_cur = state[i * 4]
                x_nxt = nxt_state[i * 4]
                if (x_cur < 1.0 and x_nxt >= 1.0) or (x_cur >= 1.0 and x_nxt < 1.0):
                    handoff = True
                    break

            full_cur.append(state)
            occluded_a.append(get_occluded_state(state, n_balls, 'A'))
            occluded_b.append(get_occluded_state(state, n_balls, 'B'))
            full_nxt.append(nxt_state)
            handoffs.append(handoff)

            current = nxt_balls

    return (np.array(full_cur), np.array(occluded_a), np.array(occluded_b),
            np.array(full_nxt), np.array(handoffs))


# ── Spring-Mass Simulator (Phase 9) ─────────────────────────────

class SpringSimulator:
    """2D spring-mass chain in a box with damping and gravity."""
    def __init__(self, config, k=5.0, rest_length=0.4, damping=0.1):
        self.config = config
        self.k = k
        self.rest_length = rest_length
        self.damping = damping
        self.dt = config.dt

    def step(self, balls):
        """Step with spring forces between consecutive balls (chain)."""
        n = len(balls)
        new_balls = []
        for i in range(n):
            ax, ay = 0.0, -self.config.gravity
            # Spring forces from neighbors
            for j in [i-1, i+1]:
                if 0 <= j < n:
                    dx = balls[j].x - balls[i].x
                    dy = balls[j].y - balls[i].y
                    dist = max(np.sqrt(dx*dx + dy*dy), 1e-6)
                    force = self.k * (dist - self.rest_length)
                    ax += force * dx / dist / balls[i].mass
                    ay += force * dy / dist / balls[i].mass
            # Damping
            ax -= self.damping * balls[i].vx / balls[i].mass
            ay -= self.damping * balls[i].vy / balls[i].mass
            # Integrate
            vx = balls[i].vx + ax * self.dt
            vy = balls[i].vy + ay * self.dt
            x = balls[i].x + vx * self.dt
            y = balls[i].y + vy * self.dt
            # Wall bounces
            r = balls[i].radius
            if x - r < 0: x, vx = r, abs(vx) * 0.8
            if x + r > self.config.width: x, vx = self.config.width - r, -abs(vx) * 0.8
            if y - r < 0: y, vy = r, abs(vy) * 0.8
            if y + r > self.config.height: y, vy = self.config.height - r, -abs(vy) * 0.8
            new_balls.append(Ball(x, y, vx, vy, balls[i].radius, balls[i].mass))
        return new_balls


def generate_spring_dataset(n_trajectories, n_steps, n_balls=3, seed=42):
    """Generate spring-mass occlusion dataset (same interface as generate_occlusion_dataset)."""
    np.random.seed(seed)
    cfg = SimConfig()
    spring_sim = SpringSimulator(cfg)
    full_cur, occluded_a, occluded_b, full_nxt, handoffs = [], [], [], [], []

    for _ in range(n_trajectories):
        # Random initial positions spread across box, variable masses
        balls = []
        for i in range(n_balls):
            x = 0.3 + i * (cfg.width - 0.6) / max(n_balls - 1, 1) + np.random.uniform(-0.2, 0.2)
            y = np.random.uniform(0.5, cfg.height - 0.5)
            vx = np.random.uniform(-0.5, 0.5)
            vy = np.random.uniform(-0.3, 0.3)
            mass = np.random.uniform(0.5, 2.0)
            balls.append(Ball(x, y, vx, vy, radius=0.05, mass=mass))

        current = [Ball(b.x, b.y, b.vx, b.vy, b.radius, b.mass) for b in balls]
        for _ in range(n_steps):
            state = []
            for b in current:
                state.extend([b.x, b.y, b.vx, b.vy])
            state = np.array(state, dtype=np.float32)
            nxt_balls = spring_sim.step(current)
            nxt_state = []
            for b in nxt_balls:
                nxt_state.extend([b.x, b.y, b.vx, b.vy])
            nxt_state = np.array(nxt_state, dtype=np.float32)
            # Handoff detection
            handoff = False
            for i in range(n_balls):
                x_c, x_n = state[i*4], nxt_state[i*4]
                if (x_c < 1.0 and x_n >= 1.0) or (x_c >= 1.0 and x_n < 1.0):
                    handoff = True; break
            full_cur.append(state)
            occluded_a.append(get_occluded_state(state, n_balls, 'A'))
            occluded_b.append(get_occluded_state(state, n_balls, 'B'))
            full_nxt.append(nxt_state)
            handoffs.append(handoff)
            current = nxt_balls

    return (np.array(full_cur), np.array(occluded_a), np.array(occluded_b),
            np.array(full_nxt), np.array(handoffs))


# ── Half-frame rendering (Phase 11) ─────────────────────────────

def render_half_frame(sim, balls, side, resolution=64):
    """Render only balls visible from one side (A=left, B=right)."""
    midline = sim.config.width / 2.0  # 1.0
    visible = []
    for b in balls:
        if side == 'A' and b.x < midline:
            visible.append(b)
        elif side == 'B' and b.x >= midline:
            visible.append(b)
        else:
            visible.append(Ball(999, 999, 0, 0, b.radius, b.mass))  # offscreen
    return sim.render_frame(visible, resolution)


# ── N-agent strip occlusion (Phase 13) ──────────────────────────

def get_strip_occluded_state(state, n_balls, agent_id, n_agents, box_width=2.0):
    """Zero out balls not in this agent's strip."""
    strip_w = box_width / n_agents
    lo = agent_id * strip_w
    hi = lo + strip_w
    occ = np.zeros_like(state)
    for b in range(n_balls):
        x = state[b * 4]
        if lo <= x < hi:
            occ[b*4:b*4+4] = state[b*4:b*4+4]
    return occ


def generate_multiagent_dataset(n_trajectories, n_steps, n_balls, n_agents, seed=42):
    """Generate dataset for N-agent strip topology."""
    np.random.seed(seed)
    cfg = SimConfig()
    sim = PhysicsSimulator(cfg)
    full_cur, agent_views, full_nxt, handoffs = [], [], [], []

    for _ in range(n_trajectories):
        balls = generate_random_balls(n_balls, cfg)
        current = [Ball(b.x, b.y, b.vx, b.vy, b.radius, b.mass) for b in balls]
        for _ in range(n_steps):
            state = np.array([v for b in current for v in [b.x, b.y, b.vx, b.vy]],
                             dtype=np.float32)
            nxt_balls = sim.step(current)
            nxt_state = np.array([v for b in nxt_balls for v in [b.x, b.y, b.vx, b.vy]],
                                 dtype=np.float32)
            views = [get_strip_occluded_state(state, n_balls, a, n_agents)
                     for a in range(n_agents)]
            # Handoff: ball crosses any strip boundary
            strip_w = cfg.width / n_agents
            ho = any((int(state[i*4] / strip_w) != int(nxt_state[i*4] / strip_w))
                     for i in range(n_balls)
                     if nxt_state[i*4] < cfg.width)
            full_cur.append(state)
            agent_views.append(views)
            full_nxt.append(nxt_state)
            handoffs.append(ho)
            current = nxt_balls

    agent_arrays = [np.array([v[a] for v in agent_views], dtype=np.float32)
                    for a in range(n_agents)]
    return (np.array(full_cur), agent_arrays,
            np.array(full_nxt), np.array(handoffs))


# ── Phase 14: Controllable simulator ────────────────────────────

class ControllableSimulator(PhysicsSimulator):
    """Physics sim where agents can apply forces to balls in their half."""
    def step_with_actions(self, balls, action_a, action_b, max_force=2.0, dt=0.02):
        """Apply agent forces then simulate one step.
        action_a/b: array of [fx, fy] per ball (len = n_balls*2).
        Only applies force to balls in the agent's half (x<1 for A, x>=1 for B).
        """
        midline = self.config.width / 2
        for i, b in enumerate(balls):
            if i*2+1 >= len(action_a):
                continue
            if b.x < midline:
                fx = np.clip(action_a[i*2], -max_force, max_force)
                fy = np.clip(action_a[i*2+1], -max_force, max_force)
            else:
                fx = np.clip(action_b[i*2], -max_force, max_force)
                fy = np.clip(action_b[i*2+1], -max_force, max_force)
            b.vx += fx * dt / b.mass
            b.vy += fy * dt / b.mass
        return self.step(balls)


def generate_planning_episodes(n_episodes, n_balls=3, seed=42):
    """Generate (initial_state, goal_state) pairs for planning training."""
    np.random.seed(seed)
    cfg = SimConfig()
    episodes = []
    for _ in range(n_episodes):
        balls_init = generate_random_balls(n_balls, cfg)
        init_state = np.array([v for b in balls_init
                               for v in [b.x, b.y, b.vx, b.vy]], dtype=np.float32)
        # Random goal: random positions, zero velocity
        goal = np.zeros(n_balls * 4, dtype=np.float32)
        for b in range(n_balls):
            goal[b*4] = np.random.uniform(0.15, 1.85)
            goal[b*4+1] = np.random.uniform(0.15, 1.85)
        episodes.append((init_state, goal))
    return episodes


# ── Phase 17: Complementary sensing ─────────────────────────────

def get_complementary_obs(state, n_balls=3):
    """Split full state into position-only (A) and velocity-only (B)."""
    sd = n_balls * 4
    occ_a = np.zeros(sd, dtype=np.float32)
    occ_b = np.zeros(sd, dtype=np.float32)
    for i in range(n_balls):
        occ_a[i*4]   = state[i*4]      # x
        occ_a[i*4+1] = state[i*4+1]    # y
        occ_b[i*4+2] = state[i*4+2]    # vx
        occ_b[i*4+3] = state[i*4+3]    # vy
    return occ_a, occ_b


def generate_complementary_dataset(n_trajectories=2000, n_steps=50, n_balls=3, seed=42):
    """Generate dataset with position/velocity split observations."""
    np.random.seed(seed)
    cfg = SimConfig()
    sim = PhysicsSimulator(cfg)
    inputs_a, inputs_b, targets = [], [], []
    for _ in range(n_trajectories):
        balls = generate_random_balls(n_balls, cfg)
        for step in range(n_steps):
            state = np.array([v for b in balls for v in [b.x,b.y,b.vx,b.vy]], dtype=np.float32)
            oa, ob = get_complementary_obs(state, n_balls)
            balls = sim.step(balls)
            nxt = np.array([v for b in balls for v in [b.x,b.y,b.vx,b.vy]], dtype=np.float32)
            inputs_a.append(oa); inputs_b.append(ob); targets.append(nxt)
    return (np.array(inputs_a), np.array(inputs_b), np.array(targets))


# ── Phase 21: Causal Physics ───────────────────────────────────

import math

class CausalPhysicsSimulator:
    """Physics with explicit causal structure: two regimes.
    
    CAUSAL: Ball 0 ↔ Ball 1 connected by spring. Ball 2 independent.
    CORRELATIONAL: Ball 0 & Ball 1 driven by hidden oscillator. Ball 2 independent.
    
    Both regimes produce similar statistical correlations between Ball 0 and Ball 1,
    but the causal structure is completely different.
    """
    def __init__(self, width=2.0, height=2.0, gravity=9.81, regime='causal'):
        self.width = width
        self.height = height
        self.gravity = gravity
        self.regime = regime
        # Spring params (causal)
        self.spring_k = 2.0
        self.spring_rest = 0.8
        # Oscillator params (correlational)
        self.osc_freq = 2.0
        self.osc_amplitude = 1.5
        self.osc_phase = 0.0

    def step(self, balls, t, dt=0.02):
        """One physics step with regime-dependent coupling."""
        if self.regime == 'causal':
            # Spring between ball 0 and ball 1
            dx = balls[1].x - balls[0].x
            dy = balls[1].y - balls[0].y
            dist = math.sqrt(dx*dx + dy*dy) + 1e-6
            force = self.spring_k * (dist - self.spring_rest)
            fx = force * dx / dist
            fy = force * dy / dist
            balls[0].vx += fx / balls[0].mass * dt
            balls[0].vy += fy / balls[0].mass * dt
            balls[1].vx -= fx / balls[1].mass * dt
            balls[1].vy -= fy / balls[1].mass * dt
        elif self.regime == 'correlational':
            # Hidden oscillator drives both 0 and 1
            self.osc_phase += self.osc_freq * dt * 2 * math.pi
            sfx = self.osc_amplitude * math.sin(self.osc_phase)
            sfy = self.osc_amplitude * math.cos(self.osc_phase * 0.7)
            balls[0].vx += sfx / balls[0].mass * dt
            balls[0].vy += sfy / balls[0].mass * dt
            balls[1].vx += sfx / balls[1].mass * dt
            balls[1].vy += sfy / balls[1].mass * dt

        # Standard physics for all balls
        for b in balls:
            b.vy -= self.gravity * dt
            b.x += b.vx * dt
            b.y += b.vy * dt
            r = b.radius
            if b.y - r < 0:
                b.y = r; b.vy = abs(b.vy) * 0.95
            if b.y + r > self.height:
                b.y = self.height - r; b.vy = -abs(b.vy) * 0.95
            if b.x - r < 0:
                b.x = r; b.vx = abs(b.vx) * 0.95
            if b.x + r > self.width:
                b.x = self.width - r; b.vx = -abs(b.vx) * 0.95

        # Ball-ball collisions
        for i in range(len(balls)):
            for j in range(i+1, len(balls)):
                bi, bj = balls[i], balls[j]
                dx = bj.x - bi.x; dy = bj.y - bi.y
                dist = math.sqrt(dx*dx + dy*dy)
                md = bi.radius + bj.radius
                if dist < md and dist > 1e-10:
                    nx, ny = dx/dist, dy/dist
                    dvn = (bi.vx-bj.vx)*nx + (bi.vy-bj.vy)*ny
                    if dvn > 0:
                        imp = 2*dvn / (bi.mass + bj.mass)
                        bi.vx -= imp*bj.mass*nx; bi.vy -= imp*bj.mass*ny
                        bj.vx += imp*bi.mass*nx; bj.vy += imp*bi.mass*ny
                    overlap = md - dist
                    bi.x -= overlap*nx*0.5; bi.y -= overlap*ny*0.5
                    bj.x += overlap*nx*0.5; bj.y += overlap*ny*0.5
        return balls

    def random_balls(self, n=3):
        """Ball 0: left, Ball 1: right, Ball 2: anywhere."""
        return [
            Ball(x=np.random.uniform(0.2, 0.8), y=np.random.uniform(0.5, 1.5),
                 vx=np.random.uniform(-1, 1), vy=np.random.uniform(-1, 1)),
            Ball(x=np.random.uniform(1.2, 1.8), y=np.random.uniform(0.5, 1.5),
                 vx=np.random.uniform(-1, 1), vy=np.random.uniform(-1, 1)),
            Ball(x=np.random.uniform(0.3, 1.7), y=np.random.uniform(0.5, 1.5),
                 vx=np.random.uniform(-1, 1), vy=np.random.uniform(-1, 1)),
        ]

    def get_state(self, balls):
        return np.array([v for b in balls for v in [b.x, b.y, b.vx, b.vy]],
                        dtype=np.float32)


def generate_causal_dataset(regime, n_traj=1500, n_steps=50, n_balls=3, seed=42):
    """Generate dataset from specified causal regime."""
    np.random.seed(seed)
    sim = CausalPhysicsSimulator(regime=regime)
    inputs_a, inputs_b, targets = [], [], []
    for _ in range(n_traj):
        sim.osc_phase = 0.0  # reset oscillator per trajectory
        balls = sim.random_balls(n_balls)
        for step in range(n_steps):
            state = sim.get_state(balls)
            oa = get_occluded_state(state, n_balls, 'A')
            ob = get_occluded_state(state, n_balls, 'B')
            balls = sim.step(balls, t=step*0.02)
            nxt = sim.get_state(balls)
            inputs_a.append(oa); inputs_b.append(ob); targets.append(nxt)
    return (np.array(inputs_a, dtype=np.float32),
            np.array(inputs_b, dtype=np.float32),
            np.array(targets, dtype=np.float32))


# ── Phase 22: Isolated Causal Physics ──────────────────────────

class IsolatedCausalSimulator:
    """Clean causal test: 4 balls in 4×4 box, well-separated, NO collisions.
    3 regimes: causal (spring 0↔1), correlational (oscillator), independent."""
    def __init__(self, width=4.0, height=4.0, gravity=9.81, regime='causal'):
        self.width = width
        self.height = height
        self.gravity = gravity
        self.regime = regime
        self.spring_k = 3.0
        self.spring_rest = 1.5
        self.osc_freq = 1.5
        self.osc_amplitude = 1.0
        self.osc_phase = 0.0
        self.dt = 0.02

    def random_balls(self):
        return [
            Ball(x=0.7+np.random.uniform(-0.2,0.2), y=2.5+np.random.uniform(-0.3,0.3),
                 vx=np.random.uniform(-0.5,0.5), vy=np.random.uniform(-0.5,0.5)),
            Ball(x=3.3+np.random.uniform(-0.2,0.2), y=2.5+np.random.uniform(-0.3,0.3),
                 vx=np.random.uniform(-0.5,0.5), vy=np.random.uniform(-0.5,0.5)),
            Ball(x=0.7+np.random.uniform(-0.2,0.2), y=0.7+np.random.uniform(-0.2,0.2),
                 vx=np.random.uniform(-0.5,0.5), vy=np.random.uniform(-0.5,0.5)),
            Ball(x=3.3+np.random.uniform(-0.2,0.2), y=0.7+np.random.uniform(-0.2,0.2),
                 vx=np.random.uniform(-0.5,0.5), vy=np.random.uniform(-0.5,0.5)),
        ]

    def step(self, balls, t):
        dt = self.dt
        if self.regime == 'causal':
            dx = balls[1].x - balls[0].x
            dy = balls[1].y - balls[0].y
            dist = math.sqrt(dx*dx + dy*dy) + 1e-6
            force = self.spring_k * (dist - self.spring_rest)
            fx, fy = force*dx/dist, force*dy/dist
            balls[0].vx += fx/balls[0].mass*dt; balls[0].vy += fy/balls[0].mass*dt
            balls[1].vx -= fx/balls[1].mass*dt; balls[1].vy -= fy/balls[1].mass*dt
        elif self.regime == 'correlational':
            self.osc_phase += self.osc_freq * dt * 2 * math.pi
            fx = self.osc_amplitude * math.sin(self.osc_phase)
            fy = self.osc_amplitude * math.cos(self.osc_phase * 0.7)
            balls[0].vx += fx*dt; balls[0].vy += fy*dt
            balls[1].vx += fx*dt; balls[1].vy += fy*dt
        # 'independent': no coupling
        for b in balls:
            b.vy -= self.gravity * dt
            b.x += b.vx * dt; b.y += b.vy * dt
            if b.y < 0.15: b.y = 0.15; b.vy = abs(b.vy)*0.8
            if b.y > self.height-0.15: b.y = self.height-0.15; b.vy = -abs(b.vy)*0.8
            if b.x < 0.15: b.x = 0.15; b.vx = abs(b.vx)*0.8
            if b.x > self.width-0.15: b.x = self.width-0.15; b.vx = -abs(b.vx)*0.8

    def get_state(self, balls):
        return np.array([v for b in balls for v in [b.x, b.y, b.vx, b.vy]], dtype=np.float32)


def generate_isolated_causal_dataset(regime, n_traj=1500, n_steps=50, seed=42):
    np.random.seed(seed)
    sim = IsolatedCausalSimulator(regime=regime)
    inputs_a, inputs_b, targets = [], [], []
    midline = 2.0
    for _ in range(n_traj):
        sim.osc_phase = 0.0
        balls = sim.random_balls()
        for step in range(n_steps):
            state = sim.get_state(balls)
            oa, ob = state.copy(), state.copy()
            for i in range(4):
                b = i*4
                if state[b] >= midline:
                    oa[b:b+4] = 0.0
                else:
                    ob[b:b+4] = 0.0
            inputs_a.append(oa); inputs_b.append(ob)
            sim.step(balls, t=step*0.02)
            targets.append(sim.get_state(balls))
    return (np.array(inputs_a, dtype=np.float32),
            np.array(inputs_b, dtype=np.float32),
            np.array(targets, dtype=np.float32))


# ── Phase 23: 3D Physics ──────────────────────────────────────

class Physics3D:
    """3D sphere physics in a 3×3×3 box. State per object: [x,y,z,vx,vy,vz]=6d."""
    def __init__(self, box_size=3.0, gravity=-9.8):
        self.box_size = box_size
        self.gravity = gravity
        self.dt = 0.02
        self.spring_k = 0.0
        self.spring_rest = 1.0

    def random_spheres(self, n=4):
        spheres = []
        for _ in range(n):
            spheres.append({
                'x': np.random.uniform(0.3,2.7), 'y': np.random.uniform(0.3,2.7),
                'z': np.random.uniform(0.5,2.5),
                'vx': np.random.uniform(-1.5,1.5), 'vy': np.random.uniform(-1.5,1.5),
                'vz': np.random.uniform(-1.5,1.5),
                'mass': 1.0, 'radius': 0.15,
            })
        return spheres

    def step(self, spheres):
        dt = self.dt
        if self.spring_k > 0 and len(spheres) >= 2:
            s0, s1 = spheres[0], spheres[1]
            dx = s1['x']-s0['x']; dy = s1['y']-s0['y']; dz = s1['z']-s0['z']
            dist = math.sqrt(dx**2+dy**2+dz**2) + 1e-6
            f = self.spring_k * (dist - self.spring_rest)
            fx, fy, fz = f*dx/dist, f*dy/dist, f*dz/dist
            for dim, fd in [('vx',fx),('vy',fy),('vz',fz)]:
                s0[dim] += fd/s0['mass']*dt; s1[dim] -= fd/s1['mass']*dt
        for s in spheres:
            s['vz'] += self.gravity * dt
            s['x'] += s['vx']*dt; s['y'] += s['vy']*dt; s['z'] += s['vz']*dt
            r = s['radius']
            for dim in ['x','y','z']:
                vd = 'v'+dim
                if s[dim] < r: s[dim] = r; s[vd] = abs(s[vd])*0.8
                if s[dim] > self.box_size-r: s[dim] = self.box_size-r; s[vd] = -abs(s[vd])*0.8
        for i in range(len(spheres)):
            for j in range(i+1, len(spheres)):
                self._collide(spheres[i], spheres[j])

    def _collide(self, a, b):
        dx = b['x']-a['x']; dy = b['y']-a['y']; dz = b['z']-a['z']
        dist = math.sqrt(dx**2+dy**2+dz**2)
        md = a['radius']+b['radius']
        if dist < md and dist > 1e-6:
            nx,ny,nz = dx/dist, dy/dist, dz/dist
            dvn = (a['vx']-b['vx'])*nx+(a['vy']-b['vy'])*ny+(a['vz']-b['vz'])*nz
            if dvn > 0:
                j = -(1+0.8)*dvn/(1/a['mass']+1/b['mass'])
                for dim, nd in [('vx',nx),('vy',ny),('vz',nz)]:
                    a[dim] += j*nd/a['mass']; b[dim] -= j*nd/b['mass']
                ov = md-dist
                for dim, nd in [('x',nx),('y',ny),('z',nz)]:
                    a[dim] -= ov*nd*0.5; b[dim] += ov*nd*0.5

    def get_state(self, spheres):
        return np.array([v for s in spheres for v in [s['x'],s['y'],s['z'],s['vx'],s['vy'],s['vz']]],
                        dtype=np.float32)

    def get_occluded_3d(self, state, n_objects=4, midline=1.5):
        oa, ob = state.copy(), state.copy()
        for i in range(n_objects):
            b = i*6
            if state[b] >= midline:
                oa[b:b+6] = 0.0
            else:
                ob[b:b+6] = 0.0
        return oa, ob


def generate_3d_dataset(n_traj=1500, n_steps=50, spring_k=0.0, seed=42):
    np.random.seed(seed)
    sim = Physics3D(); sim.spring_k = spring_k
    inputs_a, inputs_b, targets = [], [], []
    for _ in range(n_traj):
        spheres = sim.random_spheres(4)
        for step in range(n_steps):
            state = sim.get_state(spheres)
            oa, ob = sim.get_occluded_3d(state)
            inputs_a.append(oa); inputs_b.append(ob)
            sim.step(spheres)
            targets.append(sim.get_state(spheres))
    return (np.array(inputs_a, dtype=np.float32),
            np.array(inputs_b, dtype=np.float32),
            np.array(targets, dtype=np.float32))


# ── Phase 25: Visual Physics Renderer ──────────────────────────

class VisualPhysics3D:
    """3D physics with perspective rendering to 64×64 RGB images.
    Two cameras: A (left) and B (right), each seeing a partial view."""

    def __init__(self, box_size=2.0, gravity=-9.8, n_objects=5, img_size=64):
        self.box_size = box_size
        self.gravity = gravity
        self.n_objects = n_objects
        self.img_size = img_size
        self.dt = 0.02
        # Camera A: far left, viewing left half
        self.cam_a_pos = np.array([-0.8, box_size/2, box_size*0.6])
        self.cam_a_target = np.array([box_size*0.3, box_size/2, box_size*0.25])
        # Camera B: far right, viewing right half
        self.cam_b_pos = np.array([box_size+0.8, box_size/2, box_size*0.6])
        self.cam_b_target = np.array([box_size*0.7, box_size/2, box_size*0.25])
        self.objects = []

    def reset(self):
        self.objects = []
        for _ in range(self.n_objects):
            self.objects.append({
                'x': np.random.uniform(0.3, self.box_size-0.3),
                'y': np.random.uniform(0.3, self.box_size-0.3),
                'z': np.random.uniform(0.3, self.box_size*0.7),
                'vx': np.random.uniform(-1.5, 1.5),
                'vy': np.random.uniform(-1.5, 1.5),
                'vz': np.random.uniform(-1.0, 1.0),
                'mass': float(np.random.choice([0.5, 1.0, 2.0, 3.0])),
                'radius': np.random.uniform(0.08, 0.18),
                'color': [np.random.uniform(0.3, 1.0), np.random.uniform(0.2, 1.0),
                          np.random.uniform(0.2, 1.0)],
                'shape': int(np.random.randint(0, 3)),
            })
        for _ in range(10):
            self.step()
        return self.render()

    def step(self, action=None):
        if action is not None:
            idx = int(action[0])
            if 0 <= idx < len(self.objects):
                o = self.objects[idx]
                o['vx'] += action[1]*self.dt/o['mass']
                o['vy'] += action[2]*self.dt/o['mass']
                o['vz'] += action[3]*self.dt/o['mass']
        for o in self.objects:
            o['vz'] += self.gravity * self.dt
            o['x'] += o['vx']*self.dt; o['y'] += o['vy']*self.dt; o['z'] += o['vz']*self.dt
            r = o['radius']
            if o['z'] < r: o['z'] = r; o['vz'] = abs(o['vz'])*0.7; o['vx'] *= 0.95; o['vy'] *= 0.95
            for d in ['x','y']:
                if o[d] < r: o[d] = r; o['v'+d] = abs(o['v'+d])*0.8
                if o[d] > self.box_size-r: o[d] = self.box_size-r; o['v'+d] = -abs(o['v'+d])*0.8
            if o['z'] > self.box_size-r: o['z'] = self.box_size-r; o['vz'] = -abs(o['vz'])*0.8
        for i in range(len(self.objects)):
            for j in range(i+1, len(self.objects)):
                a, b = self.objects[i], self.objects[j]
                dx = b['x']-a['x']; dy = b['y']-a['y']; dz = b['z']-a['z']
                dist = math.sqrt(dx**2+dy**2+dz**2)
                md = a['radius']+b['radius']
                if dist < md and dist > 1e-6:
                    nx,ny,nz = dx/dist, dy/dist, dz/dist
                    dvn = (a['vx']-b['vx'])*nx+(a['vy']-b['vy'])*ny+(a['vz']-b['vz'])*nz
                    if dvn > 0:
                        ji = -1.8*dvn/(1/a['mass']+1/b['mass'])
                        a['vx']+=ji*nx/a['mass']; a['vy']+=ji*ny/a['mass']; a['vz']+=ji*nz/a['mass']
                        b['vx']-=ji*nx/b['mass']; b['vy']-=ji*ny/b['mass']; b['vz']-=ji*nz/b['mass']
                        ov = md-dist
                        a['x']-=ov*nx*0.5; a['y']-=ov*ny*0.5; a['z']-=ov*nz*0.5
                        b['x']+=ov*nx*0.5; b['y']+=ov*ny*0.5; b['z']+=ov*nz*0.5
        return self.render()

    def render(self):
        return self._render_view(self.cam_a_pos, self.cam_a_target), \
               self._render_view(self.cam_b_pos, self.cam_b_target)

    def _render_view(self, cam_pos, cam_target):
        S = self.img_size
        img = np.ones((S, S, 3), dtype=np.float32) * 0.85
        for row in range(S):
            img[row, :, :] *= (0.7 + 0.3 * row / S)
        forward = cam_target - cam_pos
        forward = forward / (np.linalg.norm(forward) + 1e-8)
        right = np.cross(forward, np.array([0., 0., 1.]))
        right = right / (np.linalg.norm(right) + 1e-8)
        up = np.cross(right, forward)
        depths = []
        for o in self.objects:
            pos = np.array([o['x'], o['y'], o['z']])
            depths.append((np.dot(pos - cam_pos, forward), o))
        depths.sort(key=lambda x: -x[0])
        focal = S / (2 * np.tan(np.radians(22.5)))  # 45° FOV
        for depth, o in depths:
            if depth < 0.1: continue
            pos = np.array([o['x'], o['y'], o['z']])
            to = pos - cam_pos
            sx = int(np.dot(to, right) / depth * focal + S / 2)
            sy = int(-np.dot(to, up) / depth * focal + S / 2)
            sr = max(2, int(o['radius'] / depth * focal))
            shade = max(0.3, min(1.0, 2.0 / (depth + 0.5)))
            col = np.array(o['color']) * shade
            if o['shape'] == 0:
                for dy in range(-sr, sr+1):
                    for dx in range(-sr, sr+1):
                        if dx*dx+dy*dy <= sr*sr:
                            px, py = sx+dx, sy+dy
                            if 0 <= px < S and 0 <= py < S:
                                light = 1.0 + 0.3*(-dx-dy)/(sr+1)
                                img[py, px] = np.clip(col * light, 0, 1)
            elif o['shape'] == 1:
                for dy in range(-sr, sr+1):
                    for dx in range(-sr, sr+1):
                        px, py = sx+dx, sy+dy
                        if 0 <= px < S and 0 <= py < S:
                            img[py, px] = np.clip(col * 0.9, 0, 1)
            else:
                for dy in range(-sr*2, sr*2+1):
                    for dx in range(-sr, sr+1):
                        px, py = sx+dx, sy+dy
                        if 0 <= px < S and 0 <= py < S:
                            light = 1.0 + 0.2*(-dx)/(sr+1)
                            img[py, px] = np.clip(col * light, 0, 1)
        return img

    def get_state(self):
        s = []
        for o in self.objects:
            s.extend([o['x'], o['y'], o['z'], o['vx'], o['vy'], o['vz'], o['mass'], o['radius']])
        return s


def collect_visual_dataset(n_episodes=300, steps_per_episode=40, n_objects=5, img_size=64):
    """Collect (img_a, img_b, action, next) dataset for visual JEPA training."""
    env = VisualPhysics3D(n_objects=n_objects, img_size=img_size)
    all_ia, all_ib, all_na, all_nb, all_act, all_st = [], [], [], [], [], []
    for ep in range(n_episodes):
        img_a, img_b = env.reset()
        for step in range(steps_per_episode):
            if np.random.random() < 0.6:
                oi = np.random.randint(0, n_objects)
                fx, fy, fz = np.random.uniform(-3,3), np.random.uniform(-3,3), np.random.uniform(-2,2)
                action = (oi, fx, fy, fz)
                avec = [oi/n_objects, fx/3.0, fy/3.0, fz/2.0]
            else:
                action = None; avec = [0,0,0,0]
            all_ia.append(img_a.transpose(2,0,1))
            all_ib.append(img_b.transpose(2,0,1))
            all_act.append(avec); all_st.append(env.get_state())
            na, nb = env.step(action)
            all_na.append(na.transpose(2,0,1)); all_nb.append(nb.transpose(2,0,1))
            img_a, img_b = na, nb
        if (ep+1) % 100 == 0:
            print(f"  Collected {ep+1}/{n_episodes} episodes ({(ep+1)*steps_per_episode} frames)")
    import torch
    return {
        'img_a': torch.tensor(np.array(all_ia), dtype=torch.float32),
        'img_b': torch.tensor(np.array(all_ib), dtype=torch.float32),
        'next_img_a': torch.tensor(np.array(all_na), dtype=torch.float32),
        'next_img_b': torch.tensor(np.array(all_nb), dtype=torch.float32),
        'action': torch.tensor(np.array(all_act), dtype=torch.float32),
        'state': torch.tensor(np.array(all_st), dtype=torch.float32),
    }


# ── Phase 25d: Split View Physics ────────────────────────────────

class SplitViewPhysics3D:
    """3D physics with HARD information separation via split rendering.

    Same physics as VisualPhysics3D (5 objects in 2×2×2 box).
    But rendering is SPLIT:
    - Agent A's image: only objects with x < midline
    - Agent B's image: only objects with x >= midline
    - Target image: ALL objects (overhead camera)

    This creates hard information separation — Agent A literally
    cannot see right-side objects, and vice versa.
    """

    def __init__(self, box_size=2.0, gravity=-9.8, n_objects=5, img_size=64):
        self.box_size = box_size
        self.gravity = gravity
        self.n_objects = n_objects
        self.img_size = img_size
        self.dt = 0.02
        self.midline = box_size / 2  # x = 1.0

        # Agent A camera: centered on left half
        self.cam_a_pos = np.array([box_size * 0.25, box_size / 2, box_size * 1.2])
        self.cam_a_target = np.array([box_size * 0.25, box_size / 2, 0.0])
        # Agent B camera: centered on right half
        self.cam_b_pos = np.array([box_size * 0.75, box_size / 2, box_size * 1.2])
        self.cam_b_target = np.array([box_size * 0.75, box_size / 2, 0.0])
        # Overhead camera: full scene
        self.cam_over_pos = np.array([box_size / 2, box_size / 2, box_size * 1.5])
        self.cam_over_target = np.array([box_size / 2, box_size / 2, 0.0])
        self.objects = []

    def reset(self):
        self.objects = []
        for _ in range(self.n_objects):
            self.objects.append({
                'x': np.random.uniform(0.15, self.box_size - 0.15),
                'y': np.random.uniform(0.15, self.box_size - 0.15),
                'z': np.random.uniform(0.3, self.box_size * 0.6),
                'vx': np.random.uniform(-1.5, 1.5),
                'vy': np.random.uniform(-1.5, 1.5),
                'vz': np.random.uniform(-0.5, 0.5),
                'mass': float(np.random.choice([0.5, 1.0, 2.0, 3.0])),
                'radius': np.random.uniform(0.08, 0.15),
                'color': [np.random.uniform(0.3, 1.0), np.random.uniform(0.2, 1.0),
                          np.random.uniform(0.2, 1.0)],
                'shape': int(np.random.randint(0, 3)),
            })
        for _ in range(10):
            self._physics_step()
        return self.render()

    def _physics_step(self):
        dt = self.dt
        for o in self.objects:
            o['vz'] += self.gravity * dt
            o['x'] += o['vx'] * dt
            o['y'] += o['vy'] * dt
            o['z'] += o['vz'] * dt
            r = o['radius']
            if o['z'] < r:
                o['z'] = r; o['vz'] = abs(o['vz']) * 0.7
                o['vx'] *= 0.95; o['vy'] *= 0.95
            for d in ['x', 'y']:
                if o[d] < r: o[d] = r; o['v' + d] = abs(o['v' + d]) * 0.8
                if o[d] > self.box_size - r:
                    o[d] = self.box_size - r; o['v' + d] = -abs(o['v' + d]) * 0.8
            if o['z'] > self.box_size - r:
                o['z'] = self.box_size - r; o['vz'] = -abs(o['vz']) * 0.8
        for i in range(len(self.objects)):
            for j in range(i + 1, len(self.objects)):
                a, b = self.objects[i], self.objects[j]
                dx = b['x'] - a['x']; dy = b['y'] - a['y']; dz = b['z'] - a['z']
                dist = math.sqrt(dx**2 + dy**2 + dz**2)
                md = a['radius'] + b['radius']
                if dist < md and dist > 1e-6:
                    nx, ny, nz = dx / dist, dy / dist, dz / dist
                    dvn = ((a['vx'] - b['vx']) * nx + (a['vy'] - b['vy']) * ny +
                           (a['vz'] - b['vz']) * nz)
                    if dvn > 0:
                        ji = -1.8 * dvn / (1 / a['mass'] + 1 / b['mass'])
                        a['vx'] += ji * nx / a['mass']
                        a['vy'] += ji * ny / a['mass']
                        a['vz'] += ji * nz / a['mass']
                        b['vx'] -= ji * nx / b['mass']
                        b['vy'] -= ji * ny / b['mass']
                        b['vz'] -= ji * nz / b['mass']
                        ov = md - dist
                        a['x'] -= ov * nx * 0.5; a['y'] -= ov * ny * 0.5; a['z'] -= ov * nz * 0.5
                        b['x'] += ov * nx * 0.5; b['y'] += ov * ny * 0.5; b['z'] += ov * nz * 0.5

    def step(self, action=None):
        if action is not None:
            idx = int(action[0])
            if 0 <= idx < len(self.objects):
                o = self.objects[idx]
                o['vx'] += action[1] * self.dt / o['mass']
                o['vy'] += action[2] * self.dt / o['mass']
                o['vz'] += action[3] * self.dt / o['mass']
        self._physics_step()
        return self.render()

    def render(self):
        left_objs = [o for o in self.objects if o['x'] < self.midline]
        right_objs = [o for o in self.objects if o['x'] >= self.midline]
        img_a = self._render_view(self.cam_a_pos, self.cam_a_target, left_objs)
        img_b = self._render_view(self.cam_b_pos, self.cam_b_target, right_objs)
        img_t = self._render_view(self.cam_over_pos, self.cam_over_target, self.objects)
        return img_a, img_b, img_t

    def _render_view(self, cam_pos, cam_target, objs_to_render):
        S = self.img_size
        img = np.ones((S, S, 3), dtype=np.float32) * 0.85
        for row in range(S):
            img[row, :, :] *= (0.7 + 0.3 * row / S)
        forward = cam_target - cam_pos
        forward = forward / (np.linalg.norm(forward) + 1e-8)
        right = np.cross(forward, np.array([0., 0., 1.]))
        right = right / (np.linalg.norm(right) + 1e-8)
        up = np.cross(right, forward)
        focal = S / (2 * np.tan(np.radians(25)))  # 50° FOV
        depths = []
        for o in objs_to_render:
            pos = np.array([o['x'], o['y'], o['z']])
            depths.append((np.dot(pos - cam_pos, forward), o))
        depths.sort(key=lambda x: -x[0])
        for depth, o in depths:
            if depth < 0.1:
                continue
            pos = np.array([o['x'], o['y'], o['z']])
            to = pos - cam_pos
            sx = int(np.dot(to, right) / depth * focal + S / 2)
            sy = int(-np.dot(to, up) / depth * focal + S / 2)
            sr = max(2, int(o['radius'] / depth * focal))
            shade = max(0.3, min(1.0, 2.0 / (depth + 0.5)))
            col = np.array(o['color']) * shade
            for dy in range(-sr, sr + 1):
                for dx in range(-sr, sr + 1):
                    if dx * dx + dy * dy <= sr * sr:
                        px, py = sx + dx, sy + dy
                        if 0 <= px < S and 0 <= py < S:
                            light = 1.0 + 0.3 * (-dx - dy) / (sr + 1)
                            img[py, px] = np.clip(col * light, 0, 1)
        return img

    def get_state(self):
        s = []
        for o in self.objects:
            s.extend([o['x'], o['y'], o['z'], o['vx'], o['vy'], o['vz'],
                      o['mass'], o['radius']])
        return s

    def get_side_labels(self):
        return [0 if o['x'] < self.midline else 1 for o in self.objects]


class RichVisualPhysics3D:
    """Physics sim with visually rich scenes for object discovery.

    Compared to SplitViewPhysics3D:
    - 8 objects (was 5)
    - 3 shape types: sphere, cube, pyramid
    - Checkerboard floor
    - Dual colors per object (primary + highlight)
    """

    def __init__(self, box_size=2.0, gravity=-9.8, n_objects=8, img_size=64):
        self.box_size = box_size
        self.gravity = gravity
        self.n_objects = n_objects
        self.img_size = img_size
        self.dt = 0.02
        self.midline = box_size / 2

        self.cam_overhead_pos = np.array([box_size/2, box_size/2, box_size*1.5])
        self.cam_overhead_target = np.array([box_size/2, box_size/2, 0.0])
        self.cam_a_pos = np.array([box_size*0.25, box_size/2, box_size*1.2])
        self.cam_a_target = np.array([box_size*0.25, box_size/2, 0.0])
        self.cam_b_pos = np.array([box_size*0.75, box_size/2, box_size*1.2])
        self.cam_b_target = np.array([box_size*0.75, box_size/2, 0.0])
        self.objects = []

    def reset(self):
        import random
        self.objects = []
        shapes = [0, 0, 0, 1, 1, 1, 2, 2]
        random.shuffle(shapes)

        palettes = [
            ([0.9, 0.2, 0.2], [1.0, 0.5, 0.3]),
            ([0.2, 0.7, 0.2], [0.4, 1.0, 0.4]),
            ([0.2, 0.3, 0.9], [0.5, 0.5, 1.0]),
            ([0.9, 0.9, 0.1], [1.0, 1.0, 0.6]),
            ([0.8, 0.2, 0.8], [1.0, 0.5, 1.0]),
            ([0.1, 0.8, 0.8], [0.4, 1.0, 1.0]),
            ([0.9, 0.5, 0.1], [1.0, 0.7, 0.3]),
            ([0.6, 0.6, 0.6], [0.9, 0.9, 0.9]),
        ]
        random.shuffle(palettes)

        for i in range(self.n_objects):
            self.objects.append({
                'x': random.uniform(0.2, self.box_size - 0.2),
                'y': random.uniform(0.2, self.box_size - 0.2),
                'z': random.uniform(0.3, self.box_size * 0.5),
                'vx': random.uniform(-1.5, 1.5),
                'vy': random.uniform(-1.5, 1.5),
                'vz': random.uniform(-0.5, 0.5),
                'mass': random.choice([0.5, 1.0, 2.0, 3.0]),
                'radius': random.uniform(0.08, 0.14),
                'color': palettes[i][0],
                'color2': palettes[i][1],
                'shape': shapes[i],
            })

        for _ in range(10):
            self._physics_step()
        return self.render()

    def _physics_step(self):
        dt = self.dt
        for obj in self.objects:
            obj['vz'] += self.gravity * dt
            obj['x'] += obj['vx'] * dt
            obj['y'] += obj['vy'] * dt
            obj['z'] += obj['vz'] * dt
            r = obj['radius']
            if obj['z'] < r:
                obj['z'] = r; obj['vz'] = abs(obj['vz']) * 0.7
                obj['vx'] *= 0.95; obj['vy'] *= 0.95
            for dim in ['x', 'y']:
                if obj[dim] < r:
                    obj[dim] = r; obj['v'+dim] = abs(obj['v'+dim]) * 0.8
                if obj[dim] > self.box_size - r:
                    obj[dim] = self.box_size - r
                    obj['v'+dim] = -abs(obj['v'+dim]) * 0.8
            if obj['z'] > self.box_size - r:
                obj['z'] = self.box_size - r
                obj['vz'] = -abs(obj['vz']) * 0.8

        for i in range(len(self.objects)):
            for j in range(i+1, len(self.objects)):
                a, b = self.objects[i], self.objects[j]
                dx = b['x']-a['x']; dy = b['y']-a['y']; dz = b['z']-a['z']
                dist = (dx**2 + dy**2 + dz**2) ** 0.5
                min_d = a['radius'] + b['radius']
                if dist < min_d and dist > 1e-6:
                    nx, ny, nz = dx/dist, dy/dist, dz/dist
                    dvn = ((a['vx']-b['vx'])*nx + (a['vy']-b['vy'])*ny
                           + (a['vz']-b['vz'])*nz)
                    if dvn > 0:
                        j_imp = -1.8 * dvn / (1/a['mass'] + 1/b['mass'])
                        a['vx'] += j_imp*nx/a['mass']
                        a['vy'] += j_imp*ny/a['mass']
                        a['vz'] += j_imp*nz/a['mass']
                        b['vx'] -= j_imp*nx/b['mass']
                        b['vy'] -= j_imp*ny/b['mass']
                        b['vz'] -= j_imp*nz/b['mass']
                        overlap = min_d - dist
                        a['x'] -= overlap*nx*0.5; a['y'] -= overlap*ny*0.5
                        a['z'] -= overlap*nz*0.5
                        b['x'] += overlap*nx*0.5; b['y'] += overlap*ny*0.5
                        b['z'] += overlap*nz*0.5

    def step(self, action=None):
        if action is not None:
            idx, fx, fy, fz = action
            idx = int(idx)
            if 0 <= idx < len(self.objects):
                obj = self.objects[idx]
                obj['vx'] += fx * self.dt / obj['mass']
                obj['vy'] += fy * self.dt / obj['mass']
                obj['vz'] += fz * self.dt / obj['mass']
        self._physics_step()
        return self.render()

    def render(self):
        left = [o for o in self.objects if o['x'] < self.midline]
        right = [o for o in self.objects if o['x'] >= self.midline]
        img_a = self._render_view(self.cam_a_pos, self.cam_a_target, left)
        img_b = self._render_view(self.cam_b_pos, self.cam_b_target, right)
        img_t = self._render_view(
            self.cam_overhead_pos, self.cam_overhead_target, self.objects)
        return img_a, img_b, img_t

    def _render_view(self, cam_pos, cam_target, objects_to_render):
        S = self.img_size
        img = np.ones((S, S, 3), dtype=np.float32) * 0.85

        # Checkerboard floor
        cs = S // 8
        for row in range(S):
            for col in range(S):
                t = row / S
                checker = ((row // cs) + (col // cs)) % 2
                if checker == 0:
                    img[row, col] = np.array([0.75, 0.75, 0.80]) * (0.6 + 0.4*t)
                else:
                    img[row, col] = np.array([0.85, 0.82, 0.78]) * (0.6 + 0.4*t)

        forward = cam_target - cam_pos
        forward = forward / (np.linalg.norm(forward) + 1e-8)
        right = np.cross(forward, np.array([0, 0, 1]))
        right = right / (np.linalg.norm(right) + 1e-8)
        up = np.cross(right, forward)

        fov = 50
        focal = S / (2 * np.tan(np.radians(fov / 2)))

        obj_depths = []
        for obj in objects_to_render:
            pos = np.array([obj['x'], obj['y'], obj['z']])
            depth = np.dot(pos - cam_pos, forward)
            obj_depths.append((depth, obj))
        obj_depths.sort(key=lambda x: -x[0])

        for depth, obj in obj_depths:
            if depth < 0.1:
                continue
            pos = np.array([obj['x'], obj['y'], obj['z']])
            to_obj = pos - cam_pos
            sx = np.dot(to_obj, right) / depth * focal + S / 2
            sy = -np.dot(to_obj, up) / depth * focal + S / 2
            sr = max(2, int(obj['radius'] / depth * focal))

            cx, cy = int(sx), int(sy)
            shade = max(0.3, min(1.0, 2.0 / (depth + 0.5)))
            c1 = np.array(obj['color']) * shade
            c2 = np.array(obj['color2']) * shade
            shape = obj['shape']

            for dy in range(-sr-1, sr+2):
                for dx in range(-sr-1, sr+2):
                    px, py = cx + dx, cy + dy
                    if not (0 <= px < S and 0 <= py < S):
                        continue

                    if shape == 0:  # Sphere
                        if dx*dx + dy*dy <= sr*sr:
                            d = (dx*dx + dy*dy)**0.5 / (sr + 1e-6)
                            t = min(1.0, d)
                            color = c1*(1-t*0.5) + c2*(t*0.5)
                            light = 1.0 + 0.3*(-dx-dy)/(sr+1)
                            img[py, px] = np.clip(color*light, 0, 1)

                    elif shape == 1:  # Cube
                        if abs(dx) <= sr and abs(dy) <= sr:
                            color = c1 if dy < 0 else c2
                            if abs(dx) >= sr-1 or abs(dy) >= sr-1:
                                color = color * 0.7
                            light = 1.0 + 0.2*(-dx)/(sr+1)
                            img[py, px] = np.clip(color*light, 0, 1)

                    elif shape == 2:  # Pyramid
                        prog = (dy + sr) / (2*sr + 1e-6)
                        w = int(sr * prog)
                        if abs(dx) <= w and abs(dy) <= sr:
                            color = c1*(1-prog) + c2*prog
                            light = 1.0 + 0.2*(-dx)/(sr+1)
                            img[py, px] = np.clip(color*light, 0, 1)

        return img

    def get_state(self):
        state = []
        for obj in self.objects:
            state.extend([obj['x'], obj['y'], obj['z'],
                          obj['vx'], obj['vy'], obj['vz'],
                          obj['mass'], obj['radius'],
                          float(obj['shape'])])
        return state

    def get_side_labels(self):
        return ['left' if o['x'] < self.midline else 'right'
                for o in self.objects]


def collect_rich_dataset(n_episodes=300, steps_per_episode=40,
                         n_objects=8, img_size=64):
    """Collect dataset with rich visual scenes and split views."""
    import random
    env = RichVisualPhysics3D(n_objects=n_objects, img_size=img_size)

    all_a, all_b, all_t = [], [], []
    all_na, all_nb, all_nt = [], [], []
    all_actions, all_states = [], []

    for ep in range(n_episodes):
        img_a, img_b, img_t = env.reset()

        for step in range(steps_per_episode):
            if random.random() < 0.6:
                obj_idx = random.randint(0, n_objects - 1)
                fx = random.uniform(-3, 3)
                fy = random.uniform(-3, 3)
                fz = random.uniform(-2, 2)
                action = (obj_idx, fx, fy, fz)
                action_vec = [obj_idx/n_objects, fx/3.0, fy/3.0, fz/2.0]
            else:
                action = None
                action_vec = [0, 0, 0, 0]

            all_a.append(img_a.transpose(2, 0, 1))
            all_b.append(img_b.transpose(2, 0, 1))
            all_t.append(img_t.transpose(2, 0, 1))
            all_actions.append(action_vec)
            all_states.append(env.get_state())

            next_a, next_b, next_t = env.step(action)
            all_na.append(next_a.transpose(2, 0, 1))
            all_nb.append(next_b.transpose(2, 0, 1))
            all_nt.append(next_t.transpose(2, 0, 1))
            img_a, img_b, img_t = next_a, next_b, next_t

        if (ep + 1) % 50 == 0:
            print(f"  Collected {ep+1}/{n_episodes} episodes")

    import torch
    return {
        'img_a': torch.tensor(np.array(all_a), dtype=torch.float32),
        'img_b': torch.tensor(np.array(all_b), dtype=torch.float32),
        'img_target': torch.tensor(np.array(all_t), dtype=torch.float32),
        'next_img_a': torch.tensor(np.array(all_na), dtype=torch.float32),
        'next_img_b': torch.tensor(np.array(all_nb), dtype=torch.float32),
        'next_img_target': torch.tensor(np.array(all_nt), dtype=torch.float32),
        'action': torch.tensor(np.array(all_actions), dtype=torch.float32),
        'state': torch.tensor(np.array(all_states), dtype=torch.float32),
    }


class SimplifiedRichPhysics3D(RichVisualPhysics3D):
    """RichVisualPhysics3D with simple gradient background instead of checkerboard.

    The checkerboard floor is a high-frequency texture that drowns out object
    signals in pixel MSE reconstruction. Replace with a simple vertical gradient.
    All physics, objects, cameras, and shape rendering are inherited unchanged.
    """

    def _render_view(self, cam_pos, cam_target, objects_to_render):
        S = self.img_size
        img = np.ones((S, S, 3), dtype=np.float32)

        # Simple gradient background (low frequency, easy to separate from objects)
        base_color = np.array([0.85, 0.85, 0.88])
        for row in range(S):
            t = row / S
            img[row, :] = base_color * (0.7 + 0.3 * t)

        # Camera projection (identical to RichVisualPhysics3D)
        forward = cam_target - cam_pos
        forward = forward / (np.linalg.norm(forward) + 1e-8)
        right = np.cross(forward, np.array([0, 0, 1]))
        right = right / (np.linalg.norm(right) + 1e-8)
        up = np.cross(right, forward)

        fov = 50
        focal = S / (2 * np.tan(np.radians(fov / 2)))

        obj_depths = []
        for obj in objects_to_render:
            pos = np.array([obj['x'], obj['y'], obj['z']])
            depth = np.dot(pos - cam_pos, forward)
            obj_depths.append((depth, obj))
        obj_depths.sort(key=lambda x: -x[0])

        for depth, obj in obj_depths:
            if depth < 0.1:
                continue
            pos = np.array([obj['x'], obj['y'], obj['z']])
            to_obj = pos - cam_pos
            sx = np.dot(to_obj, right) / depth * focal + S / 2
            sy = -np.dot(to_obj, up) / depth * focal + S / 2
            sr = max(2, int(obj['radius'] / depth * focal))

            cx, cy = int(sx), int(sy)
            shade = max(0.3, min(1.0, 2.0 / (depth + 0.5)))
            c1 = np.array(obj['color']) * shade
            c2 = np.array(obj['color2']) * shade
            shape = obj['shape']

            for dy in range(-sr-1, sr+2):
                for dx in range(-sr-1, sr+2):
                    px, py = cx + dx, cy + dy
                    if not (0 <= px < S and 0 <= py < S):
                        continue

                    if shape == 0:  # Sphere
                        if dx*dx + dy*dy <= sr*sr:
                            d = (dx*dx + dy*dy)**0.5 / (sr + 1e-6)
                            t = min(1.0, d)
                            color = c1*(1-t*0.5) + c2*(t*0.5)
                            light = 1.0 + 0.3*(-dx-dy)/(sr+1)
                            img[py, px] = np.clip(color*light, 0, 1)

                    elif shape == 1:  # Cube
                        if abs(dx) <= sr and abs(dy) <= sr:
                            color = c1 if dy < 0 else c2
                            if abs(dx) >= sr-1 or abs(dy) >= sr-1:
                                color = color * 0.7
                            light = 1.0 + 0.2*(-dx)/(sr+1)
                            img[py, px] = np.clip(color*light, 0, 1)

                    elif shape == 2:  # Pyramid
                        prog = (dy + sr) / (2*sr + 1e-6)
                        w = int(sr * prog)
                        if abs(dx) <= w and abs(dy) <= sr:
                            color = c1*(1-prog) + c2*prog
                            light = 1.0 + 0.2*(-dx)/(sr+1)
                            img[py, px] = np.clip(color*light, 0, 1)

        return img


def collect_simplified_rich_dataset(n_episodes=300, steps_per_episode=40,
                                    n_objects=8, img_size=64):
    """Collect dataset with gradient-background rich scenes."""
    import random
    env = SimplifiedRichPhysics3D(n_objects=n_objects, img_size=img_size)

    all_a, all_b, all_t = [], [], []
    all_na, all_nb, all_nt = [], [], []
    all_actions, all_states = [], []

    for ep in range(n_episodes):
        img_a, img_b, img_t = env.reset()

        for step in range(steps_per_episode):
            if random.random() < 0.6:
                obj_idx = random.randint(0, n_objects - 1)
                fx = random.uniform(-3, 3)
                fy = random.uniform(-3, 3)
                fz = random.uniform(-2, 2)
                action = (obj_idx, fx, fy, fz)
                action_vec = [obj_idx/n_objects, fx/3.0, fy/3.0, fz/2.0]
            else:
                action = None
                action_vec = [0, 0, 0, 0]

            all_a.append(img_a.transpose(2, 0, 1))
            all_b.append(img_b.transpose(2, 0, 1))
            all_t.append(img_t.transpose(2, 0, 1))
            all_actions.append(action_vec)
            all_states.append(env.get_state())

            next_a, next_b, next_t = env.step(action)
            all_na.append(next_a.transpose(2, 0, 1))
            all_nb.append(next_b.transpose(2, 0, 1))
            all_nt.append(next_t.transpose(2, 0, 1))
            img_a, img_b, img_t = next_a, next_b, next_t

        if (ep + 1) % 50 == 0:
            print(f"  Collected {ep+1}/{n_episodes} episodes")

    import torch
    return {
        'img_a': torch.tensor(np.array(all_a), dtype=torch.float32),
        'img_b': torch.tensor(np.array(all_b), dtype=torch.float32),
        'img_target': torch.tensor(np.array(all_t), dtype=torch.float32),
        'next_img_a': torch.tensor(np.array(all_na), dtype=torch.float32),
        'next_img_b': torch.tensor(np.array(all_nb), dtype=torch.float32),
        'next_img_target': torch.tensor(np.array(all_nt), dtype=torch.float32),
        'action': torch.tensor(np.array(all_actions), dtype=torch.float32),
        'state': torch.tensor(np.array(all_states), dtype=torch.float32),
    }


def generate_clevr_images(n_images=5000, img_size=64, max_objects=4):
    """Generate simple CLEVR-like images: colored circles on gray background.

    For diagnostic testing of Slot Attention — the simplest possible scene.
    Returns ground-truth masks so we can compute ARI.
    """
    import torch
    import random

    palette = [
        [1.0, 0.0, 0.0],    # red
        [0.0, 1.0, 0.0],    # green
        [0.0, 0.0, 1.0],    # blue
        [1.0, 1.0, 0.0],    # yellow
        [0.0, 1.0, 1.0],    # cyan
        [1.0, 0.0, 1.0],    # magenta
        [1.0, 0.5, 0.0],    # orange
        [0.5, 0.0, 1.0],    # purple
    ]

    S = img_size
    all_images = []
    all_masks = []      # [N, max_objects+1, H, W] — channel 0 = background
    all_n_obj = []

    for _ in range(n_images):
        img = np.ones((S, S, 3), dtype=np.float32) * 0.5  # gray background
        mask = np.zeros((max_objects + 1, S, S), dtype=np.float32)
        mask[0] = 1.0  # background mask starts as all-1

        n_obj = random.randint(2, max_objects)
        circles = []

        for obj_i in range(n_obj):
            # Reject sampling for non-overlapping placement
            for _attempt in range(100):
                r = random.randint(5, 12)
                cx = random.randint(r + 1, S - r - 2)
                cy = random.randint(r + 1, S - r - 2)
                # Check no overlap with existing circles
                ok = True
                for (ox, oy, orr) in circles:
                    dist = ((cx - ox)**2 + (cy - oy)**2)**0.5
                    if dist < r + orr + 2:  # 2px gap
                        ok = False
                        break
                if ok:
                    break
            else:
                continue  # skip this object if can't place

            circles.append((cx, cy, r))
            color = np.array(palette[obj_i % len(palette)])

            # Draw filled circle
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    if dx*dx + dy*dy <= r*r:
                        px, py = cx + dx, cy + dy
                        if 0 <= px < S and 0 <= py < S:
                            img[py, px] = color
                            mask[0, py, px] = 0.0          # not background
                            mask[obj_i + 1, py, px] = 1.0  # this object

        all_images.append(img.transpose(2, 0, 1))  # CHW
        all_masks.append(mask)
        all_n_obj.append(n_obj)

        if (_ + 1) % 1000 == 0:
            print(f"  Generated {_+1}/{n_images} CLEVR images")

    return {
        'images': torch.tensor(np.array(all_images), dtype=torch.float32),
        'masks_gt': torch.tensor(np.array(all_masks), dtype=torch.float32),
        'n_objects': torch.tensor(all_n_obj, dtype=torch.int64),
    }


def collect_split_view_dataset(n_episodes=300, steps_per_episode=40,
                               n_objects=5, img_size=64):
    """Collect dataset with SPLIT views for Phase 25d.

    Returns dict with keys: img_a, img_b, img_target, next_img_a, next_img_b,
    next_img_target, action, state, side_labels.
    """
    env = SplitViewPhysics3D(n_objects=n_objects, img_size=img_size)
    all_a, all_b, all_t = [], [], []
    all_na, all_nb, all_nt = [], [], []
    all_act, all_st, all_sides = [], [], []
    for ep in range(n_episodes):
        img_a, img_b, img_t = env.reset()
        for step in range(steps_per_episode):
            if np.random.random() < 0.6:
                oi = np.random.randint(0, n_objects)
                fx = np.random.uniform(-3, 3)
                fy = np.random.uniform(-3, 3)
                fz = np.random.uniform(-2, 2)
                action = (oi, fx, fy, fz)
                avec = [oi / n_objects, fx / 3.0, fy / 3.0, fz / 2.0]
            else:
                action = None; avec = [0, 0, 0, 0]
            all_a.append(img_a.transpose(2, 0, 1))
            all_b.append(img_b.transpose(2, 0, 1))
            all_t.append(img_t.transpose(2, 0, 1))
            all_act.append(avec)
            all_st.append(env.get_state())
            all_sides.append(env.get_side_labels())
            na, nb, nt = env.step(action)
            all_na.append(na.transpose(2, 0, 1))
            all_nb.append(nb.transpose(2, 0, 1))
            all_nt.append(nt.transpose(2, 0, 1))
            img_a, img_b, img_t = na, nb, nt
        if (ep + 1) % 100 == 0:
            print(f"  Collected {ep+1}/{n_episodes} episodes "
                  f"({(ep+1)*steps_per_episode} frames)")
    import torch
    return {
        'img_a': torch.tensor(np.array(all_a), dtype=torch.float32),
        'img_b': torch.tensor(np.array(all_b), dtype=torch.float32),
        'img_target': torch.tensor(np.array(all_t), dtype=torch.float32),
        'next_img_a': torch.tensor(np.array(all_na), dtype=torch.float32),
        'next_img_b': torch.tensor(np.array(all_nb), dtype=torch.float32),
        'next_img_target': torch.tensor(np.array(all_nt), dtype=torch.float32),
        'action': torch.tensor(np.array(all_act), dtype=torch.float32),
        'state': torch.tensor(np.array(all_st), dtype=torch.float32),
        'side_labels': torch.tensor(np.array(all_sides), dtype=torch.float32),
    }


# ── Phase 18-20: Property-Based Objects ─────────────────────────

class PropertyObject:
    """Object defined by continuous properties, not discrete types.
    State vector: [x, y, vx, vy, mass, elasticity, friction, flatness, rigidity, stuck_to] = 10d"""
    def __init__(self, mass=1.0, elasticity=0.5, friction=0.3, flatness=0.0,
                 rigidity=0.5, x=None, y=None, vx=None, vy=None):
        self.mass = mass
        self.elasticity = elasticity
        self.friction = friction
        self.flatness = flatness
        self.rigidity = rigidity
        self.x = x if x is not None else np.random.uniform(0.2, 1.8)
        self.y = y if y is not None else np.random.uniform(0.3, 1.7)
        self.vx = vx if vx is not None else np.random.uniform(-1, 1)
        self.vy = vy if vy is not None else np.random.uniform(-1, 1)
        self.stuck_to = -1  # index of bonded partner, or -1
        self.stuck_timer = 0

    @property
    def effective_radius(self):
        return 0.08 * (1 + self.flatness * 1.5)

    @property
    def effective_height(self):
        return 0.08 * (1 - self.flatness * 0.6)

    def state_vector(self):
        return np.array([self.x, self.y, self.vx, self.vy,
                         self.mass, self.elasticity, self.friction,
                         self.flatness, self.rigidity, float(self.stuck_to)],
                        dtype=np.float32)

    @staticmethod
    def from_preset(preset, **kwargs):
        presets = {
            'ball':     dict(mass=1.0, elasticity=0.8, friction=0.3, flatness=0.0, rigidity=0.5),
            'heavy':    dict(mass=3.0, elasticity=0.5, friction=0.4, flatness=0.1, rigidity=0.8),
            'light':    dict(mass=0.3, elasticity=0.95, friction=0.2, flatness=0.0, rigidity=0.4),
            'sticky':   dict(mass=1.0, elasticity=0.1, friction=0.8, flatness=0.0, rigidity=0.3),
            'platform': dict(mass=2.0, elasticity=0.3, friction=0.6, flatness=0.8, rigidity=0.7),
        }
        props = {**presets[preset], **kwargs}
        return PropertyObject(**props)


OBJ_DIM = 10  # state vector dimension per object


class RichPhysicsSimulator:
    """2D physics with property-dependent interactions."""
    def __init__(self, width=2.0, height=2.0, gravity=9.81, dt=0.02):
        self.width = width
        self.height = height
        self.gravity = gravity
        self.dt = dt

    def step(self, objects, actions=None, dt=None):
        """Physics step with property-dependent collisions, sticking, support."""
        dt = dt or self.dt
        n = len(objects)

        # Apply actions (forces)
        if actions is not None:
            for i, obj in enumerate(objects):
                if i * 2 + 1 < len(actions):
                    fx = np.clip(actions[i*2], -2.0, 2.0)
                    fy = np.clip(actions[i*2+1], -2.0, 2.0)
                    obj.vx += fx * dt / obj.mass
                    obj.vy += fy * dt / obj.mass

        # Gravity
        for obj in objects:
            obj.vy -= self.gravity * dt

        # Stuck bonds: bonded objects move together
        for i, obj in enumerate(objects):
            if obj.stuck_to >= 0 and obj.stuck_to < n:
                partner = objects[obj.stuck_to]
                # Average velocities (momentum-weighted)
                total_m = obj.mass + partner.mass
                avg_vx = (obj.vx * obj.mass + partner.vx * partner.mass) / total_m
                avg_vy = (obj.vy * obj.mass + partner.vy * partner.mass) / total_m
                obj.vx, obj.vy = avg_vx, avg_vy
                partner.vx, partner.vy = avg_vx, avg_vy
                obj.stuck_timer -= 1
                if obj.stuck_timer <= 0:
                    obj.stuck_to = -1
                    partner.stuck_to = -1

        # Update positions
        for obj in objects:
            obj.x += obj.vx * dt
            obj.y += obj.vy * dt

        # Wall collisions
        for obj in objects:
            r = obj.effective_radius
            if obj.x - r < 0:
                obj.x = r; obj.vx = abs(obj.vx) * obj.elasticity
            elif obj.x + r > self.width:
                obj.x = self.width - r; obj.vx = -abs(obj.vx) * obj.elasticity
            if obj.y - obj.effective_height < 0:
                obj.y = obj.effective_height; obj.vy = abs(obj.vy) * obj.elasticity
                obj.vx *= (1 - obj.friction * 0.1)  # floor friction
            elif obj.y + obj.effective_height > self.height:
                obj.y = self.height - obj.effective_height
                obj.vy = -abs(obj.vy) * obj.elasticity

        # Object-object collisions
        for i in range(n):
            for j in range(i+1, n):
                a, b = objects[i], objects[j]
                dx = b.x - a.x; dy = b.y - a.y
                dist = np.sqrt(dx*dx + dy*dy) + 1e-8
                min_dist = a.effective_radius + b.effective_radius
                if dist < min_dist:
                    # Separate objects
                    overlap = min_dist - dist
                    nx, ny = dx/dist, dy/dist
                    a.x -= nx * overlap * 0.5
                    a.y -= ny * overlap * 0.5
                    b.x += nx * overlap * 0.5
                    b.y += ny * overlap * 0.5

                    # Elasticity-dependent restitution
                    e = min(a.elasticity, b.elasticity)

                    # Check sticky condition: both low elasticity
                    if a.elasticity < 0.2 or b.elasticity < 0.2:
                        if a.stuck_to < 0 and b.stuck_to < 0:
                            a.stuck_to = j; b.stuck_to = i
                            a.stuck_timer = 50; b.stuck_timer = 50  # 1 second bond

                    # Momentum-based collision (mass-dependent)
                    rel_vx = a.vx - b.vx; rel_vy = a.vy - b.vy
                    rel_dot = rel_vx * nx + rel_vy * ny
                    if rel_dot > 0:
                        j_imp = -(1 + e) * rel_dot / (1/a.mass + 1/b.mass)
                        a.vx += j_imp * nx / a.mass
                        a.vy += j_imp * ny / a.mass
                        b.vx -= j_imp * nx / b.mass
                        b.vy -= j_imp * ny / b.mass

        # Support check: flat objects support objects above them
        for i in range(n):
            for j in range(n):
                if i == j: continue
                below, above = objects[i], objects[j]
                if (below.flatness > 0.5 and below.rigidity > 0.3 and
                    above.y > below.y and
                    abs(above.x - below.x) < below.effective_radius and
                    abs(above.y - below.y) < below.effective_height + above.effective_height + 0.05 and
                    above.vy < 0.3):
                    # Support: stop downward motion, match horizontal motion
                    above.vy = max(above.vy, 0)
                    above.y = max(above.y, below.y + below.effective_height + above.effective_height)
                    above.vx = above.vx * 0.95 + below.vx * 0.05  # slight coupling

        return objects

    def get_scene_state(self, objects):
        """Return [n_objects, OBJ_DIM] state matrix."""
        return np.array([obj.state_vector() for obj in objects], dtype=np.float32)

    def get_flat_state(self, objects):
        """Return flattened state [n_objects * OBJ_DIM]."""
        return self.get_scene_state(objects).flatten()

    def random_scene(self, n_objects=5, presets=None):
        """Generate random scene with mixed object types."""
        if presets is None:
            presets = ['ball', 'ball', 'heavy', 'light', 'platform']
        objects = []
        for i, p in enumerate(presets[:n_objects]):
            obj = PropertyObject.from_preset(p)
            objects.append(obj)
        return objects


class AffordanceSimulator(RichPhysicsSimulator):
    """Goal-conditioned affordance tasks using property objects."""

    def setup_task(self, task_name, n_objects=5):
        """Set up initial conditions and goal for a specific task.
        Returns (objects, goal_dict)."""
        if task_name == 'stack':
            objects = [
                PropertyObject.from_preset('platform', x=1.0, y=0.3, vx=0, vy=0),
                PropertyObject.from_preset('ball', x=0.5, y=1.5, vx=0, vy=0),
            ]
            # Pad to n_objects
            while len(objects) < n_objects:
                objects.append(PropertyObject.from_preset('ball',
                    x=np.random.uniform(0.3, 1.7), y=np.random.uniform(0.5, 1.5), vx=0, vy=0))
            return objects, {'type': 'stack', 'subject': 1, 'surface': 0}

        elif task_name == 'capture':
            objects = [
                PropertyObject.from_preset('sticky', x=0.3, y=1.0, vx=0, vy=0),
                PropertyObject.from_preset('ball', x=1.7, y=1.0, vx=0, vy=0),
            ]
            while len(objects) < n_objects:
                objects.append(PropertyObject.from_preset('light',
                    x=np.random.uniform(0.3, 1.7), y=np.random.uniform(0.5, 1.5), vx=0, vy=0))
            return objects, {'type': 'capture', 'obj_a': 0, 'obj_b': 1}

        elif task_name == 'shield':
            objects = [
                PropertyObject.from_preset('heavy', x=1.0, y=0.5, vx=0, vy=0),
                PropertyObject.from_preset('ball', x=0.3, y=0.5, vx=2.0, vy=0),
            ]
            while len(objects) < n_objects:
                objects.append(PropertyObject.from_preset('ball',
                    x=np.random.uniform(0.3, 1.7), y=np.random.uniform(0.5, 1.5), vx=0, vy=0))
            return objects, {'type': 'shield', 'shield_obj': 0, 'protected': 1}

        elif task_name == 'launch':
            objects = [
                PropertyObject.from_preset('heavy', x=0.3, y=1.0, vx=3.0, vy=0),
                PropertyObject.from_preset('light', x=1.0, y=1.0, vx=0, vy=0),
            ]
            while len(objects) < n_objects:
                objects.append(PropertyObject.from_preset('ball',
                    x=np.random.uniform(0.3, 1.7), y=np.random.uniform(0.5, 1.5), vx=0, vy=0))
            return objects, {'type': 'launch', 'target': 1, 'min_speed': 4.0}

        elif task_name == 'bridge':
            objects = [
                PropertyObject.from_preset('platform', x=0.6, y=0.3, vx=0, vy=0),
                PropertyObject.from_preset('platform', x=1.4, y=0.3, vx=0, vy=0),
                PropertyObject.from_preset('ball', x=0.3, y=0.5, vx=1.0, vy=0),
            ]
            while len(objects) < n_objects:
                objects.append(PropertyObject.from_preset('light',
                    x=np.random.uniform(0.3, 1.7), y=np.random.uniform(0.5, 1.5), vx=0, vy=0))
            return objects, {'type': 'bridge', 'ball': 2, 'target_x': 1.7}

        raise ValueError(f"Unknown task: {task_name}")

    def check_goal(self, objects, goal):
        """Return distance to goal (0 = achieved)."""
        if goal['type'] == 'stack':
            s, p = objects[goal['subject']], objects[goal['surface']]
            # Ball should be on top of platform, nearly stationary
            dx = abs(s.x - p.x)
            dy = s.y - p.y - p.effective_height - s.effective_height
            speed = np.sqrt(s.vx**2 + s.vy**2)
            return dx + abs(dy) + speed * 0.5

        elif goal['type'] == 'capture':
            a, b = objects[goal['obj_a']], objects[goal['obj_b']]
            if a.stuck_to == goal['obj_b']:
                return 0.0  # captured!
            dx = a.x - b.x; dy = a.y - b.y
            return np.sqrt(dx*dx + dy*dy)

        elif goal['type'] == 'shield':
            sh, pr = objects[goal['shield_obj']], objects[goal['protected']]
            # Shield should be between protected and right wall
            if sh.x > pr.x and sh.x < self.width - 0.2:
                return max(0, pr.x - sh.x + 0.5)  # good position
            return abs(sh.x - (pr.x + 0.5))

        elif goal['type'] == 'launch':
            t = objects[goal['target']]
            speed = np.sqrt(t.vx**2 + t.vy**2)
            return max(0, goal['min_speed'] - speed) + max(0, 1.8 - t.x) * 0.5

        elif goal['type'] == 'bridge':
            b = objects[goal['ball']]
            return max(0, goal['target_x'] - b.x)

        return float('inf')


def generate_rich_dataset(n_trajectories=1000, n_steps=30, n_objects=5, seed=42):
    """Generate dataset with property-diverse objects for world model training."""
    np.random.seed(seed)
    sim = RichPhysicsSimulator()
    all_states, all_next = [], []
    preset_combos = [
        ['ball', 'ball', 'heavy', 'light', 'platform'],
        ['ball', 'heavy', 'sticky', 'light', 'ball'],
        ['platform', 'ball', 'ball', 'heavy', 'sticky'],
        ['light', 'light', 'heavy', 'platform', 'ball'],
        ['sticky', 'ball', 'platform', 'light', 'heavy'],
    ]
    for traj_i in range(n_trajectories):
        presets = preset_combos[traj_i % len(preset_combos)]
        objects = sim.random_scene(n_objects, presets)
        for step in range(n_steps):
            state = sim.get_scene_state(objects)  # [N, OBJ_DIM]
            objects = sim.step(objects)
            nxt = sim.get_scene_state(objects)
            all_states.append(state)
            all_next.append(nxt)
    return np.array(all_states), np.array(all_next)


def get_rich_occluded(scene_state, agent='A', midline=1.0):
    """Occlude objects based on x-position in rich physics.
    scene_state: [n_objects, OBJ_DIM]. Zeros out objects not visible."""
    occ = scene_state.copy()
    for i in range(len(occ)):
        x = occ[i, 0]
        if agent == 'A' and x >= midline:
            occ[i, :4] = 0  # zero dynamics, keep properties visible
        elif agent == 'B' and x < midline:
            occ[i, :4] = 0
    return occ


if __name__ == "__main__":
    config = SimConfig()
    sim = PhysicsSimulator(config)
    ball = Ball(x=1.0, y=1.5, vx=0.5, vy=0.0)
    traj = sim.simulate([ball], 200)
    print(f"Trajectory shape: {traj.shape}")
    print(f"Y range: [{traj[:, 1].min():.3f}, {traj[:, 1].max():.3f}]")
    X, Y = generate_dataset(1000, 100, n_balls=1)
    print(f"Dataset: X={X.shape}, Y={Y.shape}, {len(X)} pairs")

