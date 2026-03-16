"""
Kubric Ramp Dataset Generator
==============================
Phase 54: Generate ball-on-ramp scenes for compositional communication.

Each scene: one ball spawned near top of a ~70° ramp. The ball slides
down (friction signal), then bounces when it hits flat ground (elasticity signal).

Property grid 5x5:
  restitution in {0.1, 0.3, 0.5, 0.7, 0.9}
  friction    in {0.1, 0.3, 0.5, 0.7, 0.9}
  12 scenes per combo = 300 total

Key physics: tan(70deg) ~ 2.747, μ_crit=(2/7)*tan(70°)=0.785
  friction < 0.785 -> ball slides at friction-dependent speed (f=0.1..0.7)
  friction > 0.785 -> ball rolls at fixed speed (f=0.9 only)

Output per scene: rgba_00000.png through rgba_00023.png (24 frames at 12fps = 2s)
Training code subsamples 8 from 24 (same pattern as elasticity dataset).

Usage (from ~/AI/kubric/):
  docker run --rm --interactive \
    --user $(id -u):$(id -g) \
    --volume "$(pwd):/kubric" \
    kubricdockerhub/kubruntu:latest \
    python3 generate_ramp_dataset.py

  # Resume:
  docker run --rm --interactive \
    --user $(id -u):$(id -g) \
    --volume "$(pwd):/kubric" \
    kubricdockerhub/kubruntu:latest \
    python3 generate_ramp_dataset.py --start_id 150
"""

import argparse
import json
import logging
import math
import os
import gc
import sys
import time
import numpy as np

import kubric as kb
from kubric.simulator.pybullet import PyBullet as KubricSimulator

logging.basicConfig(level="WARNING")

# Grid definition
RESTITUTION_LEVELS = [0.1, 0.3, 0.5, 0.7, 0.9]
FRICTION_LEVELS = [0.1, 0.3, 0.5, 0.7, 0.9]
SCENES_PER_CELL = 12
TOTAL_SCENES = len(RESTITUTION_LEVELS) * len(FRICTION_LEVELS) * SCENES_PER_CELL  # 300

RAMP_ANGLE_DEG = 70.0
RAMP_ANGLE_RAD = math.radians(RAMP_ANGLE_DEG)


def get_grid_cell(scene_id):
    """Map scene_id -> (restitution, friction, e_bin, f_bin)."""
    cell_idx = scene_id // SCENES_PER_CELL
    e_bin = cell_idx // len(FRICTION_LEVELS)
    f_bin = cell_idx % len(FRICTION_LEVELS)
    return (RESTITUTION_LEVELS[e_bin], FRICTION_LEVELS[f_bin], e_bin, f_bin)


def generate_scene(scene_id, output_dir, rng, render=False, resolution=256,
                   samples_per_pixel=32):
    """Generate one ball-on-ramp scene."""
    restitution, friction, e_bin, f_bin = get_grid_cell(scene_id)

    # Random visual properties (NOT correlated with physics)
    ball_color_rgb = [float(rng.random()), float(rng.random()), float(rng.random())]
    light_intensity = float(rng.uniform(1.0, 2.0))
    spawn_jitter = float(rng.uniform(-0.05, 0.05))

    ball_scale = 0.12
    mass = 1.0

    # --- Scene setup ---
    scene = kb.Scene(resolution=(resolution, resolution))
    scene.frame_end = 24      # 24 frames at 12fps = 2s of simulation
    scene.frame_rate = 12
    scene.step_rate = 240

    simulator = KubricSimulator(scene)
    renderer = None
    if render:
        from kubric.renderer.blender import Blender as KubricBlender
        renderer = KubricBlender(scene, samples_per_pixel=samples_per_pixel)

    # --- Floor ---
    floor_material = kb.PrincipledBSDFMaterial(
        color=kb.Color(0.5, 0.5, 0.5, 1.0))
    scene += kb.Cube(name="floor", scale=(5, 5, 0.1), position=(0, 0, -0.1),
                     static=True, friction=0.3, restitution=0.9,
                     material=floor_material)

    # --- Ramp ---
    # Thin slab rotated +30deg around Y axis.
    # VERIFIED: with +angle, t=-1 end is HIGH (-X, high Z), t=+1 is LOW (+X, low Z).
    # Ball starts at HIGH end (-X), slides RIGHT toward LOW end (+X), lands on floor.
    # Ramp friction = 1.0 so combined = ball.friction * 1.0 = ball.friction.
    ramp_half_len = 0.6      # total length 1.2m (shorter at 70° since ball accelerates fast)
    ramp_half_width = 0.5
    ramp_half_thick = 0.04   # thin: 8cm total
    ca = math.cos(RAMP_ANGLE_RAD)
    sa = math.sin(RAMP_ANGLE_RAD)

    # Position ramp so LOW end (+X) is just above floor:
    # Low end top surface: z = cz + (-half_len)*(-sa) + half_thick*ca
    #   wait, at t=+half_len: z = cz - half_len*sa + half_thick*ca
    # Want this ≈ 0.06 (just above floor)
    # cz = 0.06 + half_len*sa - half_thick*ca = 0.06 + 0.5 - 0.035 = 0.525
    ramp_cz = 0.06 + ramp_half_len * sa - ramp_half_thick * ca
    # Center X: low end at x = cx + half_len*ca + half_thick*sa
    # High end at x = cx - half_len*ca + half_thick*sa
    # Put low end near x=0: cx = 0.0 - half_len*ca - half_thick*sa
    #                          ≈ 0.0 - 0.205 - 0.038 = -0.243
    ramp_cx = -0.25

    ramp_material = kb.PrincipledBSDFMaterial(
        color=kb.Color(0.35, 0.35, 0.4, 1.0))

    ramp = kb.Cube(
        name="ramp",
        scale=(ramp_half_len, ramp_half_width, ramp_half_thick),
        position=(ramp_cx, 0, ramp_cz),
        static=True,
        friction=1.0,
        restitution=0.5,
        material=ramp_material,
    )
    ramp.quaternion = kb.Quaternion(axis=(0, 1, 0), angle=RAMP_ANGLE_RAD)
    scene += ramp

    # --- Ball ---
    # Spawn ~65% up from center toward HIGH end (t < 0, i.e. -X, +Z direction)
    frac_up = 0.65 + spawn_jitter
    t_ball = -frac_up * ramp_half_len  # negative t = toward high end
    # Ramp top surface at parameter t (after rotation around Y by +angle):
    #   surface_x = cx + t*ca + half_thick*sa
    #   surface_z = cz - t*sa + half_thick*ca
    surface_x = ramp_cx + t_ball * ca + ramp_half_thick * sa
    surface_z = ramp_cz - t_ball * sa + ramp_half_thick * ca
    # Ball center: offset by ball_scale along surface normal (sa, 0, ca)
    spawn_x = surface_x + ball_scale * sa
    spawn_z = surface_z + ball_scale * ca
    spawn_y = 0.0

    ball_material = kb.PrincipledBSDFMaterial(
        color=kb.Color(*ball_color_rgb, 1.0))
    ball = kb.Sphere(
        name="ball", scale=ball_scale,
        position=(spawn_x, spawn_y, spawn_z),
        velocity=(0, 0, 0), mass=mass,
        friction=friction, restitution=restitution,
        material=ball_material,
    )
    scene += ball

    # --- Lighting ---
    scene += kb.DirectionalLight(
        name="sun", position=(-1, -0.5, 3),
        look_at=(0, 0, 0.5), intensity=light_intensity)

    # --- Camera: side view, pulled back to capture ramp + bounce
    scene.camera = kb.PerspectiveCamera(
        name="camera",
        position=(0.0, -6, 1.8),
        look_at=(0.0, 0, 0.4))

    # --- Simulate ---
    simulator.run()

    # --- Extract positions ---
    pos_kf = ball.keyframes.get("position", {})
    positions = {}
    for frame in sorted(pos_kf.keys()):
        p = pos_kf[frame]
        positions[int(frame)] = [float(p[0]), float(p[1]), float(p[2])]

    sorted_frames = sorted(positions.keys())
    pos_array = np.array([positions[f] for f in sorted_frames], dtype=np.float32)

    x_pos = pos_array[:, 0] if len(pos_array) > 0 else np.array([0.0])
    z_pos = pos_array[:, 2] if len(pos_array) > 0 else np.array([0.0])
    x_travel = float(x_pos[-1] - x_pos[0]) if len(x_pos) > 1 else 0.0
    max_z = float(z_pos.max()) if len(z_pos) > 0 else 0.0
    min_z = float(z_pos.min()) if len(z_pos) > 0 else 0.0

    # --- Output directory ---
    scene_dir = os.path.join(output_dir, f"scene_{scene_id:04d}")
    os.makedirs(scene_dir, exist_ok=True)

    # --- Render (optional) ---
    n_rendered = 0
    if render and renderer is not None:
        frames_dict = renderer.render()
        if "rgba" in frames_dict:
            for i, frame_img in enumerate(frames_dict["rgba"]):
                from PIL import Image
                img = Image.fromarray(frame_img)
                img.save(os.path.join(scene_dir, f"rgba_{i:05d}.png"))
                n_rendered += 1

    np.save(os.path.join(scene_dir, "positions.npy"), pos_array)

    metadata = {
        "scene_id": scene_id,
        "restitution": restitution,
        "friction": friction,
        "elasticity_bin": e_bin,
        "friction_bin": f_bin,
        "mass": mass,
        "ball_scale": ball_scale,
        "ball_color_rgb": ball_color_rgb,
        "n_rendered": n_rendered,
        "frame_rate": scene.frame_rate,
        "ramp_angle_deg": RAMP_ANGLE_DEG,
        "light_intensity": light_intensity,
        "x_travel": x_travel,
        "max_z": max_z,
        "min_z": min_z,
    }

    # --- Cleanup ---
    try:
        import pybullet as pb
        pb.disconnect(simulator.physics_client)
    except Exception:
        pass
    del simulator
    if renderer is not None:
        del renderer
    del scene
    gc.collect()

    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Generate Kubric ramp dataset for compositional communication")
    parser.add_argument("--output_dir", type=str, default="output/ramp_dataset")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start_id", type=int, default=0)
    parser.add_argument("--render", action="store_true",
                        help="Render RGB frames (slow! ~30-60s/scene)")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # Burn RNG states for resume
    if args.start_id > 0:
        for _ in range(args.start_id):
            rng.random(), rng.random(), rng.random()  # color
            rng.uniform(1.0, 2.0)   # light
            rng.uniform(-0.05, 0.05)  # jitter

    os.makedirs(args.output_dir, exist_ok=True)

    index_path = os.path.join(args.output_dir, "index.json")
    if os.path.exists(index_path) and args.start_id > 0:
        with open(index_path) as f:
            all_metadata = json.load(f)
        print(f"Loaded {len(all_metadata)} existing scenes from index")
    else:
        all_metadata = []

    mode = "render" if args.render else "trajectories-only"
    print(f"\n{'='*60}")
    print(f"Kubric Ramp Dataset Generator")
    print(f"{'='*60}")
    print(f"  Mode:       {mode}")
    print(f"  Grid:       5x5 = 25 cells, {SCENES_PER_CELL}/cell = {TOTAL_SCENES} total")
    print(f"  Restitution: {RESTITUTION_LEVELS}")
    print(f"  Friction:    {FRICTION_LEVELS}")
    print(f"  Ramp angle:  {RAMP_ANGLE_DEG}deg (tan={math.tan(RAMP_ANGLE_RAD):.3f})")
    print(f"  Resolution:  {args.resolution}x{args.resolution}")
    print(f"  Frames:      24 at 12fps = 2.0s")
    print(f"  Output:      {args.output_dir}")
    print(f"  Start ID:    {args.start_id}")
    print(f"{'='*60}\n")

    t_start = time.time()
    errors = 0

    for i in range(args.start_id, TOTAL_SCENES):
        t_scene = time.time()
        try:
            meta = generate_scene(
                scene_id=i, output_dir=args.output_dir, rng=rng,
                render=args.render,
                resolution=args.resolution, samples_per_pixel=args.samples)
            all_metadata.append(meta)

            dt = time.time() - t_scene
            done = i - args.start_id + 1
            total_remaining = TOTAL_SCENES - args.start_id
            eta = (time.time() - t_start) / done * (total_remaining - done)

            if done % 10 == 0 or done <= 3:
                print(f"  [{done:4d}/{total_remaining}] scene_{i:04d}  "
                      f"e={meta['restitution']:.1f} f={meta['friction']:.1f}  "
                      f"x_travel={meta['x_travel']:+.2f}  "
                      f"rendered={meta['n_rendered']}f  "
                      f"{dt:.1f}s  ETA {eta/60:.0f}min")

            if done % 25 == 0:
                with open(index_path, "w") as f:
                    json.dump(all_metadata, f)

        except Exception as e:
            errors += 1
            print(f"  ERROR scene_{i:04d}: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            if errors > 20:
                print("Too many errors, aborting.", file=sys.stderr)
                break

    with open(index_path, "w") as f:
        json.dump(all_metadata, f, indent=2)

    dt_total = time.time() - t_start
    n = len(all_metadata)

    if n > 0:
        print(f"\n{'='*60}")
        print(f"Done! {n} scenes, {errors} errors, {dt_total/60:.1f}min ({dt_total/n:.1f}s/scene)")
        print(f"{'='*60}")

        rests = np.array([m["restitution"] for m in all_metadata])
        fricts = np.array([m["friction"] for m in all_metadata])
        x_travels = np.array([m["x_travel"] for m in all_metadata])

        print(f"\n  Grid distribution (scenes per cell):")
        for e_val in RESTITUTION_LEVELS:
            row = []
            for f_val in FRICTION_LEVELS:
                count = int(((rests == e_val) & (fricts == f_val)).sum())
                row.append(f"{count:3d}")
            print(f"    e={e_val:.1f}: [{', '.join(row)}]")

        print(f"\n  Physics signal (x_travel by friction):")
        for f_val in FRICTION_LEVELS:
            mask = fricts == f_val
            if mask.sum() > 0:
                xt = x_travels[mask]
                print(f"    f={f_val:.1f}: x_travel={xt.mean():+.3f} +/- {xt.std():.3f}")


if __name__ == "__main__":
    main()
