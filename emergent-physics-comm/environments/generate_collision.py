"""
Kubric Collision Dynamics Dataset Generator
=============================================
Phase 79: Two visually IDENTICAL spheres collide. The ONLY way to infer
their relative properties is by watching the collision unfold over time.

Property grid 5×5:
  mass_ratio_bin: Sphere B mass ∈ {1.0, 2.0, 3.0, 4.0, 5.0} kg (Sphere A always 1.0 kg)
  restitution_bin: collision restitution ∈ {0.1, 0.3, 0.5, 0.7, 0.9}
  24 scenes per combo = 600 total

Key physics (1D collision with coefficient of restitution e):
  v_a_final = (m_a - e*m_b) / (m_a + m_b) * v_a_initial
  v_b_final = (1 + e) * m_a / (m_a + m_b) * v_a_initial

Both spheres are visually identical (same size, color, texture) —
mass ratio is INVISIBLE in any single frame. You MUST track velocities
across time to distinguish mass ratios.

Output per scene: rgba_00000.png through rgba_00047.png (48 frames at 24fps = 2s)
Resolution: 256×256

Usage (from ~/AI/kubric/):
  docker run --rm --interactive \
    --user $(id -u):$(id -g) \
    --volume "$(pwd):/kubric" \
    kubricdockerhub/kubruntu:latest \
    python3 generate_collision_dataset.py --render

  # Resume:
  docker run --rm --interactive \
    --user $(id -u):$(id -g) \
    --volume "$(pwd):/kubric" \
    kubricdockerhub/kubruntu:latest \
    python3 generate_collision_dataset.py --render --start_id 300
"""

import argparse
import json
import logging
import os
import gc
import sys
import time
import numpy as np

import kubric as kb
from kubric.simulator.pybullet import PyBullet as KubricSimulator

logging.basicConfig(level="WARNING")

# Grid definition
MASS_RATIOS = [1.0, 2.0, 3.0, 4.0, 5.0]  # Sphere B mass (A is always 1.0)
RESTITUTION_LEVELS = [0.1, 0.3, 0.5, 0.7, 0.9]
SCENES_PER_CELL = 24
TOTAL_SCENES = len(MASS_RATIOS) * len(RESTITUTION_LEVELS) * SCENES_PER_CELL  # 600

SPHERE_A_MASS = 1.0
SPHERE_RADIUS = 0.15


def get_grid_cell(scene_id):
    """Map scene_id -> (mass_b, restitution, mass_bin, rest_bin)."""
    cell_idx = scene_id // SCENES_PER_CELL
    mass_bin = cell_idx // len(RESTITUTION_LEVELS)
    rest_bin = cell_idx % len(RESTITUTION_LEVELS)
    return (MASS_RATIOS[mass_bin], RESTITUTION_LEVELS[rest_bin], mass_bin, rest_bin)


def expected_velocities(m_a, m_b, e, v_a_init):
    """Compute expected post-collision velocities for 1D collision."""
    v_a_final = (m_a - e * m_b) / (m_a + m_b) * v_a_init
    v_b_final = (1 + e) * m_a / (m_a + m_b) * v_a_init
    return v_a_final, v_b_final


def generate_scene(scene_id, output_dir, rng, render=False, resolution=256,
                   samples_per_pixel=32):
    """Generate one collision scene with two identical-looking spheres."""
    mass_b, restitution, mass_bin, rest_bin = get_grid_cell(scene_id)

    # Random visual properties — BOTH spheres get SAME appearance
    sphere_color_rgb = [float(rng.random()), float(rng.random()), float(rng.random())]
    light_intensity = float(rng.uniform(1.0, 2.0))
    # Random initial velocity for sphere A (prevents memorization of exact speeds)
    initial_velocity = float(rng.uniform(1.5, 2.5))
    # Slight position jitter on sphere starting positions
    pos_jitter_a = float(rng.uniform(-0.05, 0.05))
    pos_jitter_b = float(rng.uniform(-0.05, 0.05))
    # Slight y-offset jitter (keeps collision nearly head-on but not pixel-perfect)
    y_jitter = float(rng.uniform(-0.02, 0.02))

    # --- Scene setup ---
    scene = kb.Scene(resolution=(resolution, resolution))
    scene.frame_end = 48      # 48 frames at 24fps = 2s
    scene.frame_rate = 24
    scene.step_rate = 240      # 240Hz substeps for collision accuracy

    simulator = KubricSimulator(scene)
    renderer = None
    if render:
        from kubric.renderer.blender import Blender as KubricBlender
        renderer = KubricBlender(scene, samples_per_pixel=samples_per_pixel)

    # --- Floor: flat gray surface ---
    floor_material = kb.PrincipledBSDFMaterial(
        color=kb.Color(0.45, 0.45, 0.45, 1.0))
    scene += kb.Cube(
        name="floor", scale=(5, 5, 0.05), position=(0, 0, -0.05),
        static=True, friction=0.05, restitution=0.5,
        material=floor_material)

    # --- Shared sphere material (IDENTICAL for both) ---
    sphere_material = kb.PrincipledBSDFMaterial(
        color=kb.Color(*sphere_color_rgb, 1.0))

    # --- Sphere A: starts left, moves right ---
    # Position so collision happens ~frame 12-16 (0.5-0.67s)
    # Separation = initial_velocity * 0.58s ≈ 1.16m at v=2.0
    # Center-to-center at contact = 2*radius = 0.30m
    # So initial center-to-center ≈ 1.16 + 0.30 = 1.46m
    sphere_a_x = -0.75 + pos_jitter_a
    sphere_b_x = 0.75 + pos_jitter_b

    sphere_a = kb.Sphere(
        name="sphere_a", scale=SPHERE_RADIUS,
        position=(sphere_a_x, y_jitter, SPHERE_RADIUS),
        velocity=(initial_velocity, 0, 0),
        mass=SPHERE_A_MASS,
        friction=0.05,
        restitution=restitution,
        material=sphere_material,
    )
    scene += sphere_a

    # --- Sphere B: starts right, stationary ---
    sphere_b = kb.Sphere(
        name="sphere_b", scale=SPHERE_RADIUS,
        position=(sphere_b_x, -y_jitter, SPHERE_RADIUS),
        velocity=(0, 0, 0),
        mass=mass_b,
        friction=0.05,
        restitution=restitution,
        material=sphere_material,
    )
    scene += sphere_b

    # --- Lighting ---
    scene += kb.DirectionalLight(
        name="sun", position=(0, -2, 4),
        look_at=(0, 0, 0), intensity=light_intensity)
    # Fill light from opposite side
    scene += kb.DirectionalLight(
        name="fill", position=(0, 2, 3),
        look_at=(0, 0, 0), intensity=light_intensity * 0.3)

    # --- Camera: 45-degree elevated view ---
    scene.camera = kb.PerspectiveCamera(
        name="camera",
        position=(0.0, -4.0, 3.5),
        look_at=(0.0, 0.0, 0.0))

    # --- Simulate ---
    simulator.run()

    # --- Extract positions for both spheres ---
    def extract_positions(obj):
        pos_kf = obj.keyframes.get("position", {})
        positions = {}
        for frame in sorted(pos_kf.keys()):
            p = pos_kf[frame]
            positions[int(frame)] = [float(p[0]), float(p[1]), float(p[2])]
        return positions

    pos_a = extract_positions(sphere_a)
    pos_b = extract_positions(sphere_b)

    sorted_frames = sorted(pos_a.keys())
    pos_a_array = np.array([pos_a[f] for f in sorted_frames], dtype=np.float32)
    pos_b_array = np.array([pos_b[f] for f in sorted_frames], dtype=np.float32)

    # Compute velocities from positions (finite differences)
    dt = 1.0 / scene.frame_rate
    if len(pos_a_array) > 1:
        vel_a = np.diff(pos_a_array, axis=0) / dt
        vel_b = np.diff(pos_b_array, axis=0) / dt
        # Pre-collision velocity (frame 1-5 average)
        pre_vel_a = vel_a[:5, 0].mean() if len(vel_a) > 5 else vel_a[0, 0]
        # Post-collision velocity (last 5 frames average)
        post_vel_a = vel_a[-5:, 0].mean() if len(vel_a) > 5 else vel_a[-1, 0]
        post_vel_b = vel_b[-5:, 0].mean() if len(vel_b) > 5 else vel_b[-1, 0]
    else:
        pre_vel_a = initial_velocity
        post_vel_a = 0.0
        post_vel_b = 0.0

    # Expected velocities for sanity check
    exp_va, exp_vb = expected_velocities(
        SPHERE_A_MASS, mass_b, restitution, initial_velocity)

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

    # Save trajectories
    np.save(os.path.join(scene_dir, "positions_a.npy"), pos_a_array)
    np.save(os.path.join(scene_dir, "positions_b.npy"), pos_b_array)

    metadata = {
        "scene_id": scene_id,
        "mass_ratio_bin": mass_bin,
        "restitution_bin": rest_bin,
        "sphere_a_mass": SPHERE_A_MASS,
        "sphere_b_mass": mass_b,
        "restitution": restitution,
        "initial_velocity": initial_velocity,
        "sphere_color_rgb": sphere_color_rgb,
        "light_intensity": light_intensity,
        "n_rendered": n_rendered,
        "frame_rate": scene.frame_rate,
        "n_frames": scene.frame_end,
        "resolution": resolution,
        # Measured velocities
        "pre_collision_vel_a": float(pre_vel_a),
        "post_collision_vel_a": float(post_vel_a),
        "post_collision_vel_b": float(post_vel_b),
        # Expected velocities (1D theory)
        "expected_vel_a_final": float(exp_va),
        "expected_vel_b_final": float(exp_vb),
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
        description="Generate Kubric collision dynamics dataset")
    parser.add_argument("--output_dir", type=str, default="output/collision_dataset")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=123)
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
            rng.uniform(1.5, 2.5)   # velocity
            rng.uniform(-0.05, 0.05)  # pos jitter a
            rng.uniform(-0.05, 0.05)  # pos jitter b
            rng.uniform(-0.02, 0.02)  # y jitter

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
    print(f"Kubric Collision Dynamics Dataset Generator")
    print(f"{'='*60}")
    print(f"  Mode:         {mode}")
    print(f"  Grid:         5×5 = 25 cells, {SCENES_PER_CELL}/cell = {TOTAL_SCENES} total")
    print(f"  Mass ratios:  {MASS_RATIOS} (Sphere B mass, A=1.0)")
    print(f"  Restitution:  {RESTITUTION_LEVELS}")
    print(f"  Resolution:   {args.resolution}×{args.resolution}")
    print(f"  Frames:       48 at 24fps = 2.0s")
    print(f"  Output:       {args.output_dir}")
    print(f"  Start ID:     {args.start_id}")
    print(f"{'='*60}\n")

    # Print expected velocity table
    print(f"  Expected post-collision velocities (v_a_init=2.0):")
    print(f"  {'mass_b':>6} | {'e=0.1':>8} {'e=0.3':>8} {'e=0.5':>8} {'e=0.7':>8} {'e=0.9':>8}")
    print(f"  {'------':>6}-+-{'--------':>8}-{'--------':>8}-{'--------':>8}-{'--------':>8}-{'--------':>8}")
    for mb in MASS_RATIOS:
        row = f"  {mb:>6.1f} |"
        for e in RESTITUTION_LEVELS:
            va, vb = expected_velocities(1.0, mb, e, 2.0)
            row += f" {vb:>7.3f} "
        print(row)
    print()

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
                      f"m_b={meta['sphere_b_mass']:.1f} e={meta['restitution']:.1f}  "
                      f"v_b_post={meta['post_collision_vel_b']:+.2f} "
                      f"(exp={meta['expected_vel_b_final']:+.2f})  "
                      f"rendered={meta['n_rendered']}f  "
                      f"{dt:.1f}s  ETA {eta/60:.0f}min")

            if done % 50 == 0:
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

    # Save final index
    with open(index_path, "w") as f:
        json.dump(all_metadata, f, indent=2)

    dt_total = time.time() - t_start
    n = len(all_metadata)

    if n > 0:
        print(f"\n{'='*60}")
        print(f"Done! {n} scenes, {errors} errors, {dt_total/60:.1f}min ({dt_total/n:.1f}s/scene)")
        print(f"{'='*60}")

        mass_bins = np.array([m["mass_ratio_bin"] for m in all_metadata])
        rest_bins = np.array([m["restitution_bin"] for m in all_metadata])
        masses_b = np.array([m["sphere_b_mass"] for m in all_metadata])
        rests = np.array([m["restitution"] for m in all_metadata])
        post_vb = np.array([m["post_collision_vel_b"] for m in all_metadata])
        exp_vb = np.array([m["expected_vel_b_final"] for m in all_metadata])

        print(f"\n  Grid distribution (scenes per cell):")
        for mb_val in MASS_RATIOS:
            row = []
            for r_val in RESTITUTION_LEVELS:
                count = int(((masses_b == mb_val) & (rests == r_val)).sum())
                row.append(f"{count:3d}")
            print(f"    m_b={mb_val:.1f}: [{', '.join(row)}]")

        print(f"\n  Post-collision Sphere B velocity by mass ratio:")
        for mb_val in MASS_RATIOS:
            mask = masses_b == mb_val
            if mask.sum() > 0:
                vb = post_vb[mask]
                ev = exp_vb[mask]
                print(f"    m_b={mb_val:.1f}: measured={vb.mean():+.3f}±{vb.std():.3f}  "
                      f"expected={ev.mean():+.3f}±{ev.std():.3f}")

        print(f"\n  Post-collision Sphere B velocity by restitution:")
        for r_val in RESTITUTION_LEVELS:
            mask = rests == r_val
            if mask.sum() > 0:
                vb = post_vb[mask]
                ev = exp_vb[mask]
                print(f"    e={r_val:.1f}: measured={vb.mean():+.3f}±{vb.std():.3f}  "
                      f"expected={ev.mean():+.3f}±{ev.std():.3f}")

        # Sanity: velocity error
        vel_errors = np.abs(post_vb - exp_vb)
        print(f"\n  Velocity prediction error (measured vs theory):")
        print(f"    Mean: {vel_errors.mean():.3f} m/s")
        print(f"    Max:  {vel_errors.max():.3f} m/s")
        print(f"    >0.5: {(vel_errors > 0.5).sum()}/{n} scenes")


if __name__ == "__main__":
    main()
