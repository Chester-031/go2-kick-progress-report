"""Render recorded episodes on the real Go2 model with Isaac Sim 6's RTX renderer.

Isaac Sim 4.5 (the Isaac Lab 2.0.2 stack the policies train in) segfaults
as soon as its RTX renderer starts on NVIDIA driver 610 (Torch's GPU
nodes since October 2026; isaac-sim/IsaacSim#650), so cameras cannot be
used there.  Physics and policies still run fine headless, so
``scripts/record_trajectories.py`` saves every robot link pose and the
ball pose, and this script replays those poses kinematically (no physics:
each link is placed where the simulation had it) in Isaac Sim 6, which
renders on that driver.  It rebuilds the soccer field of
``soccer_scene_cfg.py``, follows the robot and ball with a side camera and
writes one JPEG per frame (25 fps, every second 50 Hz tick) into
``OUT_DIR/<episode key>/``; ``scripts/encode_replay.py`` turns the frames
into captioned MP4s.

Run inside the Isaac Sim 6 container:

    /isaac-sim/python.sh scripts/replay_isaac6.py --trajectories traj.npz \
        --robot-usd go2.usd --out-dir frames [--key position_d1_e3 ...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# Field geometry of soccer_scene_cfg.py (metres).
FIELD_HALF_LENGTH = 4.0
FIELD_HALF_WIDTH = 2.5
GOAL_HALF_WIDTH = 0.6
GOAL_DEPTH = 0.6
WALL_HEIGHT = 0.35
WALL_THICKNESS = 0.10
BALL_RADIUS = 0.11
WALL_COLOR = (0.15, 0.25, 0.55)
GRASS_COLOR = (0.08, 0.28, 0.10)
LINE_COLOR = (0.92, 0.92, 0.92)
# The recording is 50 Hz; the video plays every second tick, in real time.
FRAME_STRIDE = 2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trajectories", type=Path, required=True, help="npz from record_trajectories.py")
    parser.add_argument("--robot-usd", required=True, help="the Go2 USD Isaac Lab spawns (path or URL)")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--key", action="append", help="episode keys to render (default: all)")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--samples", type=int, default=64,
                        help="path-tracing samples per pixel (each frame converges on its own, so the "
                             "teleport-style pose updates leave no ghosting)")
    args = parser.parse_args()

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True, "width": args.width, "height": args.height, "anti_aliasing": 0})

    import carb.settings
    import omni.replicator.core as rep

    # Path tracing with the OptiX denoiser: the real-time renderer's
    # denoising needs DLSS (NGX), which is unavailable in the container.
    # The path tracer accumulates up to totalSpp samples over successive
    # renders of an unchanged scene, `spp` per render, so each video frame
    # is rendered `passes` times.
    samples_per_pass = min(args.samples, 16)
    passes = -(-args.samples // samples_per_pass)
    rep.settings.set_render_pathtraced(samples_per_pixel=passes * samples_per_pass)
    settings = carb.settings.get_settings()
    settings.set("/rtx/pathtracing/spp", samples_per_pass)
    settings.set("/rtx/pathtracing/optixDenoiser/enabled", True)
    import omni.usd
    from PIL import Image
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade

    data = np.load(args.trajectories)
    index = json.loads(bytes(data["index_json"]).decode())
    body_names: list[str] = index["body_names"]
    episodes = [episode for episode in index["episodes"] if not args.key or episode["key"] in args.key]

    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    def material(name: str, color: tuple[float, float, float], roughness: float) -> UsdShade.Material:
        path = f"/World/Looks/{name}"
        made = UsdShade.Material.Define(stage, path)
        shader = UsdShade.Shader.Define(stage, path + "/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
        made.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        return made

    def box(path: str, size: tuple[float, float, float], center: tuple[float, float, float],
            look: UsdShade.Material) -> None:
        cube = UsdGeom.Cube.Define(stage, path)
        cube.CreateSizeAttr(1.0)
        cube.AddTranslateOp().Set(Gf.Vec3d(*center))
        cube.AddScaleOp().Set(Gf.Vec3f(*size))
        UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(look)

    grass = material("Grass", GRASS_COLOR, 0.9)
    wall = material("Wall", WALL_COLOR, 0.8)
    line = material("Line", LINE_COLOR, 0.7)
    white = material("Ball", (0.95, 0.95, 0.95), 0.5)
    black = material("BallPatch", (0.05, 0.05, 0.05), 0.5)

    box("/World/Field/Grass", (14.0, 9.0, 0.02), (0.0, 0.0, -0.01), grass)
    lines = [((2 * FIELD_HALF_LENGTH, 0.05, 0.002), (0.0, y, 0.001)) for y in (-FIELD_HALF_WIDTH + 0.05,
                                                                             FIELD_HALF_WIDTH - 0.05)]
    lines += [((0.05, 2 * FIELD_HALF_WIDTH, 0.002), (x, 0.0, 0.001)) for x in (-FIELD_HALF_LENGTH + 0.05, 0.0,
                                                                             FIELD_HALF_LENGTH - 0.05)]
    for number, (size, center) in enumerate(lines):
        box(f"/World/Field/Line{number}", size, center, line)
    segment = FIELD_HALF_WIDTH - GOAL_HALF_WIDTH
    walls = [((2 * FIELD_HALF_LENGTH + 2 * WALL_THICKNESS, WALL_THICKNESS, WALL_HEIGHT), (0.0, side * FIELD_HALF_WIDTH))
             for side in (-1, 1)]
    for end in (-1, 1):
        walls += [((WALL_THICKNESS, segment, WALL_HEIGHT), (end * FIELD_HALF_LENGTH,
                                                            side * (GOAL_HALF_WIDTH + segment / 2)))
                  for side in (-1, 1)]
        walls += [((GOAL_DEPTH, WALL_THICKNESS, WALL_HEIGHT), (end * (FIELD_HALF_LENGTH + GOAL_DEPTH / 2),
                                                               side * GOAL_HALF_WIDTH)) for side in (-1, 1)]
        walls.append(((WALL_THICKNESS, 2 * GOAL_HALF_WIDTH + WALL_THICKNESS, WALL_HEIGHT),
                      (end * (FIELD_HALF_LENGTH + GOAL_DEPTH), 0.0)))
    for number, (size, (x, y)) in enumerate(walls):
        box(f"/World/Field/Wall{number}", size, (x, y, WALL_HEIGHT / 2), wall)

    UsdLux.DomeLight.Define(stage, "/World/Sky").CreateIntensityAttr(900.0)
    sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
    sun.CreateIntensityAttr(2500.0)
    sun.CreateAngleAttr(1.0)
    UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(35.0, 0.0, 25.0))

    # The ball, with two dark patches so its spin shows.
    ball = UsdGeom.Xform.Define(stage, "/World/Ball")
    ball_op = ball.AddTransformOp(UsdGeom.XformOp.PrecisionDouble)
    sphere = UsdGeom.Sphere.Define(stage, "/World/Ball/Sphere")
    sphere.CreateRadiusAttr(BALL_RADIUS)
    UsdShade.MaterialBindingAPI.Apply(sphere.GetPrim()).Bind(white)
    for number, direction in enumerate(((1.0, 0.0, 0.0), (-0.5, 0.0, 0.866))):
        patch = UsdGeom.Sphere.Define(stage, f"/World/Ball/Patch{number}")
        patch.CreateRadiusAttr(0.045)
        patch.AddTranslateOp().Set(Gf.Vec3d(*(0.08 * np.array(direction))))
        UsdShade.MaterialBindingAPI.Apply(patch.GetPrim()).Bind(black)

    # The robot: the Go2 asset, each link driven by its recorded pose.
    robot = stage.DefinePrim("/World/Robot", "Xform")
    robot.GetReferences().AddReference(args.robot_usd)
    app.update()
    by_name: dict[str, Usd.Prim] = {}
    for prim in Usd.PrimRange(robot):
        if prim.GetName() in body_names and prim.GetName() not in by_name:
            by_name[prim.GetName()] = prim
    missing = [name for name in body_names if name not in by_name]
    if missing:
        raise SystemExit(f"links not found in {args.robot_usd}: {missing}")
    # A link's local transform is its recorded world pose relative to its
    # parent: the parent link's recorded pose, or the fixed pose of a
    # non-link parent (read once, before any link moves).
    parent_link: dict[str, str | None] = {}
    fixed_parent: dict[str, Gf.Matrix4d] = {}
    link_ops = {}
    for name in body_names:
        parent = by_name[name].GetParent()
        if parent.GetName() in by_name:
            parent_link[name] = parent.GetName()
            continue
        parent_link[name] = None
        fixed_parent[name] = UsdGeom.Xformable(parent).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        ancestor = parent
        while ancestor.IsValid():
            if ancestor.GetName() in by_name:
                raise SystemExit(f"{by_name[name].GetPath()} is nested in link {ancestor.GetPath()} through a "
                                 "non-link prim; replay cannot place it")
            ancestor = ancestor.GetParent()
    print("[replay] links:", {name: str(by_name[name].GetPath()) for name in body_names}, flush=True)
    for name in body_names:
        xformable = UsdGeom.Xformable(by_name[name])
        xformable.ClearXformOpOrder()
        link_ops[name] = xformable.AddTransformOp(UsdGeom.XformOp.PrecisionDouble)

    def pose(position: np.ndarray, quaternion: np.ndarray) -> Gf.Matrix4d:
        """World matrix of a position and a (w, x, y, z) quaternion."""

        matrix = Gf.Matrix4d(1.0)
        matrix.SetRotate(Gf.Quatd(float(quaternion[0]), Gf.Vec3d(*map(float, quaternion[1:]))))
        matrix.SetTranslateOnly(Gf.Vec3d(*map(float, position)))
        return matrix

    camera = UsdGeom.Camera.Define(stage, "/World/Camera")
    camera.CreateFocalLengthAttr(18.0)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.05, 100.0))
    camera_op = camera.AddTransformOp(UsdGeom.XformOp.PrecisionDouble)
    render_product = rep.create.render_product(str(camera.GetPath()), (args.width, args.height))
    rgb = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb.attach(render_product)

    base = body_names.index("base")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for episode in episodes:
        key = episode["key"]
        positions = data[key + "_bodies"]
        quaternions = data[key + "_body_quats"]
        ball_positions = data[key + "_ball"]
        ball_quaternions = data[key + "_ball_quat"]
        folder = args.out_dir / key
        folder.mkdir(parents=True, exist_ok=True)
        target = None
        for frame, tick in enumerate(range(0, len(positions), FRAME_STRIDE)):
            world = {name: pose(positions[tick, i], quaternions[tick, i]) for i, name in enumerate(body_names)}
            for name in body_names:
                parent = parent_link[name]
                frame_of_parent = fixed_parent[name] if parent is None else world[parent]
                # USD multiplies row vectors: world = local * parent world.
                link_ops[name].Set(world[name] * frame_of_parent.GetInverse())
            ball_op.Set(pose(ball_positions[tick], ball_quaternions[tick]))
            # Side camera on the robot-ball midpoint, smoothed so it glides.
            focus = 0.5 * (positions[tick, base] + ball_positions[tick]).astype(np.float64)
            focus[2] = 0.2
            target = focus if target is None else 0.92 * target + 0.08 * focus
            eye = target + np.array([-1.0, -3.0, 1.6])
            view = Gf.Matrix4d(1.0).SetLookAt(Gf.Vec3d(*map(float, eye)), Gf.Vec3d(*map(float, target)),
                                              Gf.Vec3d(0.0, 0.0, 1.0))
            camera_op.Set(view.GetInverse())
            # The annotator returns empty frames while the renderer warms up.
            for _ in range(100):
                rep.orchestrator.step(rt_subframes=passes, delta_time=0.0, pause_timeline=True)
                image = rgb.get_data()
                if np.asarray(image).size:
                    break
            else:
                raise SystemExit("the renderer returned no image after 100 steps")
            Image.fromarray(np.asarray(image)[..., :3]).save(folder / f"{frame:05d}.jpg", quality=92)
        print(f"[replay] {key}: {frame + 1} frames", flush=True)
    rep.orchestrator.wait_until_complete()
    app.close()


if __name__ == "__main__":
    main()
