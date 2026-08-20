#!/usr/bin/env python3
from pathlib import Path
import argparse, json

def replace_one(text, old, new, label):
    n = text.count(old)
    if n != 1:
        raise RuntimeError(f'{label} count={n}')
    return text.replace(old, new, 1)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--source',required=True); ap.add_argument('--out',required=True); a=ap.parse_args()
    text=Path(a.source).read_text(encoding='utf-8')
    text=replace_one(text, '    import imageio.v2 as imageio\n', '    metric_only = os.environ.get("R865_METRIC_ONLY") == "1"\n', 'metric flag')
    text=replace_one(text, '    video_path: Path,\n', '    video_path: Path | None,\n', 'video annotation')
    text=replace_one(text, '    render_env: Any = env.low_env\n', '    render_env: Any = env.low_env\n    if metric_only:\n        render_env = None\n', 'render guard')
    text=replace_one(text, '    while not callable(getattr(render_env, "render", None)):\n', '    while render_env is not None and not callable(getattr(render_env, "render", None)):\n', 'render loop')
    text=replace_one(text, '    print(\n        "[V4D_EVAL] "\n        f"render_chain={\' -> \'.join(chain)}",\n        flush=True,\n    )\n', '    if not metric_only:\n        print(\n            "[V4D_EVAL] "\n            f"render_chain={\' -> \'.join(chain)}",\n            flush=True,\n        )\n', 'render print')
    text=replace_one(text, '    video_path.parent.mkdir(\n        parents=True,\n        exist_ok=True,\n    )\n', '    if video_path is not None:\n        video_path.parent.mkdir(parents=True, exist_ok=True)\n', 'video mkdir')
    text=replace_one(text, '    writer = imageio.get_writer(\n        video_path,\n        fps=fps,\n        codec="libx264",\n        quality=8,\n        pixelformat="yuv420p",\n    )\n', '    writer = None\n    if video_path is not None:\n        import imageio.v2 as imageio\n        writer = imageio.get_writer(video_path, fps=fps, codec="libx264", quality=8, pixelformat="yuv420p")\n', 'writer')
    text=replace_one(text, '                frame = render_env.render()\n                if isinstance(frame, list):\n                    frame = frame[0]\n                if frame is None:\n                    raise RuntimeError(\n                        "render() returned no frame."\n                    )\n                writer.append_data(np.asarray(frame))\n', '                if writer is not None:\n                    frame = render_env.render()\n                    if isinstance(frame, list):\n                        frame = frame[0]\n                    if frame is None:\n                        raise RuntimeError("render() returned no frame.")\n                    writer.append_data(np.asarray(frame))\n', 'frame')
    text=replace_one(text, '    finally:\n        writer.close()\n', '    finally:\n        if writer is not None:\n            writer.close()\n', 'writer close')
    text=replace_one(text, '        "video": str(video_path),\n', '        "video": None if video_path is None else str(video_path),\n', 'summary video')
    text=replace_one(text, '    print(\n        f"V4D_EVAL_VIDEO={video_path}",\n        flush=True,\n    )\n', '    if video_path is not None:\n        print(f"V4D_EVAL_VIDEO={video_path}", flush=True)\n    else:\n        print("V4D_EVAL_METRIC_ONLY=true", flush=True)\n', 'video print')
    text=replace_one(text, '    if args_cli.eval_video:\n', '    metric_only = os.environ.get("R865_METRIC_ONLY") == "1"\n    if args_cli.eval_video or metric_only:\n', 'main branch')
    text=replace_one(text, '        video_path = Path(\n            args_cli.eval_video\n        ).resolve()\n', '        video_path = None if metric_only else Path(args_cli.eval_video).resolve()\n', 'main video path')
    old_default = ("        trajectory_path = Path(\n"
        "            args_cli.trajectory_csv\n"
        "            or (\n"
        "                str(video_path.with_suffix(\"\"))\n"
        "                + \"_trajectory.csv\"\n"
        "            )\n"
        "        ).resolve()\n"
        "        summary_path = Path(\n"
        "            args_cli.eval_summary\n"
        "            or (\n"
        "                str(video_path.with_suffix(\"\"))\n"
        "                + \"_summary.json\"\n"
        "            )\n"
        "        ).resolve()\n")
    new_default = ("        metric_stem = Path(\n"
        "            args_cli.eval_summary or args_cli.trajectory_csv or \"v4d_metric_only\"\n"
        "        ).resolve().with_suffix(\"\")\n"
        "        trajectory_path = Path(args_cli.trajectory_csv or (str(metric_stem) + \"_trajectory.csv\")).resolve()\n"
        "        summary_path = Path(args_cli.eval_summary or (str(metric_stem) + \"_summary.json\")).resolve()\n")
    text=replace_one(text, old_default, new_default, 'metric output paths')
    out=Path(a.out); out.write_text(text,encoding='utf-8')
    Path(str(out)+'.patch_report.json').write_text(json.dumps({'version':'r8_66_v4d_metric_only','training':False,'video_disabled':True,'trajectory_enabled':True,'summary_enabled':True},indent=2)+'\n')
    print('R866_METRIC_ONLY_PATCH_OK')
if __name__=='__main__': main()
