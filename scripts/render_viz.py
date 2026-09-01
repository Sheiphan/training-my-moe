#!/usr/bin/env python
"""Documents (and optionally runs) every architecture-math Manim scene from
Task 13 (Requirement 7.7) plus the training/run-observability scenes from
Task 14 (Requirement 8.1-8.2): each one is independently runnable via its
own `manim` invocation, with no shared state and no need to run any other
scene first.

Usage:
    uv run python scripts/render_viz.py --help
    uv run python scripts/render_viz.py --scene attention
    uv run python scripts/render_viz.py --scene all --quality m --preview
"""
from __future__ import annotations

import argparse
import subprocess
import sys

# name -> (scene file, Scene class) -- the 5 files task 13.1-13.5 added, plus
# task 14's two training/run-observability scenes (Requirement 8.1-8.2).
_SCENES = {
    "embedding": ("qwenmoe/viz/scenes_embedding.py", "EmbeddingPositionalScene"),
    "attention": ("qwenmoe/viz/scenes_attention.py", "AttentionScene"),
    "moe": ("qwenmoe/viz/scenes_moe.py", "MoERoutingScene"),
    "norm_act": ("qwenmoe/viz/scenes_norm_act.py", "NormActivationScene"),
    "output": ("qwenmoe/viz/scenes_output.py", "OutputProjectionScene"),
    "loss_curve": ("qwenmoe/viz/scenes_training.py", "LossCurveScene"),
    "expert_utilization": ("qwenmoe/viz/scenes_training.py", "ExpertUtilizationScene"),
}


def _epilog() -> str:
    lines = [
        "Each scene below is independently runnable via the exact `manim` command shown",
        "(Requirement 7.7) -- run any single one in a fresh process without running the",
        "others first. loss_curve/expert_utilization (Requirement 8.1-8.2) read a training",
        "log instead of a checkpoint and need one to exist first (Requirement 8.3):",
        "",
    ]
    for name, (path, scene_cls) in _SCENES.items():
        lines.append(f"  {name}:")
        lines.append(f"      manim -pql {path} {scene_cls}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=_epilog(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--scene", choices=[*_SCENES, "all"],
        help="which scene to actually render (omit to just print the commands above)",
    )
    parser.add_argument(
        "--quality", choices=["l", "m", "h"], default="l",
        help="manim quality flag: l=480p15, m=720p30, h=1080p60 (default: l)",
    )
    parser.add_argument(
        "--preview", action="store_true", help="pass manim's -p flag to open each video after rendering"
    )
    args = parser.parse_args()

    if args.scene is None:
        parser.print_help()
        return

    flag = "-" + ("p" if args.preview else "") + "q" + args.quality
    names = list(_SCENES) if args.scene == "all" else [args.scene]
    for name in names:
        path, scene_cls = _SCENES[name]
        command = ["manim", flag, path, scene_cls]
        print(f"$ {' '.join(command)}")
        subprocess.run(command, check=True)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)
