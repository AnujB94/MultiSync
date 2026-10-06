from __future__ import annotations

import argparse
from pathlib import Path

from dualpose.app import DualPoseApp
from dualpose.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DualPose webcam + IP camera 3D viewer")
    parser.add_argument(
        "--config",
        type=str,
        default="config.local.yaml",
        help="Path to YAML configuration file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    cfg = load_config(config_path)
    app = DualPoseApp(cfg)
    app.run()


if __name__ == "__main__":
    main()
