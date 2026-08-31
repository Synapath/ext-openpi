"""Run the bounded exact-loader G2.0-debug GPU update benchmark."""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import g2debug_common
import train_capacity


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", required=True, choices=sorted(g2debug_common.TASKS))
    parser.add_argument("--workers", required=True, type=int)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"capacity output identity already exists: {args.output_root}")
    args.output_root.mkdir(parents=True)
    config = g2debug_common.effective_config(
        args.config,
        workers=args.workers,
        checkpoint_dir=args.output_root / "unused-checkpoints",
        exp_name=f"nonformal-capacity-{args.config}-w{args.workers}",
    )
    train_capacity.main(dataclasses.replace(config, wandb_enabled=False))


if __name__ == "__main__":
    main()
