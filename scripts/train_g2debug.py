"""Launch one frozen G2.0-debug formal training configuration."""

from __future__ import annotations

import argparse
from pathlib import Path

import g2debug_common
import train


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", required=True, choices=sorted(g2debug_common.TASKS))
    parser.add_argument("--workers", required=True, type=int)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    args = parser.parse_args()
    config = g2debug_common.effective_config(
        args.config,
        workers=args.workers,
        checkpoint_dir=args.checkpoint_dir,
    )
    train.main(config)


if __name__ == "__main__":
    main()
