from argparse import Namespace
from pathlib import Path

from scripts import build_g51_easy5_dataset as builder


def _args(tmp_path: Path) -> Namespace:
    return Namespace(
        generated_root=tmp_path / "generated",
        adjust_bottle_root=tmp_path / "adjust",
    )


def test_source_root_routes_reused_and_generated_tasks(tmp_path: Path) -> None:
    args = _args(tmp_path)

    assert builder.source_root(args, "adjust_bottle") == tmp_path / "adjust"
    assert builder.source_root(args, "press_stapler") == (
        tmp_path / "generated" / "demo_clean" / "press_stapler" / "aloha_agilex"
    )


def test_source_files_requires_exactly_fifty_episodes(tmp_path: Path) -> None:
    args = _args(tmp_path)
    data = builder.source_root(args, "turn_switch") / "data"
    data.mkdir(parents=True)
    for index in range(50):
        (data / f"episode_{index:07d}.hdf5").touch()

    assert len(builder.source_files(args, "turn_switch")) == 50
