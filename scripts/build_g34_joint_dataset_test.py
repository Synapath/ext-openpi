from pathlib import Path

import numpy as np
import pytest

from scripts import build_g34_joint_dataset as builder


def test_choose_instruction_is_episode_deterministic() -> None:
    values = np.asarray([b"first", b"second"])

    assert builder.choose_instruction(values, 0) == "first"
    assert builder.choose_instruction(values, 3) == "second"


def test_read_instructions_accepts_scalar_json_and_legacy_array(tmp_path) -> None:
    import h5py
    import numpy as np

    path = tmp_path / "instructions.hdf5"
    with h5py.File(path, "w") as episode:
        scalar = episode.create_dataset(
            "scalar",
            data=np.bytes_('["first", "second"]'),
        )
        legacy = episode.create_dataset(
            "legacy",
            data=np.asarray([b"first", b"second"]),
        )
        assert builder.read_instructions(scalar) == ["first", "second"]
        assert builder.read_instructions(legacy) == [b"first", b"second"]


def test_next_state_actions() -> None:
    state = np.arange(42, dtype=np.float32).reshape(3, 14)

    observed = builder.next_state_actions(state)

    np.testing.assert_array_equal(observed[0], state[1])
    np.testing.assert_array_equal(observed[1], state[2])
    np.testing.assert_array_equal(observed[2], state[2])


def test_tree_sha256_depends_on_paths_and_contents(tmp_path: Path) -> None:
    (tmp_path / "a").write_bytes(b"value")
    first = builder.tree_sha256(tmp_path)
    (tmp_path / "a").rename(tmp_path / "b")

    assert builder.tree_sha256(tmp_path) != first


def test_source_files_requires_frozen_episode_count(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Expected 50"):
        builder.source_files(tmp_path, "adjust_bottle")


def test_stable_tree_sha256_matches_tree_sha256_on_quiet_tree(tmp_path: Path) -> None:
    (tmp_path / "a").write_bytes(b"value")

    assert builder.stable_tree_sha256(tmp_path) == builder.tree_sha256(tmp_path)


def test_stable_tree_sha256_rejects_concurrent_writes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "a").write_bytes(b"value")
    original = builder.tree_sha256

    def hash_then_write(root: Path) -> str:
        digest = original(root)
        (root / "late").write_bytes(b"footer")
        return digest

    monkeypatch.setattr(builder, "tree_sha256", hash_then_write)

    with pytest.raises(RuntimeError, match="changed while hashing"):
        builder.stable_tree_sha256(tmp_path)
