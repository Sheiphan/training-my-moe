"""Task 14 (Requirement 8.3): a missing training log must fail with a
clear, actionable error -- not a raw stack trace -- before any Manim
rendering is attempted. Tests `load_training_log` directly (the parsing
logic) plus both Scene subclasses' `construct()` to confirm the error
fires before either one creates a single Manim mobject.
"""
from __future__ import annotations

import json

import pytest

from qwenmoe.viz.scenes_training import (
    ExpertUtilizationScene,
    LossCurveScene,
    load_training_log,
)

_STEP_RECORD = {
    "type": "step",
    "step": 10,
    "train_loss": 3.5,
    "aux_loss": 0.01,
    "lr": 1e-3,
    "elapsed_s": 0.02,
    "expert_token_counts": [[5, 3], [2, 6]],
}
_EVAL_RECORD = {"type": "eval", "step": 10, "val_loss": 3.4, "val_perplexity": 30.0}


def _write_jsonl(path, records) -> None:
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")


# -- load_training_log: missing / empty / malformed file (Requirement 8.3) --


def test_missing_log_file_raises_actionable_filenotfounderror(tmp_path):
    missing = tmp_path / "does_not_exist" / "train.jsonl"
    with pytest.raises(FileNotFoundError) as exc_info:
        load_training_log(missing)
    message = str(exc_info.value)
    assert str(missing) in message
    assert "scripts/train.py" in message  # actionable: tells the user how to generate one


def test_empty_log_file_raises_clear_valueerror(tmp_path):
    log_path = tmp_path / "train.jsonl"
    log_path.write_text("")
    with pytest.raises(ValueError, match="no 'step' records"):
        load_training_log(log_path)


def test_log_file_with_only_blank_lines_raises_clear_valueerror(tmp_path):
    log_path = tmp_path / "train.jsonl"
    log_path.write_text("\n\n   \n")
    with pytest.raises(ValueError, match="no 'step' records"):
        load_training_log(log_path)


def test_malformed_json_line_raises_clear_valueerror_not_raw_traceback(tmp_path):
    log_path = tmp_path / "train.jsonl"
    log_path.write_text('{"type": "step", "step": 1,\n')  # truncated JSON
    with pytest.raises(ValueError, match="not valid JSON"):
        load_training_log(log_path)


# -- load_training_log: real parsing behavior ---------------------------


def test_parses_step_and_eval_records_in_file_order(tmp_path):
    log_path = tmp_path / "train.jsonl"
    second_step = {**_STEP_RECORD, "step": 20, "train_loss": 3.0, "expert_token_counts": [[4, 4], [3, 5]]}
    _write_jsonl(log_path, [_STEP_RECORD, _EVAL_RECORD, second_step])

    log = load_training_log(log_path)

    assert log.steps == [10, 20]
    assert log.train_loss == [3.5, 3.0]
    assert log.eval_steps == [10]
    assert log.val_loss == [3.4]
    assert log.val_perplexity == [30.0]
    assert log.expert_steps == [10, 20]
    assert log.expert_token_counts == [[[5, 3], [2, 6]], [[4, 4], [3, 5]]]


def test_skips_blank_lines_between_records(tmp_path):
    log_path = tmp_path / "train.jsonl"
    log_path.write_text(f"\n{json.dumps(_STEP_RECORD)}\n\n{json.dumps(_EVAL_RECORD)}\n\n")

    log = load_training_log(log_path)

    assert log.steps == [10]
    assert log.eval_steps == [10]


def test_step_record_without_expert_token_counts_is_not_added_to_expert_series(tmp_path):
    log_path = tmp_path / "train.jsonl"
    record = {k: v for k, v in _STEP_RECORD.items() if k != "expert_token_counts"}
    _write_jsonl(log_path, [record])

    log = load_training_log(log_path)

    assert log.steps == [10]
    assert log.expert_steps == []
    assert log.expert_token_counts == []


# -- Both Scene subclasses fail before any Manim rendering (Requirement 8.3) --


def test_loss_curve_scene_missing_log_raises_before_rendering(tmp_path):
    scene = LossCurveScene(log_path=tmp_path / "missing.jsonl")
    with pytest.raises(FileNotFoundError):
        scene.construct()
    assert len(scene.mobjects) == 0  # nothing was ever added/drawn


def test_expert_utilization_scene_missing_log_raises_before_rendering(tmp_path):
    scene = ExpertUtilizationScene(log_path=tmp_path / "missing.jsonl")
    with pytest.raises(FileNotFoundError):
        scene.construct()
    assert len(scene.mobjects) == 0


def test_expert_utilization_scene_no_expert_counts_raises_before_rendering(tmp_path):
    log_path = tmp_path / "train.jsonl"
    record = {k: v for k, v in _STEP_RECORD.items() if k != "expert_token_counts"}
    _write_jsonl(log_path, [record])

    scene = ExpertUtilizationScene(log_path=log_path)
    with pytest.raises(ValueError, match="expert_token_counts"):
        scene.construct()
    assert len(scene.mobjects) == 0
