"""Смоук-тест error analysis головы состояния (``scripts/error_analysis_state.py``).

Инференс подменяется, чтобы тест не зависел от наличия весов: фейковый
``predict_image`` берёт истинное состояние из имени файла и переворачивает его
для файлов из заданного множества, изображая промах головы состояния.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from scripts import error_analysis_state as mod
from src.config.classes import MODEL_CLASSES, STATE_CLASSES, state_to_idx

FAMILY = MODEL_CLASSES[0]


def _img(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color=(90, 90, 90)).save(path)


def _fake_predict(flip_names: set[str]):
    """Фейковый ``predict_image``: (idx класса, conf, idx состояния)."""

    def predict(model, image_path, transform):
        name = Path(image_path).name
        true_state = "dirty" if "dirty" in name else "clean"
        pred = STATE_CLASSES[1 - state_to_idx(true_state)] if name in flip_names else true_state
        return 0, 0.87, state_to_idx(pred)

    return predict


def test_analyze_tallies_and_copies_misses(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "syn"
    _img(root / FAMILY / "clean" / "a_clean.jpg")
    _img(root / FAMILY / "dirty" / "b_dirty.jpg")
    _img(root / FAMILY / "dirty" / "c_dirty.jpg")

    samples = mod.collect_synthetic(root)
    assert len(samples) == 3

    monkeypatch.setattr(mod, "predict_image", _fake_predict({"c_dirty.jpg"}))
    out_dir = tmp_path / "out"
    report = mod.analyze(model=None, samples=samples, transform=None, out_dir=out_dir)

    node = report["sources"]["synthetic"]
    assert node["clean"]["false_dirty"] == 0
    assert node["dirty"]["total"] == 2
    assert node["dirty"]["false_clean"] == 1
    assert node["dirty"]["dirty_recall"] == 0.5

    copied = list((out_dir / "dirty_as_clean").glob("*.jpg"))
    assert [p.name for p in copied] == ["pred_clean_conf0.87_c_dirty.jpg"]
    assert report["totals"]["dirty_false_clean_rate"] == 0.5


def test_main_writes_report_and_skips_empty_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    syn = tmp_path / "syn"
    _img(syn / FAMILY / "clean" / "x_clean.jpg")
    _img(syn / FAMILY / "dirty" / "y_dirty.jpg")
    real_dirty = tmp_path / "rd"
    _img(real_dirty / FAMILY / "z_dirty.jpg")
    real_clean = tmp_path / "rc"
    real_clean.mkdir()

    monkeypatch.setattr(mod, "load_multi_task_model", lambda *a, **k: None)
    monkeypatch.setattr(mod, "get_val_transforms", lambda *a, **k: None)
    monkeypatch.setattr(mod, "resolve_working_checkpoint", lambda: Path("fake.ckpt"))
    monkeypatch.setattr(mod, "predict_image", _fake_predict(set()))

    out_dir = tmp_path / "out"
    code = mod.main(
        [
            "--out-dir",
            str(out_dir),
            "--synthetic-dir",
            str(syn),
            "--real-dirty-dir",
            str(real_dirty),
            "--real-clean-dir",
            str(real_clean),
        ]
    )

    assert code == 0
    report = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
    assert report["samples"] == 3
    assert "real_clean_probe" not in report["sources"]
    assert report["sources"]["real_dirty_val"]["dirty"]["dirty_recall"] == 1.0
    assert "пропущен" in capsys.readouterr().out
