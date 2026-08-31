"""Тесты конвейера вливания фидбэка (``scripts/ingest_feedback.py``).

Проверяют валидацию, дедупликацию по content-hash, идемпотентность копирования
и код возврата. Предсказание состояния моделью подменяется, чтобы тесты не
зависели от наличия весов.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from scripts import ingest_feedback as mod
from src.config.classes import MODEL_CLASSES
from src.data.utils import compute_content_hash

FAMILY = MODEL_CLASSES[0]


def _write_photo(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), color=color).save(path)


def _make_feedback(
    feedback_dir: Path,
    family: str,
    name: str,
    *,
    color: tuple[int, int, int] = (10, 20, 30),
    manifest: bool = True,
    ts: str | None = "2026-08-31T12:00:00",
    user_state: str | None = None,
) -> Path:
    """Создать фото фидбэка и (опционально) манифест рядом с ним."""
    photo = feedback_dir / family / f"{name}.jpg"
    _write_photo(photo, color)
    if manifest:
        payload: dict = {"user_family": family, "origin": "user"}
        if ts is not None:
            payload["ts"] = ts
        if user_state is not None:
            payload["user_state"] = user_state
        photo.with_suffix(".json").write_text(json.dumps(payload), encoding="utf-8")
    return photo


@pytest.fixture()
def dirs(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Тройка каталогов: фидбэк, processed, dirty_clean."""
    feedback = tmp_path / "feedback"
    processed = tmp_path / "processed"
    dirty_clean = tmp_path / "dirty_clean"
    feedback.mkdir()
    processed.mkdir()
    dirty_clean.mkdir()
    return feedback, processed, dirty_clean


def _run(dirs: tuple[Path, Path, Path], **kwargs) -> mod.IngestStats:
    feedback, processed, dirty_clean = dirs
    return mod.ingest_feedback(
        feedback_dir=feedback,
        processed_dir=processed,
        dirty_clean_dir=dirty_clean,
        image_size=64,
        **kwargs,
    )


class TestContentHash:
    def test_matches_hashlib(self, tmp_path: Path) -> None:
        path = tmp_path / "a.bin"
        path.write_bytes(b"tractor")
        assert compute_content_hash(path) == hashlib.md5(b"tractor").hexdigest()

    def test_same_bytes_same_hash(self, tmp_path: Path) -> None:
        a, b = tmp_path / "a.jpg", tmp_path / "b.jpg"
        _write_photo(a, (1, 2, 3))
        b.write_bytes(a.read_bytes())
        assert compute_content_hash(a) == compute_content_hash(b)


class TestDryRun:
    def test_counts_valid_and_errors(self, dirs: tuple[Path, Path, Path]) -> None:
        feedback = dirs[0]
        _make_feedback(feedback, FAMILY, "ok1", color=(10, 10, 10), user_state="dirty")
        _make_feedback(feedback, FAMILY, "ok2", color=(20, 20, 20))
        _make_feedback(feedback, FAMILY, "no_manifest", color=(30, 30, 30), manifest=False)

        stats = _run(dirs)

        assert stats.found == 3
        assert stats.valid == 2
        assert stats.errors == 1
        assert stats.duplicates == 0
        assert stats.copied == 0

    def test_nothing_copied_in_dry_run(self, dirs: tuple[Path, Path, Path]) -> None:
        feedback, _, dirty_clean = dirs
        _make_feedback(feedback, FAMILY, "ok", user_state="clean")
        _run(dirs)
        assert not list(dirty_clean.rglob("*.jpg"))


class TestValidation:
    def test_unknown_family_is_error(self, dirs: tuple[Path, Path, Path]) -> None:
        _make_feedback(dirs[0], "spaceship", "x", user_state="dirty")
        stats = _run(dirs)
        assert stats.errors == 1
        assert stats.valid == 0

    def test_missing_ts_is_error(self, dirs: tuple[Path, Path, Path]) -> None:
        _make_feedback(dirs[0], FAMILY, "x", ts=None)
        stats = _run(dirs)
        assert stats.errors == 1

    def test_non_image_is_error(self, dirs: tuple[Path, Path, Path]) -> None:
        (dirs[0] / FAMILY).mkdir(parents=True)
        (dirs[0] / FAMILY / "notes.txt").write_text("nope", encoding="utf-8")
        stats = _run(dirs)
        assert stats.found == 1
        assert stats.errors == 1


class TestDedup:
    def test_duplicate_of_existing_dataset_file(self, dirs: tuple[Path, Path, Path]) -> None:
        feedback, _, dirty_clean = dirs
        photo = _make_feedback(feedback, FAMILY, "dup", user_state="clean")
        existing = dirty_clean / "train" / FAMILY / "clean" / "old.jpg"
        existing.parent.mkdir(parents=True)
        existing.write_bytes(photo.read_bytes())

        stats = _run(dirs, apply=True)

        assert stats.valid == 1
        assert stats.duplicates == 1
        assert stats.copied == 0

    def test_rerun_is_idempotent(self, dirs: tuple[Path, Path, Path]) -> None:
        _make_feedback(dirs[0], FAMILY, "once", user_state="dirty")

        first = _run(dirs, apply=True)
        second = _run(dirs, apply=True)

        assert first.copied == 1
        assert second.copied == 0
        assert second.duplicates == 1


class TestApply:
    def test_copies_with_feedback_prefix(self, dirs: tuple[Path, Path, Path]) -> None:
        feedback, _, dirty_clean = dirs
        photo = _make_feedback(feedback, FAMILY, "img", user_state="dirty")

        stats = _run(dirs, apply=True)

        dest_dir = dirty_clean / "train" / FAMILY / "dirty"
        assert (dest_dir / "feedback_img.jpg").is_file()
        assert (dest_dir / "feedback_img.json").is_file()
        assert photo.is_file()  # копируем, а не перемещаем
        assert stats.copied == 1

    def test_state_predicted_when_manifest_has_none(
        self, dirs: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        feedback, _, dirty_clean = dirs
        _make_feedback(feedback, FAMILY, "img", user_state=None)
        monkeypatch.setattr(
            mod, "_load_state_predictor", lambda image_size, checkpoint: (lambda photo: "dirty")
        )

        stats = _run(dirs, apply=True)

        assert (dirty_clean / "train" / FAMILY / "dirty" / "feedback_img.jpg").is_file()
        assert stats.by_state_source["прогноз"] == 1
        assert stats.copied == 1


class TestReturnCode:
    def _argv(self, dirs: tuple[Path, Path, Path], *extra: str) -> list[str]:
        feedback, processed, dirty_clean = dirs
        return [
            "--feedback-dir",
            str(feedback),
            "--processed-dir",
            str(processed),
            "--dirty-clean-dir",
            str(dirty_clean),
            *extra,
        ]

    def test_zero_when_no_errors(self, dirs: tuple[Path, Path, Path]) -> None:
        _make_feedback(dirs[0], FAMILY, "ok", user_state="clean")
        assert mod.main(self._argv(dirs)) == 0

    def test_one_when_validation_errors(self, dirs: tuple[Path, Path, Path]) -> None:
        _make_feedback(dirs[0], FAMILY, "bad", manifest=False)
        assert mod.main(self._argv(dirs)) == 1
