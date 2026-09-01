"""Тесты проверки целостности изображений (``scripts/check_images.py``).

Все данные создаются во временной директории ``tmp_path`` — реальное дерево
``data/`` тесты не трогают.
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from scripts import check_images as mod


def _write_jpeg(path: Path, size: tuple[int, int] = (64, 64)) -> None:
    """Записать валидный JPEG указанного размера."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(120, 90, 60)).save(path, format="JPEG", quality=90)


def _write_truncated_jpeg(path: Path) -> None:
    """Записать JPEG и обрезать его до половины — имитация битого файла."""
    _write_jpeg(path, size=(256, 256))
    data = path.read_bytes()
    path.write_bytes(data[: len(data) // 2])


def test_empty_dir_exit_zero(tmp_path: Path, capsys) -> None:
    """Пустая директория: битых нет, код возврата 0."""
    exit_code = mod.main(["--root", str(tmp_path)])
    report = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert report["checked"] == 0
    assert report["broken"] == []


def test_all_valid_exit_zero(tmp_path: Path, capsys) -> None:
    """Только валидные файлы: код возврата 0."""
    _write_jpeg(tmp_path / "a.jpg")
    _write_jpeg(tmp_path / "sub" / "b.jpeg")
    _write_jpeg(tmp_path / "sub" / "c.png")

    exit_code = mod.main(["--root", str(tmp_path)])
    report = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert report["checked"] == 3
    assert report["broken_count"] == 0


def test_one_broken_detected(tmp_path: Path, capsys) -> None:
    """Один битый файл среди валидных: найден ровно один, код возврата 1."""
    _write_jpeg(tmp_path / "ok1.jpg")
    _write_jpeg(tmp_path / "nested" / "ok2.jpg")
    _write_truncated_jpeg(tmp_path / "nested" / "broken.jpg")

    exit_code = mod.main(["--root", str(tmp_path), "--workers", "2"])
    report = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert report["broken_count"] == 1
    assert report["broken"][0]["path"].endswith("nested/broken.jpg")
    assert report["broken"][0]["error"]


def test_missing_root_exit_one(tmp_path: Path, capsys) -> None:
    """Несуществующий корень: код возврата 1."""
    exit_code = mod.main(["--root", str(tmp_path / "nope")])
    report = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert "error" in report
