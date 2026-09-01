"""Тесты генератора грязного датасета (``src/data/generate_dirty_dataset.py``).

Фокус блока Спеки 14: нейтральные аугментации (снег/мыло/блик) применяются к
ОБОИМ классам — и к clean-копии, и к dirty-вариантам. Реальное дерево ``data/``
не используется — всё строится в ``tmp_path`` на синтетических изображениях.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from src.data import generate_dirty_dataset as mod


def _synthetic_image(seed: int = 0, shape: tuple[int, int, int] = (48, 64, 3)) -> np.ndarray:
    """Случайное изображение ``float32`` ``[0, 1]``."""
    return np.random.default_rng(seed).random(shape).astype(np.float32)


class TestNeutralAugmentations:
    """Нейтральные аугментации: контракт и применение к обоим классам."""

    def test_probability_one_applies_all(self) -> None:
        """probability=1.0 -> применяются все аугментации по порядку."""
        image = _synthetic_image()
        out, applied = mod.apply_neutral_augmentations(
            np.random.default_rng(1), image, probability=1.0
        )

        assert applied == list(mod.NEUTRAL_AUGMENTATIONS)
        assert out.shape == image.shape
        assert out.dtype == np.float32
        assert 0.0 <= float(out.min()) and float(out.max()) <= 1.0

    def test_probability_zero_is_noop(self) -> None:
        """probability=0.0 -> изображение не меняется."""
        image = _synthetic_image()
        out, applied = mod.apply_neutral_augmentations(
            np.random.default_rng(2), image, probability=0.0
        )

        assert applied == []
        assert np.array_equal(out, image)

    def test_each_augmentation_smoke(self) -> None:
        """Каждая нейтральная аугментация сохраняет форму, dtype и диапазон."""
        image = _synthetic_image(3)
        for name, func in mod.NEUTRAL_AUGMENTATIONS.items():
            out = func(np.random.default_rng(7), image)
            assert out.shape == image.shape, name
            assert out.dtype == np.float32, name
            assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0, name


class TestNeutralAugmentationsBothClasses:
    """Нейтральные аугментации вызываются и для clean, и для dirty."""

    def test_dirty_pipeline_calls_neutral(self, monkeypatch) -> None:
        """generate_dirty_image прогоняет dirty-вариант через нейтральные аугментации."""
        calls: list[str] = []
        real = mod.apply_neutral_augmentations

        def spy(rng, image, *args, **kwargs):
            calls.append("dirty")
            return real(rng, image, *args, **kwargs)

        monkeypatch.setattr(mod, "apply_neutral_augmentations", spy)

        mod.generate_dirty_image(np.random.default_rng(0), _synthetic_image())
        assert calls == ["dirty"]

    def test_process_tree_calls_neutral_for_clean_and_dirty(self, tmp_path, monkeypatch) -> None:
        """process_tree применяет нейтральные аугментации и к clean-копии, и к dirty."""
        clean_root = tmp_path / "clean"
        src_dir = clean_root / "train" / "chtz"
        src_dir.mkdir(parents=True)
        Image.new("RGB", (40, 40), color=(100, 120, 140)).save(src_dir / "img.jpg")

        calls: list[str] = []
        real = mod.apply_neutral_augmentations

        def spy(rng, image, *args, **kwargs):
            calls.append("call")
            return real(rng, image, *args, **kwargs)

        monkeypatch.setattr(mod, "apply_neutral_augmentations", spy)

        counts = mod.process_tree(
            clean_root=clean_root,
            output_root=tmp_path / "out",
            dirty_per_clean=2,
            seed=0,
            min_effects=2,
            max_effects=3,
        )

        # 1 clean-копия + 2 dirty-варианта -> 3 вызова нейтральных аугментаций.
        assert counts == {"clean": 1, "dirty": 2}
        assert len(calls) == 3
        assert (tmp_path / "out" / "train" / "chtz" / "clean" / "img.jpg").exists()
        assert (tmp_path / "out" / "train" / "chtz" / "dirty" / "img_dirty0.jpg").exists()


class TestDirtIntensity:
    """Порог видимости синтетической грязи."""

    def test_floor_dirt_mask_removes_barely_visible(self) -> None:
        """_floor_dirt_mask: ненулевые значения поднимаются до MIN_DIRT_INTENSITY."""
        mask = np.array([0.0, 5e-4, 0.02, 0.1, mod.MIN_DIRT_INTENSITY, 0.5, 1.0], dtype=np.float32)
        floored = mod._floor_dirt_mask(mask)

        assert floored[0] == 0.0
        assert floored[1] == 0.0  # ниже порога 1e-3 -> чисто
        nonzero = floored[floored > 0.0]
        assert float(nonzero.min()) >= mod.MIN_DIRT_INTENSITY
        assert float(floored.max()) == 1.0

    def test_mud_overlay_has_no_faint_dirt_band(self) -> None:
        """apply_mud_overlay не оставляет полосы грязи слабее порога видимости.

        Проверяется на входе с сильным сигналом (нижняя половина кадра, где
        вертикальный вес близок к 1): доля едва различимых изменений мала.
        """
        image = np.full((96, 96, 3), 0.5, dtype=np.float32)
        faint_fractions = []
        for seed in range(8):
            out = mod.apply_mud_overlay(np.random.default_rng(seed), image)
            lower = np.abs(out - image).max(axis=2)[48:]
            changed = lower[lower > 1e-4]
            if changed.size:
                faint_fractions.append(float((changed < 0.03).mean()))
        assert faint_fractions
        assert float(np.mean(faint_fractions)) < 0.25

    def test_min_dirt_intensity_raised(self) -> None:
        """Константа порога поднята до аудит-уровня (>= 0.15)."""
        assert mod.MIN_DIRT_INTENSITY >= 0.15
