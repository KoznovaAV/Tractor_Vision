"""Тесты реестра моделей и разбора раздела ``models`` конфигурации."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config.config_loader import ModelConfig, _parse_config, load_config
from src.models.registry import ModelEntry, build_registry, get_model, get_model_meta


class TestModelsConfig:
    """Тесты разбора раздела ``models`` в :class:`AppConfig`."""

    def test_machine_entry_present(self) -> None:
        """В конфиге есть запись ``machine`` с ожидаемыми полями."""
        config = load_config()
        assert "machine" in config.models
        entry = config.models["machine"]
        assert isinstance(entry, ModelConfig)
        assert entry.name == "machine"
        assert entry.type == "multi_task"
        assert entry.tasks == ("family", "state")
        assert isinstance(entry.checkpoint, Path)


_MINIMAL_RAW: dict = {
    "image_size": 384,
    "num_model_classes": 4,
    "num_state_classes": 2,
    "data": {
        "processed_dir": "data/processed",
        "dirty_clean_dir": "data/dirty_clean",
        "collected_dir": "data/collected",
    },
    "weights": {"dir": "weights", "multi_task": "weights/multi_task_best.ckpt"},
    "api": {
        "max_file_size_mb": 10,
        "max_batch_size": 16,
        "allowed_extensions": [".jpg"],
        "version": "1.0.0",
        "confidence_threshold": 0.6,
        "state_dirty_threshold": 0.6,
        "feedback_dir": "data/feedback",
    },
}


class TestModelVersion:
    """Тесты поля ``version`` записи модели."""

    def test_version_from_config(self) -> None:
        """``machine`` в ``config.yaml`` несёт метку версии."""
        assert load_config().models["machine"].version == "v2-finetune"

    def test_version_absent_is_none(self) -> None:
        """Без поля ``version`` в конфиге значение — ``None``."""
        raw = {
            **_MINIMAL_RAW,
            "models": {"m": {"checkpoint": "w.ckpt", "type": "multi_task", "tasks": ["family"]}},
        }
        assert _parse_config(raw).models["m"].version is None

    def test_registry_propagates_version(self) -> None:
        """``build_registry`` переносит ``version`` в :class:`ModelEntry`."""
        registry = build_registry(load_config())
        assert registry["machine"].version == "v2-finetune"


class TestGetModelMeta:
    """Тесты :func:`get_model_meta`."""

    def _machine_entry(self) -> ModelEntry:
        return build_registry(load_config())["machine"]

    def test_meta_shape_and_sha(self) -> None:
        """Возвращает ``(model, version, sha)``; sha — 12 hex-символов."""
        entry = self._machine_entry()
        try:
            model, version, checkpoint_sha = get_model_meta(entry)
        except FileNotFoundError:
            pytest.skip("Чекпоинт multi-task недоступен в этом окружении")
        assert model is get_model(entry)
        assert version == "v2-finetune"
        assert len(checkpoint_sha) == 12
        assert all(c in "0123456789abcdef" for c in checkpoint_sha)

    def test_sha_stable_between_calls(self) -> None:
        """Хеш чекпоинта стабилен между двумя вызовами."""
        entry = self._machine_entry()
        try:
            first = get_model_meta(entry)[2]
        except FileNotFoundError:
            pytest.skip("Чекпоинт multi-task недоступен в этом окружении")
        assert get_model_meta(entry)[2] == first


class TestBuildRegistry:
    """Тесты :func:`build_registry`."""

    def test_registry_mirrors_config(self) -> None:
        """Реестр содержит те же ключи, что и раздел ``models``."""
        config = load_config()
        registry = build_registry(config)
        assert set(registry) == set(config.models)
        assert isinstance(registry["machine"], ModelEntry)
        assert registry["machine"].type == "multi_task"


class TestGetModel:
    """Тесты :func:`get_model`."""

    def test_unsupported_type_raises(self) -> None:
        """Неизвестный тип загрузчика — :class:`ValueError`."""
        entry = ModelEntry(name="x", checkpoint=Path("nope.ckpt"), type="unknown", tasks=())
        with pytest.raises(ValueError):
            get_model(entry)

    def test_load_is_cached(self) -> None:
        """Повторный вызов с той же записью возвращает тот же объект."""
        entry = build_registry(load_config())["machine"]
        try:
            first = get_model(entry)
        except FileNotFoundError:
            pytest.skip("Чекпоинт multi-task недоступен в этом окружении")
        assert get_model(entry) is first
