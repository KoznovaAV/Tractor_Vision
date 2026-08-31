"""Тесты реестра моделей и разбора раздела ``models`` конфигурации."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config.config_loader import ModelConfig, load_config
from src.models.registry import ModelEntry, build_registry, get_model


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
