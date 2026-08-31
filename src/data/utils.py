"""Общие утилиты файловых конвейеров данных.

Функции, переиспользуемые скриптами подготовки данных (``prepare_dataset``,
``split_real_dirty``, ``ingest_feedback``). Сейчас здесь только вычисление
content-hash файла: MD5 его содержимого используется для дедупликации одного и
того же изображения, пришедшего из разных источников, чтобы оно не попало
одновременно в разные сплиты.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

_CHUNK_SIZE: int = 65536


def compute_content_hash(path: str | Path) -> str:
    """Вычислить MD5 содержимого файла.

    Хеш считается по байтам содержимого, а не по имени или пути, поэтому одна и
    та же картинка из двух источников даёт одинаковый хеш.

    Args:
        path: Путь к файлу.

    Returns:
        Шестнадцатеричный дайджест MD5.
    """
    hasher = hashlib.md5()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_SIZE), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


__all__ = ["compute_content_hash"]
