#!/usr/bin/env python3
"""Сбор и полуавторазметка изображений тракторов для расширения датасета.

Пайплайн для каждого из 4 классов модели трактора:

1. **Поиск** изображений через подключаемый :class:`SourceAdapter`. Дефолтный
   адаптер — DuckDuckGo (``ddgs``), не требующий ключа. Позже можно добавить
   Bing-адаптер с ключом из переменной окружения, реализовав тот же интерфейс.
2. **Скачивание** с ограничением частоты запросов (rate limiting) и таймаутами.
3. **Дедупликация** через perceptual hashing (``imagehash.phash``) с порогом по
   расстоянию Хэмминга — отсеиваются как точные, так и почти-дубликаты.
4. **Отбраковка** фото без трактора и **полуавторазметка** через CLIP zero-shot
   (``open_clip``, ViT-B-32, веса laion2b). Изображения с уверенностью ниже
   порога попадают в папку ``to_review`` вместо автоматической раскладки.
5. **Сохранение** в структуру датасета ``<split>/<model_class>/`` с сохранением
   исходного URL каждого изображения в сопутствующем JSON-манифесте.

Пример::

    python -m scripts.collect_tractors \\
        --output-dir data/processed \\
        --split train \\
        --per-class 500 \\
        --clip-threshold 0.7

Тяжёлые зависимости (``torch``/``open_clip``) импортируются лениво внутри
CLIP-разметчика, поэтому этап поиска/дедупликации можно запускать и без них.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from PIL import Image, UnidentifiedImageError

from src.config.classes import MODEL_CLASSES

# Ключевые слова для каждого класса. Первый элемент — «якорь», остальные
# расширяют покрытие. Класс mtz_82 включает исторически слитый mtz_1221.
CLASS_KEYWORDS: dict[str, tuple[str, ...]] = {
    "chtz_b10m": ("трактор ЧТЗ Б10М", "бульдозер Б10М", "ChTZ B10M bulldozer"),
    "johndeere": ("трактор John Deere", "John Deere tractor", "трактор Джон Дир"),
    "kirovets_k744": ("трактор Кировец К-744", "Kirovets K-744", "трактор Кировец"),
    "mtz_82": (
        "трактор МТЗ-82 Беларус",
        "трактор Беларус МТЗ",
        "MTZ-82 Belarus tractor",
        "трактор МТЗ-1221",
    ),
}

# Текстовые промпты CLIP для zero-shot проверки «есть ли трактор на фото».
# Позитивные промпты сопоставляются с негативными (не-трактор) — если максимум
# приходится на негативный промпт, изображение отбраковывается.
CLIP_POSITIVE_PROMPTS: tuple[str, ...] = (
    "a photo of a tractor",
    "a photo of a farm tractor",
    "a photo of a bulldozer",
    "a photo of heavy agricultural machinery",
)
CLIP_NEGATIVE_PROMPTS: tuple[str, ...] = (
    "a photo of a car",
    "a photo of a truck",
    "a photo of a person",
    "a photo of a landscape without vehicles",
    "a diagram or drawing",
    "a photo of an interior room",
)

IMAGE_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".webp"})
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)


@dataclass
class ImageCandidate:
    """Кандидат-изображение на этапе сбора.

    Attributes:
        url: Прямой URL изображения.
        source_query: Поисковый запрос, по которому найдено изображение.
        title: Заголовок/подпись из выдачи (может быть пустым).
        data: Сырые байты изображения (заполняются после скачивания).
        phash: Perceptual hash (заполняется после декодирования).
    """

    url: str
    source_query: str
    title: str = ""
    data: bytes | None = None
    phash: Any | None = None


# ---------------------------------------------------------------------------
# Источники изображений (подключаемые адаптеры).
# ---------------------------------------------------------------------------
class SourceAdapter(ABC):
    """Абстрактный источник изображений.

    Реализации инкапсулируют конкретный бэкенд поиска (DuckDuckGo, Bing и т.д.).
    Чтобы добавить новый источник, достаточно реализовать :meth:`search`.
    """

    name: str = "abstract"

    @abstractmethod
    def search(self, query: str, max_results: int) -> list[ImageCandidate]:
        """Найти изображения по запросу.

        Args:
            query: Поисковый запрос.
            max_results: Максимальное число результатов.

        Returns:
            Список кандидатов (без скачанных байтов).
        """
        raise NotImplementedError


class DuckDuckGoAdapter(SourceAdapter):
    """Источник изображений на базе DuckDuckGo (пакет ``ddgs``).

    Не требует API-ключа. Учитывает вариативность имён ключей в выдаче между
    версиями ``ddgs`` (``image`` / ``url`` / ``thumbnail``).
    """

    name = "duckduckgo"

    def __init__(self, region: str = "wt-wt", safesearch: str = "off") -> None:
        """Инициализировать адаптер.

        Args:
            region: Регион выдачи DuckDuckGo (``wt-wt`` — без региона).
            safesearch: Режим безопасного поиска (``off``/``moderate``/``on``).
        """
        self.region = region
        self.safesearch = safesearch

    def search(self, query: str, max_results: int) -> list[ImageCandidate]:
        """См. :meth:`SourceAdapter.search`."""
        # Ленивый импорт: пакет нужен только этому адаптеру.
        from ddgs import DDGS

        candidates: list[ImageCandidate] = []
        try:
            results = DDGS().images(
                query=query,
                region=self.region,
                safesearch=self.safesearch,
                max_results=max_results,
            )
        except Exception as exc:  # noqa: BLE001 — сеть непредсказуема, логируем
            print(f"    [ddg] ошибка поиска '{query}': {exc}")
            return candidates

        for item in results:
            url = item.get("image") or item.get("url") or item.get("thumbnail")
            if not url:
                continue
            candidates.append(
                ImageCandidate(
                    url=url,
                    source_query=query,
                    title=str(item.get("title", "")),
                )
            )
        return candidates


# ---------------------------------------------------------------------------
# Дедупликация через perceptual hashing.
# ---------------------------------------------------------------------------
class PerceptualDeduplicator:
    """Отсев точных и почти-дубликатов через ``imagehash.phash``.

    Хранит хеши уже принятых изображений и отклоняет новые, чьё расстояние
    Хэмминга до любого принятого не превышает порога.
    """

    def __init__(self, hamming_threshold: int = 6) -> None:
        """Инициализировать дедупликатор.

        Args:
            hamming_threshold: Максимальное расстояние Хэмминга, при котором
                изображения считаются дубликатами. 0 — только точные совпадения,
                типичное значение для почти-дубликатов — 4–8.
        """
        self.hamming_threshold = hamming_threshold
        self._hashes: list[Any] = []

    def compute_hash(self, image: Image.Image) -> Any:
        """Вычислить perceptual hash изображения.

        Args:
            image: Изображение PIL.

        Returns:
            Объект ``imagehash.ImageHash``.
        """
        import imagehash

        return imagehash.phash(image)

    def is_duplicate(self, image_hash: Any) -> bool:
        """Проверить, является ли хеш дубликатом уже принятого.

        Args:
            image_hash: Хеш проверяемого изображения.

        Returns:
            ``True``, если найден достаточно близкий принятый хеш.
        """
        return any(
            (image_hash - existing) <= self.hamming_threshold
            for existing in self._hashes
        )

    def add(self, image_hash: Any) -> None:
        """Зарегистрировать хеш как принятый.

        Args:
            image_hash: Хеш принятого изображения.
        """
        self._hashes.append(image_hash)


# ---------------------------------------------------------------------------
# CLIP zero-shot разметка / отбраковка.
# ---------------------------------------------------------------------------
@dataclass
class ClipVerdict:
    """Результат CLIP-проверки изображения.

    Attributes:
        is_tractor: Прошло ли изображение проверку «на фото трактор».
        confidence: Уверенность (доля вероятности лучшего позитивного промпта).
        best_prompt: Промпт с максимальной вероятностью.
    """

    is_tractor: bool
    confidence: float
    best_prompt: str


class ClipLabeler:
    """Zero-shot проверка наличия трактора через CLIP (``open_clip``).

    Модель и веса загружаются лениво при первом использовании, чтобы этапы,
    не требующие CLIP, работали без импорта torch.
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: str | None = None,
    ) -> None:
        """Инициализировать разметчик (без загрузки модели).

        Args:
            model_name: Имя архитектуры open_clip.
            pretrained: Тег предобученных весов (открытые веса laion2b).
            device: Устройство (``cuda``/``cpu``). ``None`` — автоопределение.
        """
        self.model_name = model_name
        self.pretrained = pretrained
        self._device = device
        self._model: Any | None = None
        self._preprocess: Any | None = None
        self._text_features: Any | None = None
        self._prompts: tuple[str, ...] = CLIP_POSITIVE_PROMPTS + CLIP_NEGATIVE_PROMPTS
        self._num_positive = len(CLIP_POSITIVE_PROMPTS)

    def _ensure_loaded(self) -> None:
        """Лениво загрузить модель, препроцессор и текстовые эмбеддинги."""
        if self._model is not None:
            return

        import open_clip
        import torch

        if self._device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"

        model, _, preprocess = open_clip.create_model_and_transforms(
            self.model_name, pretrained=self.pretrained
        )
        model = model.to(self._device).eval()
        tokenizer = open_clip.get_tokenizer(self.model_name)

        with torch.no_grad():
            tokens = tokenizer(list(self._prompts)).to(self._device)
            text_features = model.encode_text(tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self._model = model
        self._preprocess = preprocess
        self._text_features = text_features

    def classify(self, image: Image.Image) -> ClipVerdict:
        """Оценить, изображён ли на фото трактор.

        Args:
            image: Изображение PIL (RGB).

        Returns:
            Вердикт CLIP с уверенностью и лучшим промптом.
        """
        self._ensure_loaded()

        import torch

        assert self._model is not None
        assert self._preprocess is not None
        assert self._text_features is not None

        image_input = self._preprocess(image).unsqueeze(0).to(self._device)
        with torch.no_grad():
            image_features = self._model.encode_image(image_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            logits = (100.0 * image_features @ self._text_features.T).softmax(dim=-1)

        probs = logits.squeeze(0).cpu().tolist()
        best_idx = int(max(range(len(probs)), key=lambda i: probs[i]))
        best_prompt = self._prompts[best_idx]

        # Уверенность «трактор» = суммарная вероятность позитивных промптов.
        positive_confidence = float(sum(probs[: self._num_positive]))
        is_tractor = best_idx < self._num_positive

        return ClipVerdict(
            is_tractor=is_tractor,
            confidence=positive_confidence,
            best_prompt=best_prompt,
        )


# ---------------------------------------------------------------------------
# Скачивание с rate limiting.
# ---------------------------------------------------------------------------
class RateLimitedDownloader:
    """Скачиватель изображений с ограничением частоты и валидацией.

    Attributes:
        min_interval: Минимальный интервал между запросами в секундах.
        timeout: Таймаут одного запроса в секундах.
    """

    def __init__(
        self,
        min_interval: float = 1.0,
        timeout: float = 15.0,
        user_agent: str = DEFAULT_USER_AGENT,
    ) -> None:
        """Инициализировать скачиватель.

        Args:
            min_interval: Минимальная пауза между сетевыми запросами.
            timeout: Таймаут запроса.
            user_agent: Заголовок User-Agent.
        """
        self.min_interval = min_interval
        self.timeout = timeout
        self.user_agent = user_agent
        self._last_request_ts = 0.0

    def _throttle(self) -> None:
        """Выдержать паузу для соблюдения ограничения частоты."""
        elapsed = time.monotonic() - self._last_request_ts
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_request_ts = time.monotonic()

    def download(self, url: str) -> bytes | None:
        """Скачать изображение по URL.

        Args:
            url: Прямой URL изображения.

        Returns:
            Сырые байты либо ``None`` при ошибке.
        """
        self._throttle()
        try:
            request = Request(url, headers={"User-Agent": self.user_agent})
            with urlopen(request, timeout=self.timeout) as response:
                return response.read()
        except Exception as exc:  # noqa: BLE001 — сеть непредсказуема
            print(f"    [dl] не удалось скачать {url[:80]}: {exc}")
            return None


# ---------------------------------------------------------------------------
# Оркестрация сбора.
# ---------------------------------------------------------------------------
@dataclass
class CollectionStats:
    """Статистика сбора по одному классу.

    Attributes:
        found: Найдено кандидатов в выдаче.
        downloaded: Успешно скачано и декодировано.
        deduped: Отброшено как дубликаты.
        rejected_no_tractor: Отбраковано CLIP (не трактор).
        to_review: Отправлено в to_review (низкая уверенность CLIP).
        accepted: Принято в целевой класс.
    """

    found: int = 0
    downloaded: int = 0
    deduped: int = 0
    rejected_no_tractor: int = 0
    to_review: int = 0
    accepted: int = 0


def _decode_image(data: bytes) -> Image.Image | None:
    """Декодировать байты в RGB-изображение PIL.

    Args:
        data: Сырые байты изображения.

    Returns:
        Изображение PIL в режиме RGB либо ``None`` при ошибке декодирования.
    """
    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
        image.load()
        return image
    except (UnidentifiedImageError, OSError, ValueError):
        return None


def _save_image(
    image: Image.Image,
    data: bytes,
    dest_dir: Path,
    url: str,
) -> Path:
    """Сохранить изображение в целевую директорию с именем по хешу URL.

    Имя файла детерминировано (md5 от URL), что упрощает идемпотентность и
    предотвращает коллизии имён между источниками.

    Args:
        image: Декодированное изображение (для определения формата).
        data: Исходные байты (сохраняются как есть, без пережатия).
        dest_dir: Целевая директория.
        url: Исходный URL (для генерации имени).

    Returns:
        Путь сохранённого файла.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.md5(url.encode("utf-8")).hexdigest()[:16]
    ext = ".jpg" if image.format in (None, "JPEG") else f".{image.format.lower()}"
    if ext not in IMAGE_EXTENSIONS:
        ext = ".jpg"
    path = dest_dir / f"{digest}{ext}"
    path.write_bytes(data)
    return path


def collect_for_class(
    model_class: str,
    output_root: Path,
    split: str,
    per_class: int,
    source: SourceAdapter,
    downloader: RateLimitedDownloader,
    deduplicator: PerceptualDeduplicator,
    labeler: ClipLabeler | None,
    clip_threshold: float,
    manifest: list[dict[str, Any]],
) -> CollectionStats:
    """Собрать изображения для одного класса модели трактора.

    Args:
        model_class: Канонический класс (например, ``"mtz_82"``).
        output_root: Корень дерева датасета (например, ``data/processed``).
        split: Имя сплита (``train``/``val``/``test``).
        per_class: Целевое число принятых изображений на класс.
        source: Источник изображений.
        downloader: Скачиватель с rate limiting.
        deduplicator: Дедупликатор (общий на весь класс).
        labeler: CLIP-разметчик либо ``None`` (тогда CLIP-этап пропускается).
        clip_threshold: Порог уверенности CLIP для авто-принятия.
        manifest: Общий список записей манифеста (обновляется на месте).

    Returns:
        Статистика сбора по классу.
    """
    stats = CollectionStats()
    accepted_dir = output_root / split / model_class
    review_dir = output_root / split / "to_review" / model_class

    keywords = CLASS_KEYWORDS[model_class]
    # Запрашиваем с запасом, поскольку часть отсеется дедупом/CLIP/скачиванием.
    per_query = max(20, (per_class * 3) // len(keywords))

    print(f"\n### Класс {model_class}: цель {per_class} изображений")
    for query in keywords:
        if stats.accepted >= per_class:
            break
        print(f"  запрос: '{query}' (до {per_query})")
        candidates = source.search(query, per_query)
        stats.found += len(candidates)

        for candidate in candidates:
            if stats.accepted >= per_class:
                break

            data = downloader.download(candidate.url)
            if data is None:
                continue
            image = _decode_image(data)
            if image is None:
                continue
            stats.downloaded += 1

            image_hash = deduplicator.compute_hash(image)
            if deduplicator.is_duplicate(image_hash):
                stats.deduped += 1
                continue

            destination_dir = accepted_dir
            review_flag = False
            clip_conf = 1.0
            clip_prompt = "<clip disabled>"

            if labeler is not None:
                verdict = labeler.classify(image)
                clip_conf = verdict.confidence
                clip_prompt = verdict.best_prompt
                if not verdict.is_tractor:
                    stats.rejected_no_tractor += 1
                    continue
                if verdict.confidence < clip_threshold:
                    destination_dir = review_dir
                    review_flag = True

            deduplicator.add(image_hash)
            saved_path = _save_image(image, data, destination_dir, candidate.url)

            if review_flag:
                stats.to_review += 1
            else:
                stats.accepted += 1

            manifest.append(
                {
                    "path": str(saved_path),
                    "url": candidate.url,
                    "model_class": model_class,
                    "split": split,
                    "query": candidate.source_query,
                    "title": candidate.title,
                    "clip_confidence": round(clip_conf, 4),
                    "clip_best_prompt": clip_prompt,
                    "needs_review": review_flag,
                    "source": source.name,
                }
            )

    return stats


def _build_source(name: str) -> SourceAdapter:
    """Создать адаптер источника по имени.

    Args:
        name: Идентификатор источника (пока поддерживается ``duckduckgo``).

    Returns:
        Экземпляр :class:`SourceAdapter`.

    Raises:
        ValueError: Если источник неизвестен.
    """
    if name == "duckduckgo":
        return DuckDuckGoAdapter()
    raise ValueError(f"Неизвестный источник: {name!r}. Доступно: duckduckgo.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Разобрать аргументы командной строки.

    Args:
        argv: Список аргументов (по умолчанию ``sys.argv[1:]``).

    Returns:
        Пространство имён с аргументами.
    """
    parser = argparse.ArgumentParser(
        description="Сбор и полуавторазметка изображений тракторов.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Корень дерева датасета. По умолчанию data/processed.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Целевой сплит (train/val/test). По умолчанию train.",
    )
    parser.add_argument(
        "--per-class",
        type=int,
        default=500,
        help="Целевое число принятых изображений на класс.",
    )
    parser.add_argument(
        "--classes",
        type=str,
        nargs="*",
        default=list(MODEL_CLASSES),
        help="Подмножество классов для сбора. По умолчанию все.",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="duckduckgo",
        help="Источник изображений. По умолчанию duckduckgo.",
    )
    parser.add_argument(
        "--clip-threshold",
        type=float,
        default=0.7,
        help="Порог уверенности CLIP для авто-принятия. Ниже — в to_review.",
    )
    parser.add_argument(
        "--hamming-threshold",
        type=int,
        default=6,
        help="Порог расстояния Хэмминга для дедупликации (phash).",
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=1.0,
        help="Минимальный интервал между сетевыми запросами в секундах.",
    )
    parser.add_argument(
        "--no-clip",
        action="store_true",
        help="Отключить CLIP-этап (только поиск + дедуп + скачивание).",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Путь к JSON-манифесту. По умолчанию <output-dir>/collect_manifest.json.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Точка входа коллектора.

    Args:
        argv: Аргументы командной строки (для тестируемости).

    Returns:
        Код возврата процесса (0 — успех).
    """
    args = parse_args(argv)

    unknown = [c for c in args.classes if c not in MODEL_CLASSES]
    if unknown:
        print(f"[ошибка] Неизвестные классы: {unknown}. Доступно: {MODEL_CLASSES}")
        return 1

    source = _build_source(args.source)
    downloader = RateLimitedDownloader(min_interval=args.rate_limit)
    deduplicator = PerceptualDeduplicator(hamming_threshold=args.hamming_threshold)
    labeler = None if args.no_clip else ClipLabeler()

    manifest_path = args.manifest or (args.output_dir / "collect_manifest.json")
    manifest: list[dict[str, Any]] = []
    if manifest_path.exists():
        # Продолжаем существующий манифест, чтобы сбор был инкрементальным.
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"[warn] манифест повреждён, начинаю заново: {manifest_path}")
            manifest = []

    print(f"Источник: {source.name} | CLIP: {'off' if args.no_clip else 'on'}")
    print(f"Сплит: {args.split} | Цель на класс: {args.per_class}")

    all_stats: dict[str, CollectionStats] = {}
    for model_class in args.classes:
        stats = collect_for_class(
            model_class=model_class,
            output_root=args.output_dir,
            split=args.split,
            per_class=args.per_class,
            source=source,
            downloader=downloader,
            deduplicator=deduplicator,
            labeler=labeler,
            clip_threshold=args.clip_threshold,
            manifest=manifest,
        )
        all_stats[model_class] = stats

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 60)
    print("ИТОГ СБОРА")
    print("=" * 60)
    for model_class, stats in all_stats.items():
        print(
            f"{model_class:<16} "
            f"найдено={stats.found:<5} скачано={stats.downloaded:<5} "
            f"дубли={stats.deduped:<5} не_трактор={stats.rejected_no_tractor:<5} "
            f"review={stats.to_review:<5} принято={stats.accepted}"
        )
    print(f"\nМанифест: {manifest_path} (записей: {len(manifest)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
