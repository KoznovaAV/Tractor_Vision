#!/usr/bin/env python3
"""Генерация реалистичного «грязного» датасета из чистых изображений.

Грязь моделируется физически правдоподобно через компонуемые деградации, чтобы
голова clean/dirty обобщалась на настоящие фото, а не только на шум/blur:

* **Оверлеи грязи/брызг/пыли** — процедурные текстуры (мультиоктавный
  value-noise) с alpha-blending поверх изображения; грязь тёмная и локальная,
  пыль светлая и рассеянная, брызги — редкие капли с разбросом.
* **Дождь** — полупрозрачные наклонные штрихи + лёгкое размытие «мокрой» сцены.
* **Туман** — низкочастотная дымка, снижающая контраст и подмешивающая белёсость.
* **Частичные перекрытия** — тёмные/землистые кляксы неправильной формы,
  имитирующие налипшую грязь и загораживающие часть техники.
* **Освещение** — радиальные градиенты (пятна света/тени), гамма-сдвиги,
  цветовая температура — разные условия съёмки.

Каждая деградация — функция ``(rng, image) -> image`` над ``float32`` массивом
в диапазоне ``[0, 1]`` (RGB). Пайплайн выбирает случайное подмножество и
применяет их последовательно с рандомизированной силой.

Пример::

    python -m src.data.generate_dirty_dataset \\
        --clean-dir data/processed \\
        --output-dir data/dirty_clean \\
        --dirty-per-clean 2 \\
        --seed 42

Скрипт не хардкодит имена классов: он зеркалит структуру входного дерева
``<split>/<class>/`` в выходное ``<split>/<class>/{clean,dirty}/`` и раскладывает
оригиналы в ``clean``, а сгенерированные варианты — в ``dirty``.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

# Тип одной деградации: принимает генератор случайных чисел и изображение,
# возвращает изменённое изображение (оба — float32 [0,1], форма (H, W, 3)).
Degradation = Callable[[np.random.Generator, np.ndarray], np.ndarray]

IMAGE_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".webp"})
SPLIT_NAMES: tuple[str, ...] = ("train", "val", "test")
STATE_DIRS: tuple[str, str] = ("clean", "dirty")


# ---------------------------------------------------------------------------
# Процедурный шум (основа текстур грязи/пыли/тумана).
# ---------------------------------------------------------------------------
def _value_noise(
    rng: np.random.Generator,
    shape: tuple[int, int],
    octaves: int = 4,
    persistence: float = 0.5,
) -> np.ndarray:
    """Сгенерировать мультиоктавный value-noise в диапазоне ``[0, 1]``.

    Складывает несколько октав билинейно-интерполированного случайного шума с
    убывающей амплитудой — даёт «облачную» текстуру без жёстких пикселей.

    Args:
        rng: Генератор случайных чисел.
        shape: Итоговая форма ``(H, W)``.
        octaves: Число октав (детализация текстуры).
        persistence: Множитель амплитуды между октавами (0..1).

    Returns:
        Массив ``float32`` формы ``(H, W)`` со значениями в ``[0, 1]``.
    """
    height, width = shape
    noise = np.zeros((height, width), dtype=np.float32)
    amplitude = 1.0
    total_amplitude = 0.0

    for octave in range(octaves):
        # Разрешение базового шума растёт с номером октавы.
        base = 2 ** (octave + 1)
        grid_h = max(2, base)
        grid_w = max(2, base)
        low_res = rng.random((grid_h, grid_w)).astype(np.float32)
        upscaled = cv2.resize(low_res, (width, height), interpolation=cv2.INTER_LINEAR)
        noise += amplitude * upscaled
        total_amplitude += amplitude
        amplitude *= persistence

    noise /= max(total_amplitude, 1e-6)
    # Нормируем в [0, 1] на случай накопленных отклонений.
    noise -= noise.min()
    noise /= max(float(noise.max()), 1e-6)
    return noise


# ---------------------------------------------------------------------------
# Деградации.
# ---------------------------------------------------------------------------
def apply_mud_overlay(rng: np.random.Generator, image: np.ndarray) -> np.ndarray:
    """Налипшая грязь: тёмные землистые пятна с alpha-blending.

    Маска берётся из порогованного value-noise, поэтому грязь ложится
    локальными кляксами, а не равномерной плёнкой.

    Args:
        rng: Генератор случайных чисел.
        image: Изображение ``float32`` ``[0, 1]``, форма ``(H, W, 3)``.

    Returns:
        Изображение с наложенной грязью.
    """
    height, width = image.shape[:2]
    noise = _value_noise(rng, (height, width), octaves=5)

    # Порог оставляет часть площади чистой — грязь покрывает не весь кадр.
    threshold = rng.uniform(0.40, 0.60)
    mask = np.clip((noise - threshold) / (1.0 - threshold), 0.0, 1.0)
    mask *= rng.uniform(0.4, 0.85)  # общая интенсивность

    # Вертикальный градиент веса: у верхнего края (небо/фон) грязь подавляется
    # (~0.3), у нижнего (техника, колёса, крылья) — полная сила (1.0). Это
    # смещает грязь на объект и убирает спурный признак «точки на небе».
    vertical_weight = np.linspace(0.3, 1.0, height, dtype=np.float32)
    mask *= vertical_weight[:, None]

    # Опциональный «грязевой фартук»: в ~30% случаев добавляем полосу грязи в
    # нижней четверти кадра (зона колёс/крыльев, где грязь скапливается сильнее).
    if rng.random() < 0.30:
        apron = np.zeros(height, dtype=np.float32)
        apron_start = int(height * 0.75)
        # Плавно нарастающая к низу полоса.
        apron[apron_start:] = np.linspace(0.0, 1.0, height - apron_start, dtype=np.float32)
        apron_strength = rng.uniform(0.35, 0.7)
        # Фартук комбинируется с шумовой маской по максимуму, чтобы низ кадра
        # был гарантированно грязным, но с сохранением текстуры.
        apron_mask = apron[:, None] * apron_strength * (0.5 + 0.5 * noise)
        mask = np.maximum(mask, apron_mask)

    # Землистый цвет грязи с вариацией оттенка.
    mud_color = np.array(
        [
            rng.uniform(0.18, 0.32),  # R
            rng.uniform(0.12, 0.24),  # G
            rng.uniform(0.06, 0.16),  # B
        ],
        dtype=np.float32,
    )
    mask_3 = mask[..., None]
    return image * (1.0 - mask_3) + mud_color[None, None, :] * mask_3


def apply_dust_overlay(rng: np.random.Generator, image: np.ndarray) -> np.ndarray:
    """Пыль: светлая рассеянная плёнка, снижающая насыщенность.

    Args:
        rng: Генератор случайных чисел.
        image: Изображение ``float32`` ``[0, 1]``.

    Returns:
        Запылённое изображение.
    """
    height, width = image.shape[:2]
    noise = _value_noise(rng, (height, width), octaves=3)
    alpha = noise * rng.uniform(0.12, 0.30)

    dust_color = np.array(
        [
            rng.uniform(0.70, 0.85),
            rng.uniform(0.66, 0.80),
            rng.uniform(0.58, 0.72),
        ],
        dtype=np.float32,
    )
    alpha_3 = alpha[..., None]
    return image * (1.0 - alpha_3) + dust_color[None, None, :] * alpha_3


def apply_splatter(rng: np.random.Generator, image: np.ndarray) -> np.ndarray:
    """Брызги грязи: редкие тёмные капли разного радиуса.

    Args:
        rng: Генератор случайных чисел.
        image: Изображение ``float32`` ``[0, 1]``.

    Returns:
        Изображение с брызгами.
    """
    height, width = image.shape[:2]
    result = image.copy()
    num_drops = rng.integers(15, 60)

    for _ in range(num_drops):
        center_x = int(rng.integers(0, width))
        center_y = int(rng.integers(0, height))
        radius = int(rng.integers(1, max(2, min(height, width) // 40)))
        color = np.array(
            [
                rng.uniform(0.10, 0.25),
                rng.uniform(0.07, 0.18),
                rng.uniform(0.04, 0.12),
            ],
            dtype=np.float32,
        )
        alpha = float(rng.uniform(0.5, 0.9))

        drop: np.ndarray = np.zeros((height, width), dtype=np.float32)
        cv2.circle(drop, (center_x, center_y), radius, 1.0, thickness=-1)
        drop = cv2.GaussianBlur(drop, (0, 0), sigmaX=radius * 0.5 + 0.5)
        drop_3 = (drop * alpha)[..., None]
        result = result * (1.0 - drop_3) + color[None, None, :] * drop_3

    return result


def apply_rain(rng: np.random.Generator, image: np.ndarray) -> np.ndarray:
    """Дождь: наклонные полупрозрачные штрихи + лёгкое размытие сцены.

    Args:
        rng: Генератор случайных чисел.
        image: Изображение ``float32`` ``[0, 1]``.

    Returns:
        Изображение с дождём.
    """
    height, width = image.shape[:2]
    rain_layer: np.ndarray = np.zeros((height, width), dtype=np.float32)

    num_streaks = rng.integers(200, 600)
    angle = rng.uniform(-0.35, 0.35)  # наклон в радианах от вертикали
    length = rng.integers(8, max(9, height // 12))

    for _ in range(num_streaks):
        x0 = int(rng.integers(0, width))
        y0 = int(rng.integers(0, height))
        dx = int(np.sin(angle) * length)
        dy = int(np.cos(angle) * length)
        cv2.line(
            rain_layer,
            (x0, y0),
            (x0 + dx, y0 + dy),
            color=float(rng.uniform(0.5, 1.0)),
            thickness=1,
        )

    rain_layer = cv2.GaussianBlur(rain_layer, (3, 3), 0)
    rain_alpha = rain_layer[..., None] * rng.uniform(0.25, 0.5)

    # Мокрая сцена слегка размыта.
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=rng.uniform(0.5, 1.2))
    rainy = blurred * (1.0 - rain_alpha) + rain_alpha  # штрихи почти белые
    return np.asarray(np.clip(rainy, 0.0, 1.0), dtype=np.float32)


def apply_fog(rng: np.random.Generator, image: np.ndarray) -> np.ndarray:
    """Туман: низкочастотная белёсая дымка, снижающая контраст.

    Args:
        rng: Генератор случайных чисел.
        image: Изображение ``float32`` ``[0, 1]``.

    Returns:
        Изображение в дымке.
    """
    height, width = image.shape[:2]
    fog = _value_noise(rng, (height, width), octaves=2)
    # Сильное сглаживание — туман ложится крупными пятнами.
    fog = cv2.GaussianBlur(fog, (0, 0), sigmaX=min(height, width) * 0.05)
    fog = fog[..., None]

    density = rng.uniform(0.2, 0.5)
    fog_color = float(rng.uniform(0.75, 0.9))
    foggy = image * (1.0 - density * fog) + fog_color * density * fog
    return np.asarray(np.clip(foggy, 0.0, 1.0), dtype=np.float32)


def apply_occlusion(rng: np.random.Generator, image: np.ndarray) -> np.ndarray:
    """Частичное перекрытие: крупная землистая клякса неправильной формы.

    Имитирует налипший ком грязи, загораживающий часть техники.

    Args:
        rng: Генератор случайных чисел.
        image: Изображение ``float32`` ``[0, 1]``.

    Returns:
        Частично перекрытое изображение.
    """
    height, width = image.shape[:2]
    mask: np.ndarray = np.zeros((height, width), dtype=np.float32)

    # Случайный выпуклый-ish полигон вокруг случайного центра.
    center = np.array([rng.integers(0, width), rng.integers(0, height)], dtype=np.float32)
    num_points = int(rng.integers(6, 12))
    max_r = min(height, width) * rng.uniform(0.15, 0.35)
    angles = np.sort(rng.uniform(0, 2 * np.pi, size=num_points))
    radii = rng.uniform(0.4, 1.0, size=num_points) * max_r
    points = np.stack(
        [
            center[0] + np.cos(angles) * radii,
            center[1] + np.sin(angles) * radii,
        ],
        axis=1,
    ).astype(np.int32)
    cv2.fillPoly(mask, [points], 1.0)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=min(height, width) * 0.02)

    # Текстурируем кляксу шумом, чтобы она не была плоской заливкой.
    texture = _value_noise(rng, (height, width), octaves=4)
    mud = np.stack(
        [
            0.12 + 0.10 * texture,
            0.08 + 0.08 * texture,
            0.04 + 0.06 * texture,
        ],
        axis=-1,
    ).astype(np.float32)

    alpha = (mask * rng.uniform(0.7, 0.95))[..., None]
    blended = image * (1.0 - alpha) + mud * alpha
    return np.asarray(blended, dtype=np.float32)


def apply_lighting(rng: np.random.Generator, image: np.ndarray) -> np.ndarray:
    """Вариация освещения: радиальное пятно света/тени, гамма, цветовая температура.

    Args:
        rng: Генератор случайных чисел.
        image: Изображение ``float32`` ``[0, 1]``.

    Returns:
        Изображение с изменённым освещением.
    """
    height, width = image.shape[:2]

    # Радиальный градиент яркости (пятно света или тени).
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    cx = rng.uniform(0, width)
    cy = rng.uniform(0, height)
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    dist /= max(float(dist.max()), 1e-6)
    # Положительный gain — засветка в центре, отрицательный — затемнение.
    gain = rng.uniform(-0.35, 0.35)
    brightness = 1.0 + gain * (1.0 - dist)
    result = image * brightness[..., None]

    # Гамма-сдвиг.
    gamma = rng.uniform(0.75, 1.35)
    result = np.clip(result, 0.0, 1.0) ** gamma

    # Цветовая температура: тёплый/холодный сдвиг каналов.
    temp = rng.uniform(-0.08, 0.08)
    channel_scale = np.array([1.0 + temp, 1.0, 1.0 - temp], dtype=np.float32)
    result = result * channel_scale[None, None, :]

    return np.asarray(np.clip(result, 0.0, 1.0), dtype=np.float32)


def apply_mud_crust(rng: np.random.Generator, image: np.ndarray) -> np.ndarray:
    """Корка засохшего ила: бежево-коричневое тонирование + десатурация + текстура.

    Мост между лёгкой синтетикой (капли/пыль поверх чистого фото) и реальной
    экстремальной грязью, где машина покрыта коркой ила с изменением и цвета, и
    текстуры. В отличие от :func:`apply_mud_overlay` (локальные тёмные кляксы),
    здесь моделируется сплошной налёт: нижняя и средняя части кадра тонируются
    землистым цветом с высокой альфой, насыщенность снижается (грязь глушит
    цвета), поверх накладывается крупная низкочастотная текстура корки.

    Args:
        rng: Генератор случайных чисел.
        image: Изображение ``float32`` ``[0, 1]``, форма ``(H, W, 3)``.

    Returns:
        Изображение с коркой ила.
    """
    height, width = image.shape[:2]

    # Вертикальный профиль покрытия: верх (небо) почти чист, середина частично,
    # низ (ходовая, крылья) — под сплошной коркой.
    coverage = np.clip(np.linspace(-0.2, 1.2, height, dtype=np.float32), 0.0, 1.0)
    # Крупная текстура корки — низкочастотный шум, сглаженный сильным блюром.
    texture = _value_noise(rng, (height, width), octaves=3)
    texture = cv2.GaussianBlur(texture, (0, 0), sigmaX=min(height, width) * 0.03)
    # Текстура модулирует альфу, чтобы корка была неоднородной (комки/проплешины).
    texture_mod = 0.6 + 0.4 * texture

    base_alpha = float(rng.uniform(0.4, 0.7))
    alpha = coverage[:, None] * texture_mod * base_alpha
    alpha_3 = alpha[..., None]

    # Бежево-коричневый цвет засохшего ила с вариацией оттенка.
    crust_color = np.array(
        [
            rng.uniform(0.42, 0.55),  # R — бежево-коричневый
            rng.uniform(0.33, 0.44),  # G
            rng.uniform(0.22, 0.32),  # B
        ],
        dtype=np.float32,
    )

    # Десатурация: подмешиваем яркость (grayscale) к исходнику там, где корка.
    luminance = (image * np.array([0.299, 0.587, 0.114], dtype=np.float32)).sum(
        axis=2, keepdims=True
    )
    desat_strength = coverage[:, None, None] * float(rng.uniform(0.3, 0.6))
    desaturated = image * (1.0 - desat_strength) + luminance * desat_strength

    # Накладываем землистый цвет корки поверх десатурированного изображения.
    crusted = desaturated * (1.0 - alpha_3) + crust_color[None, None, :] * alpha_3
    return np.asarray(np.clip(crusted, 0.0, 1.0), dtype=np.float32)


DEGRADATIONS: dict[str, tuple[Degradation, float]] = {
    "mud": (apply_mud_overlay, 1.0),
    "mud_crust": (apply_mud_crust, 1.0),
    "dust": (apply_dust_overlay, 0.8),
    "splatter": (apply_splatter, 0.9),
    "occlusion": (apply_occlusion, 0.7),
    "rain": (apply_rain, 0.5),
    "fog": (apply_fog, 0.5),
    "lighting": (apply_lighting, 0.9),
}


def build_pipeline(
    rng: np.random.Generator,
    min_effects: int = 2,
    max_effects: int = 4,
) -> list[tuple[str, Degradation]]:
    """Выбрать случайное подмножество деградаций для одного изображения.

    Гарантирует, что хотя бы один «грязевой» эффект (mud/splatter/occlusion)
    присутствует — иначе изображение не выглядело бы грязным.

    Args:
        rng: Генератор случайных чисел.
        min_effects: Минимальное число эффектов.
        max_effects: Максимальное число эффектов.

    Returns:
        Список пар ``(имя, функция)`` в порядке применения.
    """
    names = list(DEGRADATIONS.keys())
    weights = np.array([DEGRADATIONS[n][1] for n in names], dtype=np.float64)
    weights /= weights.sum()

    count = int(rng.integers(min_effects, max_effects + 1))
    chosen = [
        str(name)
        for name in rng.choice(names, size=min(count, len(names)), replace=False, p=weights)
    ]

    # Обязательно хотя бы один грязевой эффект.
    dirt_effects = {"mud", "mud_crust", "splatter", "occlusion"}
    if not dirt_effects.intersection(chosen):
        chosen[0] = str(rng.choice(list(dirt_effects)))

    # Освещение, если выбрано, применяем последним (влияет на всю сцену).
    ordered = [n for n in chosen if n != "lighting"]
    if "lighting" in chosen:
        ordered.append("lighting")

    return [(name, DEGRADATIONS[name][0]) for name in ordered]


def generate_dirty_image(
    rng: np.random.Generator,
    image: np.ndarray,
    min_effects: int = 2,
    max_effects: int = 4,
) -> tuple[np.ndarray, list[str]]:
    """Применить случайный пайплайн деградаций к одному изображению.

    Args:
        rng: Генератор случайных чисел.
        image: Входное изображение ``float32`` ``[0, 1]``, форма ``(H, W, 3)``.
        min_effects: Минимальное число эффектов.
        max_effects: Максимальное число эффектов.

    Returns:
        Кортеж ``(грязное_изображение, список_применённых_эффектов)``.
    """
    pipeline = build_pipeline(rng, min_effects, max_effects)
    result = image
    applied: list[str] = []
    for name, func in pipeline:
        result = func(rng, result)
        applied.append(name)
    return np.asarray(np.clip(result, 0.0, 1.0), dtype=np.float32), applied


# ---------------------------------------------------------------------------
# I/O и обход дерева.
# ---------------------------------------------------------------------------
def _load_image(path: Path) -> np.ndarray | None:
    """Загрузить изображение как ``float32`` RGB в ``[0, 1]``.

    Args:
        path: Путь к файлу изображения.

    Returns:
        Массив ``(H, W, 3)`` или ``None`` при ошибке чтения.
    """
    data = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if data is None:
        return None
    rgb = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.float32) / 255.0


def _save_image(path: Path, image: np.ndarray) -> None:
    """Сохранить ``float32`` RGB-изображение в файл.

    Args:
        path: Путь назначения.
        image: Массив ``float32`` ``[0, 1]``, форма ``(H, W, 3)``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(np.clip(image * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)


def _iter_class_images(split_dir: Path) -> list[tuple[str, Path]]:
    """Собрать пары ``(имя_класса, путь)`` для одного сплита.

    Класс — имя поддиректории первого уровня. Служебные папки (``clean``,
    ``dirty``, ``to_review``) пропускаются, чтобы они не считались классами.

    Args:
        split_dir: Директория сплита (например, ``data/processed/train``).

    Returns:
        Список пар ``(класс, путь_к_изображению)``.
    """
    skip = set(STATE_DIRS) | {"to_review"}
    pairs: list[tuple[str, Path]] = []
    for class_dir in sorted(split_dir.iterdir()):
        if not class_dir.is_dir() or class_dir.name in skip:
            continue
        for image_path in sorted(class_dir.rglob("*")):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                pairs.append((class_dir.name, image_path))
    return pairs


def process_tree(
    clean_root: Path,
    output_root: Path,
    dirty_per_clean: int,
    seed: int,
    min_effects: int,
    max_effects: int,
) -> dict[str, int]:
    """Построить multi-task дерево clean/dirty из дерева чистых изображений.

    Для каждого сплита и класса оригиналы копируются в ``clean``, а к каждому
    оригиналу генерируется ``dirty_per_clean`` грязных вариантов в ``dirty``.

    Args:
        clean_root: Корень дерева чистых изображений (``<split>/<class>/``).
        output_root: Корень выходного дерева (``<split>/<class>/{clean,dirty}/``).
        dirty_per_clean: Сколько грязных вариантов на одно чистое изображение.
        seed: Базовое зерно ГПСЧ (детерминированность).
        min_effects: Минимум эффектов на вариант.
        max_effects: Максимум эффектов на вариант.

    Returns:
        Счётчик ``{"clean": N, "dirty": M}`` записанных изображений.
    """
    counts = {"clean": 0, "dirty": 0}
    master_rng = np.random.default_rng(seed)

    split_dirs = [child for child in sorted(clean_root.iterdir()) if child.is_dir()]
    for split_dir in split_dirs:
        split_name = split_dir.name
        pairs = _iter_class_images(split_dir)
        print(f"[{split_name}] изображений-источников: {len(pairs)}")

        for class_name, image_path in pairs:
            image = _load_image(image_path)
            if image is None:
                print(f"  [skip] не читается: {image_path}")
                continue

            # clean-копия.
            clean_dest = output_root / split_name / class_name / "clean" / image_path.name
            _save_image(clean_dest, image)
            counts["clean"] += 1

            # dirty-варианты с детерминированными под-зёрнами.
            for variant in range(dirty_per_clean):
                variant_seed = int(master_rng.integers(0, 2**32 - 1))
                variant_rng = np.random.default_rng(variant_seed)
                dirty, applied = generate_dirty_image(variant_rng, image, min_effects, max_effects)
                dirty_name = f"{image_path.stem}_dirty{variant}{image_path.suffix}"
                dirty_dest = output_root / split_name / class_name / "dirty" / dirty_name
                _save_image(dirty_dest, dirty)
                counts["dirty"] += 1

    return counts


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Разобрать аргументы командной строки.

    Args:
        argv: Список аргументов (по умолчанию ``sys.argv[1:]``).

    Returns:
        Пространство имён с аргументами.
    """
    parser = argparse.ArgumentParser(
        description="Генерация реалистичного грязного датасета из чистых фото.",
    )
    parser.add_argument(
        "--clean-dir",
        type=Path,
        default=Path("data/processed"),
        help="Корень дерева чистых изображений (<split>/<class>/).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/dirty_clean"),
        help="Корень выходного дерева (<split>/<class>/{clean,dirty}/).",
    )
    parser.add_argument(
        "--dirty-per-clean",
        type=int,
        default=2,
        help="Число грязных вариантов на одно чистое изображение.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Базовое зерно ГПСЧ для воспроизводимости.",
    )
    parser.add_argument(
        "--min-effects",
        type=int,
        default=2,
        help="Минимальное число деградаций на вариант.",
    )
    parser.add_argument(
        "--max-effects",
        type=int,
        default=4,
        help="Максимальное число деградаций на вариант.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Точка входа генератора.

    Args:
        argv: Аргументы командной строки (для тестируемости).

    Returns:
        Код возврата процесса (0 — успех).
    """
    args = parse_args(argv)

    if not args.clean_dir.is_dir():
        print(f"[ошибка] Директория чистых изображений не найдена: {args.clean_dir}")
        return 1

    print(f"Источник: {args.clean_dir} -> Выход: {args.output_dir}")
    print(f"Грязных на чистое: {args.dirty_per_clean} | seed: {args.seed}")

    counts = process_tree(
        clean_root=args.clean_dir,
        output_root=args.output_dir,
        dirty_per_clean=args.dirty_per_clean,
        seed=args.seed,
        min_effects=args.min_effects,
        max_effects=args.max_effects,
    )

    print("\n" + "=" * 50)
    print("ИТОГ ГЕНЕРАЦИИ")
    print("=" * 50)
    print(f"clean записано: {counts['clean']}")
    print(f"dirty записано: {counts['dirty']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
