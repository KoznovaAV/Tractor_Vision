# Чек-лист переобучения и оценки Tractor Vision

Порядок операций после всех правок. Команды даны для Windows + PowerShell,
Python 3.11, из корня проекта. GPU-обучение предполагает доступную CUDA.

---

## Фаза 0. Подготовка окружения

1. Установить зависимости для разработки:

   ```powershell
   pip install -r requirements.txt
   ```

2. Убедиться, что единый источник классов согласован с конфигом:

   ```powershell
   python -c "from src.config.classes import MODEL_CLASSES; from src.config.config_loader import load_config; c=load_config(); assert c.num_model_classes==len(MODEL_CLASSES)==4; print('OK', MODEL_CLASSES)"
   ```

   Ожидается: `OK ('chtz_b10m', 'johndeere', 'kirovets_k744', 'mtz_82')`.

---

## Фаза 1. Слияние классов (mtz_1221 -> mtz_82)

**Выполняется один раз** над существующим датасетом. Сначала предпросмотр:

3. Предпросмотр (ничего не меняет на диске):

   ```powershell
   python -m scripts.merge_mtz1221 --dry-run
   ```

4. Применить слияние:

   ```powershell
   python -m scripts.merge_mtz1221
   ```

   Проверить: папок `mtz_1221` не осталось ни в `data/processed`, ни в
   `data/dirty_clean`; счётчики `mtz_82` выросли на величину слитых.

---

## Фаза 2. Данные: collect -> prepare -> generate_dirty

**Строгий порядок.** Каждый шаг — вход следующего.

### 2a. Сбор изображений

5. Собрать по каждому классу (начать с малого объёма — проверить выдачу):

   ```powershell
   python -m scripts.collect_tractors --output-dir data/collected --split train --per-class 20 --clip-threshold 0.7
   ```

   Убедиться, что папки классов заполнились, `to_review` содержит
   низкоувереннные фото, `data/collected/collect_manifest.json` валиден. Затем
   догнать объём до цели (500–750 на класс для датасета 2000–3000):

   ```powershell
   python -m scripts.collect_tractors --output-dir data/collected --split train --per-class 600 --clip-threshold 0.7
   ```

6. **Ручная чистка `to_review`**: просмотреть папку, перенести валидные фото в
   соответствующий класс, мусор удалить. Это единственный ручной шаг.

### 2b. Объединение и стратифицированная пересборка сплитов 70/15/15

7. Слить существующий датасет и собранное в единое дерево с пересборкой сплитов:

   ```powershell
   python -m scripts.prepare_dataset --sources data/processed data/collected --output-dir data/processed_rebuilt --ratios 0.7 0.15 0.15 --seed 42
   ```

   Проверить распределение в выводе: каждый класс представлен во всех трёх
   сплитах, доли близки к 70/15/15. Дедупликация по содержимому отработала.

8. Заменить рабочее дерево пересобранным (после проверки распределения):

   ```powershell
   Remove-Item -Recurse -Force data/processed
   Rename-Item data/processed_rebuilt data/processed
   ```

### 2c. Генерация реалистичной грязи ПОВЕРХ финального processed

9. Построить multi-task дерево clean/dirty из чистого пересобранного датасета:

   ```powershell
   python -m src.data.generate_dirty_dataset --clean-dir data/processed --output-dir data/dirty_clean --dirty-per-clean 2 --seed 42
   ```

   Проверить: у каждого класса есть `clean/` и `dirty/`; грязь садится на технику
   (нижняя часть кадра), небо остаётся чистым. Просмотреть 5–10 сэмплов глазами.

---

## Фаза 3. Обучение

### 3a. Single-task (базовая линия, при необходимости)

10. Обучить single-task модель на пересобранном датасете:

    ```powershell
    python -m src.training.train --data-dir data/processed --image-size 384
    ```

### 3b. Multi-task с partial fine-tuning

11. Обучить multi-task с разморозкой последних 2 стадий и дифференцированным LR.
    Начать с uncertainty (стабильнее), затем сравнить с gradnorm:

    ```powershell
    python -m src.training.multi_task_train --data-dir data/dirty_clean --num-unfrozen-stages 2 --loss-balancing uncertainty --image-size 384 --max-epochs 100 --accelerator gpu
    ```

    ```powershell
    python -m src.training.multi_task_train --data-dir data/dirty_clean --num-unfrozen-stages 2 --loss-balancing gradnorm --image-size 384 --max-epochs 100 --accelerator gpu
    ```

    В логах проверить строку `Разморожены стадии backbone (features): [4, 5, 6, 7]`
    и что LR backbone в 10 раз меньше LR голов (LearningRateMonitor).

12. Если multi-task всё ещё заметно хуже single-task — попробовать
    `--num-unfrozen-stages 1` (меньше риск переобучения на малом датасете) и
    сравнить обе схемы балансировки по `val_model_acc`.

---

## Фаза 4. Оценка

13. Оценить single-task на test-сплите:

    ```powershell
    python -m src.training.evaluate --data-dir data/processed --image-size 384
    ```

14. Оценить multi-task (оценка идёт по val — в dirty_clean нет test-сплита):

    ```powershell
    python -m src.training.multi_task_evaluate --data-dir data/dirty_clean --image-size 384
    ```

15. Проверить артефакты оценки: `output/confusion_matrix.png`,
    `output/misclassified_examples.png`. Убедиться, что после слияния классов
    матрица ошибок 4×4 (не 5×5) и нет утечки бывшего `mtz_1221`.

---

## Фаза 5. Проверки консистентности перед выкладкой

16. `image_size` единый (384) в train, eval и API — читается из `config.yaml`.
    Проверить, что инференс не режется на 224:

    ```powershell
    python -c "from src.config.config_loader import load_config; assert load_config().image_size==384; print('image_size unified: 384')"
    ```

17. Прогнать тесты (гейт CI — flake8 + pytest, не mypy):

    ```powershell
    pre-commit run --all-files
    pytest tests/ -v --cov=src
    ```

18. Собрать Docker-образы и проверить, что оба варианта собираются:

    ```powershell
    docker build -t tractor-vision:cpu .
    docker build --build-arg TORCH_VARIANT=cu121 -t tractor-vision:gpu .
    ```

19. Поднять API и проверить health + предсказание:

    ```powershell
    docker compose up api
    # в другом окне:
    curl http://localhost:8000/health
    ```

    Убедиться, что `/models` отдаёт accuracy, прочитанную из метаданных
    чекпоинта (а не захардкоженные 0.9149 / 0.7917).

---

## Что проверить глазами (не автоматизируется)

- Сэмплы грязи: грязь на технике, не на небе; фартук у колёс выглядит
  правдоподобно.
- `to_review` вычищен до prepare_dataset.
- Матрица ошибок 4×4, классы сбалансированы по сплитам.
- Логи обучения: разморожены `features[4:8]`, backbone LR = 0.1 × heads LR.
