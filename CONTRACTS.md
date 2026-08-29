# Контракт-чеклист: совместимость с существующими тестами

Файлы `dataset.py`, `multi_task.py`, `dataloader.py`, `transforms.py` были
пересозданы по сигнатурам из технического отчёта (оригиналов на руках не было).
Ниже — публичные контракты, сохранённые дословно, чтобы существующие 20 тестов
прошли без правок. Каждый пункт проверен прогоном.

## 1. `TractorDataset.__getitem__`

- Возвращает `dict`.
- Ключи `"image"` и `"model_label"` — всегда.
- Ключ `"state_label"` — только при `multi_task=True`.
- `model_label` / `state_label` — `torch.Tensor` типа `long`.
- Конструктор: `TractorDataset(root_dir, transform=None, multi_task=False)`.
- Покрывает: `test_dataset_getitem`, `test_multi_task_dataset`,
  `test_dataset_initialization`, `test_invalid_image_extensions`.

Изменение поведения (не ломающее контракт): классы итерируются **явно из
`MODEL_CLASSES`**, а не через glob директорий. Служебные папки (`to_review`,
`clean`, `dirty` на уровне класса) не становятся классами. Тест
`test_invalid_image_extensions` (игнор `.txt`) продолжает проходить — фильтр по
расширению сохранён.

## 2. `MultiTaskTractorClassifier`

- `forward(x)` возвращает **кортеж** `(model_logits, state_logits)`.
- Формы: `model_logits = (B, num_model_classes)`, `state_logits = (B, 2)`.
- Признаки после backbone — 768 (`CONVNEXT_TINY_FEATURES`).
- Конструктор: `MultiTaskTractorClassifier(num_model_classes=…, num_state_classes=…)`.
- Покрывает: `test_forward_pass` (shapes), `test_feature_extraction` (768),
  `test_model_initialization`.

Про `log_var_model` / `log_var_state`: в оригинале эти `nn.Parameter` жили в
**Lightning-модуле** `MultiTaskLightningModule` (обучающий скрипт), не в самой
модели `MultiTaskTractorClassifier`. Там они и остались — сохранены как
`nn.Parameter` в режиме `loss_balancing="uncertainty"` (режим по умолчанию).
В новом режиме `gradnorm` вместо них используются `task_weights`; на контракт
модели это не влияет, а `test_model.py` тестирует модель, а не Lightning-модуль.

## 3. `get_train_transforms` / `get_val_transforms`

- Сигнатура: `get_*_transforms(image_size: int = …)`.
- Возвращают `albumentations.Compose`, дающий на выходе `torch.Tensor` формы
  `(3, H, W)` (через `ToTensorV2`), где `H = W = image_size`.
- Нормализация ImageNet mean/std сохранена.
- Покрывает: `test_train_transforms`, `test_val_transforms`,
  `test_transform_output_shape` (shape `(3, 224, 224)`).

Значение по умолчанию `image_size` теперь 384 (единый размер из `config.yaml`),
но тесты передают размер явно (224), поэтому дефолт на них не влияет.

## 4. `get_dataloader`

- Сигнатура сохранена дословно:
  `get_dataloader(data_dir, split, batch_size=16, image_size=224, num_workers=2, multi_task=False)`.
- Возвращает `torch.utils.data.DataLoader`.
- `shuffle=True` только для `split="train"`.
- `pin_memory` включается только при доступной CUDA (исправление утечки памяти
  на CPU-only из отчёта).
- Покрывает: `test_get_dataloader`.

## Замечание по API-тестам

`test_api.py` (6 тестов) завязан на `src/api/main.py`, который в этой партии
**не трогался** (обновление API под чтение из `config.yaml` — отдельная задача,
если понадобится). Тесты вида «статус 200 или 500» устойчивы к тому, загружены
модели или нет, поэтому не регрессируют от изменений в других модулях. Если
будешь переводить API на `config_loader`, свериться нужно будет с
`test_list_models` (ключи `"models"`, `"count"`) и `HealthResponse`.
