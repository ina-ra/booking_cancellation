# Booking Cancellation Prediction Service

Сервис предсказывает вероятность отмены бронирования в гостиничном бизнесе.

Проект оценивает риск отмены по признакам бронирования и использует прогноз в трёх сценариях:

- online inference для одного бронирования;
- batch scoring для списка бронирований;
- мониторинг качества модели и процесса предобработки.

## Архитектура

Код организован по Clean Architecture:

- `src/domain` — доменные сущности и бизнес-правила риска;
- `src/application` — use cases обучения, скоринга и мониторинга;
- `src/infrastructure` — адаптеры БД, S3/MinIO и ML-интеграции;
- `src/interfaces` — API, CLI и точка входа приложения;
- `tests` — unit-тесты.

Ключевые доменные сущности:

- `Booking` — предметное представление одного бронирования;
- `BookingRiskScore` — результат оценки риска отмены для одного бронирования;
- `BatchScoringResult` — результат batch scoring для группы бронирований;
- `TrainingResult` — результат обучения модели с метриками, параметрами и отчётом.

## Что делает сервис

1. Обучает LightGBM-модель на исторических бронированиях.
2. Представляет входные бронирования как доменные сущности и рассчитывает для них риск отмены.
3. Сохраняет артефакты модели только в S3-compatible storage.
4. Загружает модель из S3 при старте API.
5. Пишет метрики обучения и batch scoring в Postgres.

## Требования

- Python 3.11+
- Postgres
- S3-compatible storage
- Для локальной разработки можно использовать MinIO через `docker-compose.minio.yml`

## Локальный запуск

1. Скопировать `.env.example` в `.env`.
2. Заполнить `POSTGRES_*` и `S3_*` переменные.
3. При локальной разработке поднять MinIO:

```powershell
docker compose -f docker-compose.minio.yml up -d
```

1. Установить зависимости:

```powershell
py -m pip install -r requirements.txt
py -m pip install -r requirements-dev.txt
```

1. Обучить модель и загрузить артефакты в S3:

```powershell
py -m src.interfaces.cli.train_models_cli
```

1. Запустить API:

```powershell
py -m uvicorn src.interfaces.main:app --reload
```

1. Открыть Swagger UI:

```text
http://127.0.0.1:8000/docs
```

## MinIO / Local S3

Проект использует S3 как единственный источник истины для артефактов модели.
Локальный каталог `artifacts/` не хранится в git и не используется как fallback для загрузки модели.

Для локального MinIO:

```powershell
docker compose -f docker-compose.minio.yml up -d
```

Заполнить в `.env`:

- `S3_ENDPOINT_URL=http://localhost:9000`
- `S3_BUCKET=booking-cancellation-artifacts`
- `S3_ACCESS_KEY=<your-local-minio-user>`
- `S3_SECRET_KEY=<your-local-minio-password>`
- `S3_REGION=us-east-1`
- `S3_ARTIFACTS_PREFIX=artifacts`
- `S3_AUTO_CREATE_BUCKET=true`
- `S3_USE_PATH_STYLE=true`
- `MINIO_ROOT_USER=<your-local-minio-user>`
- `MINIO_ROOT_PASSWORD=<your-local-minio-password>`

Как это работает:

- `train_models_cli` обучает модель и публикует `lightgbm_model.txt`, `lightgbm_model.pkl` и `model_comparison.json` напрямую в S3;
- API на старте загружает модель и метаданные из S3;
- batch scoring CLI использует модель из S3 и пишет мониторинговые записи в Postgres.

## API

- `GET /health` — healthcheck и статус загрузки модели.
- `POST /predict` — предсказание вероятности отмены для одного бронирования.
- `POST /predict/batch` — batch scoring для списка бронирований.

## Мониторинг

Сервис сохраняет в Postgres:

- ML-метрики обучения: `accuracy`, `precision`, `recall`, `f1`, `roc_auc`;
- технические и процессные метрики: число записей до и после предобработки, долю high-risk, среднюю вероятность отмены, число невалидных дат.

## Тесты

```powershell
pytest
```

В проекте настроены:

- unit-тесты;
- `ruff`;
- `mypy`;
- GitHub Actions CI.

## Deploy

Для деплоя нужны:

- отчуждённый Postgres;
- S3-compatible storage для артефактов модели;
- заполненные переменные окружения из `.env.example`.

Базовый порядок деплоя:

1. Поднять Postgres и S3 storage.
2. Задать `POSTGRES_*` и `S3_*` переменные окружения.
3. Один раз выполнить `py -m src.interfaces.cli.train_models_cli`, чтобы артефакты модели появились в S3.
4. Запустить API командой `py -m uvicorn src.interfaces.main:app --host 0.0.0.0 --port 8000`.
5. Проверить `GET /health` и `POST /predict`.
