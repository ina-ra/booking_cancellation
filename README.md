# Booking Cancellation Prediction Service

Сервис для предсказания вероятности отмены бронирования в гостиничном бизнесе.

Проект помогает оценивать риск отмены бронирования на основе признаков заказа и использовать этот прогноз в прикладных сценариях: анализе бронирований, приоритизации high-risk случаев и batch scoring.

## Архитектура

Код организован по Clean Architecture:

- `src/domain` - доменные сущности и бизнес-правила определения риска.
- `src/application` - use cases и прикладная логика scoring/training.
- `src/infrastructure` - адаптеры предобработки, загрузки модели и работы с файлами.
- `src/interfaces` - API-слой, CLI entrypoints и точка входа приложения.
- `app` - FastAPI entrypoint и runtime-конфигурация сервиса.
- `tests` - unit-тесты для ключевой логики проекта.

## Структура репозитория

```text
booking_cancellation/
├── .github/
│   └── workflows/
│       └── ci.yml
├── app/
│   ├── main.py
│   ├── api/
│   ├── core/
│   └── schemas/
├── artifacts/
│   ├── lightgbm_model.pkl
│   ├── lightgbm_model.txt
│   └── model_comparison.json
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── application/
│   ├── domain/
│   ├── infrastructure/
│   ├── interfaces/
│   └── config.py
├── tests/
├── .coveragerc
├── mypy.ini
├── pytest.ini
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

## Локальный запуск

Подготовить окружение:

- скопировать `.env.example` в `.env`;
- заполнить переменные под локальную среду при необходимости.

Установить зависимости:

- `py -m pip install -r requirements.txt`
- `py -m pip install -r requirements-dev.txt`

Запустить API:

- `py -m uvicorn app.main:app --reload`

Открыть Swagger UI:

- `http://127.0.0.1:8000/docs`

## MinIO / Local S3

Проект поддерживает хранение артефактов модели в MinIO через S3-compatible API.
Если S3-переменные заданы в `.env`, сервис использует MinIO как источник истины для модели и `model_comparison.json`.

Поднять локальный MinIO:

- `docker compose -f docker-compose.minio.yml up -d`

Заполнить переменные в `.env`:

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

- `train_models_cli` сохраняет артефакты локально в `artifacts/`, затем загружает их в MinIO.
- API на старте и batch scoring CLI загружают модель и метаданные из MinIO, если S3 включен.
- Если S3 не настроен, проект продолжает работать в локальном файловом режиме.
- `docker-compose.minio.yml` читает MinIO credentials из локального `.env`, поэтому секреты не хранятся в репозитории.

## API

- `GET /health` - healthcheck и проверка загрузки модели.
- `POST /predict` - предсказание вероятности отмены для одного бронирования.
- `POST /predict/batch` - batch scoring для списка бронирований.
