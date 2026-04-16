# Booking Cancellation Prediction Service

Сервис предсказывает вероятность отмены бронирования в гостиничном бизнесе и поддерживает:

- online inference для одного бронирования;
- batch scoring для набора бронирований;
- сохранение ML- и технических метрик в Postgres;
- хранение модельных артефактов в S3-compatible storage.

## Архитектура

Проект организован по Clean Architecture:

- `src/domain` — доменные сущности и бизнес-правила;
- `src/application` — use cases обучения, скоринга и мониторинга;
- `src/infrastructure` — интеграции с Postgres, S3/MinIO и ML-адаптеры;
- `src/interfaces` — FastAPI, CLI и точки входа;
- `tests` — unit-тесты.

Ключевые доменные сущности:

- `Booking` — предметное представление одного бронирования;
- `BookingRiskScore` — результат оценки риска отмены для одного бронирования;
- `BatchScoringResult` — результат batch scoring для группы бронирований;
- `TrainingResult` — результат обучения модели с метриками и параметрами.

## Внешние зависимости

Для полной работы сервиса нужны:

- `Postgres` — хранение prediction records и мониторинга;
- `S3-compatible storage` — хранение артефактов модели;
- локально можно использовать `MinIO` как S3-compatible storage.

Важно: модель загружается из S3 при старте API, поэтому S3 должен быть доступен до запуска `uvicorn`.

## Быстрый локальный запуск

### 1. Подготовить `.env`

Скопируйте `.env.example` в `.env` и заполните локальные секреты.

Базовые локальные значения:

- `POSTGRES_HOST=localhost`
- `POSTGRES_PORT=5432`
- `POSTGRES_DB=booking_cancellation`
- `POSTGRES_USER=booking_user`
- `POSTGRES_SSLMODE=disable`
- `S3_ENDPOINT_URL=http://localhost:9000`
- `S3_BUCKET=booking-cancellation-artifacts`
- `S3_REGION=us-east-1`
- `S3_ARTIFACTS_PREFIX=artifacts`
- `S3_AUTO_CREATE_BUCKET=true`
- `S3_USE_PATH_STYLE=true`

Локально нужно задать в `.env`:

- `POSTGRES_PASSWORD`
- `S3_ACCESS_KEY`
- `S3_SECRET_KEY`
- `MINIO_ROOT_USER`
- `MINIO_ROOT_PASSWORD`

Обычно удобно использовать одинаковые пары:

- `MINIO_ROOT_USER = S3_ACCESS_KEY`
- `MINIO_ROOT_PASSWORD = S3_SECRET_KEY`

Для batch-результатов в S3 используется отдельный префикс:

- `S3_BATCH_OUTPUTS_PREFIX=batch-runs`

### 2. Установить зависимости

```powershell
py -m venv venv
.\venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
py -m pip install -r requirements-dev.txt
```

### 3. Запустить Docker Desktop

Перед запуском локальной инфраструктуры Docker Desktop должен быть открыт, а Docker Engine — запущен.

Проверка:

```powershell
docker version
```

Команда должна показывать и `Client`, и `Server`.

### 4. Поднять локальную инфраструктуру и загрузить модель

Самый короткий сценарий:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\bootstrap_local.ps1
```

Что делает скрипт:

1. поднимает `Postgres` и `MinIO` через `docker-compose.local.yml`;
2. инициализирует таблицы в Postgres;
3. обучает модель;
4. загружает артефакты модели в S3/MinIO.

Если хотите сделать это вручную:

```powershell
docker compose -f docker-compose.local.yml up -d
py -m src.interfaces.cli.init_db_cli
py -m src.interfaces.cli.train_models_cli
```

### 5. Запустить API

```powershell
py -m uvicorn src.interfaces.main:app --reload
```

После запуска можно открыть:

- Swagger UI: `http://127.0.0.1:8000/docs`
- health check: `http://127.0.0.1:8000/health`
- MinIO console: `http://127.0.0.1:9001`

## Что считается успешным запуском

Если всё поднялось правильно:

- `http://127.0.0.1:8000/health` отвечает JSON вроде:

```json
{"status":"ok","model_loaded":true,"model_name":"LightGBM"}
```

- в MinIO console есть бакет `booking-cancellation-artifacts`;
- внутри бакета лежат:
  - `artifacts/lightgbm_model.txt`
  - `artifacts/lightgbm_model.pkl`
  - `artifacts/model_comparison.json`

## Batch scoring

Базовая команда:

```powershell
py -m src.interfaces.cli.predict_cli
```

Рекомендуемый режим для batch-сервиса:

```powershell
py -m src.interfaces.cli.predict_cli --run-date 2026-04-16
```

В этом случае результаты будут записаны в:

- `artifacts/batch_runs/2026-04-16/booking_risk_scores.csv`
- `artifacts/batch_runs/2026-04-16/high_risk_bookings.csv`

Это удобно для детерминированного запуска batch-задач и дальнейшего внедрения идемпотентности и backfill.

Если run-date уже был успешно обработан, batch CLI увидит `_SUCCESS` marker в S3 и пропустит повторный запуск.
Для принудительного пересчёта можно использовать:

```powershell
py -m src.interfaces.cli.predict_cli --run-date 2026-04-16 --force
```

## Airflow

Для batch-сервиса в проект добавлен Airflow DAG:

- `airflow/dags/batch_scoring_dag.py`

Что он делает:

- запускает batch scoring по расписанию `@daily`;
- использует `DockerOperator`;
- передаёт в контейнер логическую дату запуска как `{{ ds }}`;
- запускает batch-контейнер в docker-сети `booking_cancellation_default`;
- работает с `Postgres` и `MinIO` через контейнерные адреса `postgres` и `minio`;
- поддерживает catchup и backfill.

Текущая структура DAG:

1. `check_batch_not_processed` — проверяет, есть ли `_SUCCESS` marker для run-date;
2. `run_batch_scoring` — запускает основной batch scoring в контейнере;
3. `verify_batch_outputs` — проверяет, что результаты и `_SUCCESS` marker появились в S3.

### Поднять Airflow локально

Сначала инфраструктура проекта уже должна быть поднята:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\bootstrap_local.ps1
```

Потом запустить Airflow:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\bootstrap_airflow.ps1
```

После старта:

- Airflow UI: `http://127.0.0.1:8081`
- логин/пароль по умолчанию: `airflow / airflow`

### Backfill

У DAG включён `catchup=True`, поэтому прошлые логические даты можно дообработать.

Пример backfill за три дня:

```powershell
docker compose -f docker-compose.airflow.yml exec airflow-standalone airflow dags backfill booking_batch_scoring --start-date 2026-04-14 --end-date 2026-04-16
```

Идемпотентность обеспечивается двумя слоями:

- output paths partitioned by `run_date`;
- `_SUCCESS` marker в S3 не даёт повторно выполнить уже успешный batch-run без `--force`.

## API

- `GET /health` — проверка доступности сервиса и факта загрузки модели;
- `POST /predict` — предсказание вероятности отмены для одного бронирования;
- `POST /predict/batch` — batch scoring для списка бронирований.

## Мониторинг

Сервис сохраняет в Postgres:

- ML-метрики обучения: `accuracy`, `precision`, `recall`, `f1`, `roc_auc`;
- batch-метрики: число записей до и после предобработки, долю high-risk, среднюю вероятность отмены, число невалидных дат.

## Тесты и проверки

```powershell
ruff check .
mypy src
pytest --basetemp=.pytest_tmp -o cache_dir=.pytest_cache_local
```

## Docker

`Dockerfile` нужен для контейнеризации приложения: он собирает воспроизводимый образ с Python, зависимостями и кодом сервиса.

`docker-compose.local.yml` нужен для удобного локального старта инфраструктуры:

- `Postgres`
- `MinIO`

Это подготовка к следующему этапу — batch-сервису на Airflow с `DockerOperator`.

## Deploy

Для деплоя нужны:

- отчуждённый `Postgres`;
- `S3-compatible storage` для артефактов модели;
- переменные окружения из `.env.example`;
- предварительный запуск `py -m src.interfaces.cli.train_models_cli`, чтобы модель оказалась в S3.

Базовый порядок:

1. поднять Postgres и S3;
2. задать `POSTGRES_*` и `S3_*`;
3. выполнить `py -m src.interfaces.cli.train_models_cli`;
4. запустить API командой `py -m uvicorn src.interfaces.main:app --host 0.0.0.0 --port 8000`;
5. проверить `GET /health`.
