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

## Запуск без ручной магии

Теперь у проекта есть production-friendly сценарий запуска, где больше не нужно вручную:

- собирать и экспортировать Docker image в `.tar`;
- отдельно запускать `uvicorn`;
- отдельно запускать `npm run dev`;
- отдельно инициализировать схему Postgres;
- отдельно обучать модель и грузить артефакты в S3 перед стартом API.

Это заменено на один orchestrated flow через `docker-compose.prod.yml`.

### 1. Подготовить `.env`

Для локального self-contained запуска:

```powershell
Copy-Item .env.prod.example .env
```

Дальше задайте в `.env` свои секреты:

- `POSTGRES_PASSWORD`
- `S3_SECRET_KEY`
- `MINIO_ROOT_PASSWORD`
- `AIRFLOW_ADMIN_PASSWORD`

Базовые значения в `.env.prod.example` уже настроены под compose-сеть:

- `POSTGRES_HOST=postgres`
- `POSTGRES_PORT=5432`
- `S3_ENDPOINT_URL=http://minio:9000`
- `S3_BUCKET=booking-cancellation-artifacts`
- `S3_ACCESS_KEY=booking_minio`

### 2. Запустить весь стек одной командой

```powershell
docker compose -f docker-compose.prod.yml up --build
```

Эта команда теперь делает весь runtime-path автоматически:

1. собирает production image `booking-cancellation-app:latest`;
2. собирает React frontend в `frontend/dist` внутри Docker build;
3. поднимает `Postgres` и `MinIO`;
4. запускает one-off job `init-db`, который ждёт Postgres и накатывает схему;
5. запускает one-off job `seed-model`, который:
   - ждёт Postgres и S3,
   - проверяет, есть ли артефакты модели в S3,
   - если артефактов нет, обучает модель и публикует их;
6. запускает `app`, который уже:
   - грузит модель из S3,
   - отдаёт API,
   - отдаёт собранный frontend;
7. только после этого поднимает `Airflow`, чтобы batch DAG не стартовал раньше готовности БД и модели.

### 3. Что открывать после старта

- UI и API с одного сервиса: `http://127.0.0.1:8000`
- Swagger UI: `http://127.0.0.1:8000/docs`
- readiness check: `http://127.0.0.1:8000/ready`
- health check: `http://127.0.0.1:8000/health`
- MinIO console: `http://127.0.0.1:9001`
- Airflow UI: `http://127.0.0.1:8081`

### Как теперь работает фронтенд

Frontend больше не требует отдельного `npm run dev` для production-like сценария.

Во время Docker build:

1. Node stage собирает React/Vite приложение;
2. артефакты сборки попадают в финальный Python image;
3. FastAPI отдает:
   - API маршруты;
   - собранный frontend;
   - статику из `frontend/dist/assets`.

То есть в итоге фронтенд и бэкенд работают с одного хоста и одного порта.

### Как теперь устроены автоматические шаги

#### `init-db`

Запускается как одноразовый контейнер:

```powershell
python -m src.interfaces.cli.runtime_prepare_cli --step init-db
```

Что он делает:

- ждёт доступности Postgres;
- вызывает `ensure_database_schema()`;
- завершает работу после успешной инициализации.

#### `seed-model`

Запускается как одноразовый контейнер:

```powershell
python -m src.interfaces.cli.runtime_prepare_cli --step seed-model
```

Что он делает:

- ждёт доступности Postgres;
- ждёт доступности S3/MinIO;
- проверяет, есть ли model artifacts в S3;
- если артефакты уже есть, ничего не переобучает;
- если артефактов нет, запускает обучение и публикует модель.

Это и убирает главную ручную магию перед первым стартом.

## Локальный compose в том же стиле

`docker-compose.local.yml` теперь использует тот же автоматический граф сервисов:

- `postgres`
- `minio`
- `init-db`
- `seed-model`
- `app`
- `airflow-standalone`

### Канонический локальный запуск

```powershell
docker compose -f docker-compose.local.yml up --build
```

Или через wrapper:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\bootstrap_local.ps1
```

Если Airflow пока не нужен:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\bootstrap_local.ps1 -SkipAirflow
```

Что теперь делает local compose:

1. собирает production-like image приложения;
2. собирает frontend;
3. поднимает `Postgres` и `MinIO`;
4. выполняет `init-db`;
5. выполняет `seed-model`;
6. запускает `app` на `http://127.0.0.1:8000`;
7. при полном сценарии поднимает `Airflow`.

То есть локально больше не нужно вручную:

- запускать `uvicorn`;
- запускать `npm run dev`;
- вручную выполнять `init_db_cli`;
- вручную обучать модель перед стартом API.

## CI/CD и registry flow

В репу добавлены два workflow:

- `.github/workflows/ci.yml`
- `.github/workflows/docker-publish.yml`

### `ci.yml`

Этот workflow:

1. поднимает Python 3.11;
2. поднимает Node 20;
3. устанавливает backend зависимости;
4. выполняет `npm ci` во `frontend`;
5. собирает frontend;
6. запускает:
   - `ruff check .`
   - `mypy src`
   - `pytest`
7. отдельно проверяет production Docker build:

```bash
docker build -t booking-cancellation-app:test .
```

### `docker-publish.yml`

Этот workflow начинает registry flow через GitHub Container Registry (`ghcr.io`).

Он запускается:

- при push в `main` или `master`;
- при push тегов `v*`;
- вручную через `workflow_dispatch`.

Workflow:

1. логинится в `ghcr.io`;
2. вычисляет Docker tags через `docker/metadata-action`;
3. собирает production image;
4. публикует его в registry.

Целевой образ:

```text
ghcr.io/<owner>/booking-cancellation-app
```

Теги включают:

- branch tag;
- tag release;
- sha;
- `latest` для default branch.

Это создаёт основу для следующего шага: staging/prod deploy уже не из локального build, а из image registry.

## Staging deploy flow

Пошаговая подготовка будущего staging-сервера вынесена в [deploy/staging/SERVER_ONBOARDING.md](deploy/staging/SERVER_ONBOARDING.md).

В проект добавлен первый staging deployment skeleton:

- `docker-compose.staging.yml`
- `deploy/staging/.env.staging.example`
- `.github/workflows/deploy-staging.yml`

### Что делает staging compose

`docker-compose.staging.yml` запускает тот же граф сервисов, но уже без локального build:

- `postgres`
- `minio`
- `init-db`
- `seed-model`
- `app`
- `airflow-standalone`

Вместо `build:` он использует image из registry:

```text
ghcr.io/<owner>/booking-cancellation-app:<tag>
```

Значение image приходит через переменную:

- `APP_IMAGE`

### Что нужно подготовить на staging-сервере

На сервере должна быть директория, например:

```text
/opt/booking-cancellation
```

Там workflow будет размещать:

- `docker-compose.staging.yml`
- `.env.staging`
- `airflow/dags`

На сервере должны быть установлены:

- Docker
- Docker Compose plugin

### Какие GitHub secrets нужны для staging deploy

#### SSH и доступ к серверу

- `STAGING_HOST`
- `STAGING_SSH_USER`
- `STAGING_SSH_KEY`
- `STAGING_SSH_PORT`
- `STAGING_APP_DIR`

#### Registry read access

- `GHCR_READ_USER`
- `GHCR_READ_TOKEN`

#### Staging Postgres

- `STAGING_POSTGRES_DB`
- `STAGING_POSTGRES_USER`
- `STAGING_POSTGRES_PASSWORD`
- `STAGING_POSTGRES_PORT`
- `STAGING_POSTGRES_SSLMODE`

#### Staging S3 / MinIO

- `STAGING_S3_BUCKET`
- `STAGING_S3_ACCESS_KEY`
- `STAGING_S3_SECRET_KEY`
- `STAGING_S3_REGION`
- `STAGING_MINIO_ROOT_USER`
- `STAGING_MINIO_ROOT_PASSWORD`

#### Staging Airflow

- `STAGING_AIRFLOW_ADMIN_USERNAME`
- `STAGING_AIRFLOW_ADMIN_PASSWORD`

### Какие GitHub environment variables полезно задать

Для environment `staging` можно задать `vars`:

- `STAGING_DEFAULT_HIGH_RISK_THRESHOLD`
- `STAGING_DEFAULT_BATCH_RISK_SHARE`
- `STAGING_RANDOM_STATE`
- `STAGING_TEST_SIZE`
- `STAGING_S3_ARTIFACTS_PREFIX`
- `STAGING_S3_BATCH_OUTPUTS_PREFIX`
- `STAGING_S3_AUTO_CREATE_BUCKET`
- `STAGING_S3_USE_PATH_STYLE`
- `STAGING_MINIO_API_PORT`
- `STAGING_MINIO_CONSOLE_PORT`
- `STAGING_AIRFLOW_PORT`
- `STAGING_APP_PORT`

### Как работает `.github/workflows/deploy-staging.yml`

Workflow запускается:

- автоматически после успешного `Docker Publish` из ветки `main`
- вручную через `workflow_dispatch`

Что он делает:

1. вычисляет image tag;
2. подключается к staging-серверу по SSH;
3. готовит директорию приложения;
4. копирует `docker-compose.staging.yml` и `airflow/dags`;
5. рендерит `.env.staging` из GitHub secrets и vars;
6. логинится на сервере в `ghcr.io`;
7. выполняет:

```bash
docker compose --env-file .env.staging -f docker-compose.staging.yml pull
docker compose --env-file .env.staging -f docker-compose.staging.yml up -d
```

8. выполняет smoke check:

```bash
curl --fail http://127.0.0.1:<APP_PORT>/ready
```

### Как будет выглядеть deploy path в итоге

1. Push в `main`
2. `CI`
3. `Docker Publish` публикует image в `ghcr.io`
4. `Deploy Staging` берет image по тегу `sha-<shortsha>`
5. staging-сервер поднимает:
   - `init-db`
   - `seed-model`
   - `app`
   - `airflow-standalone`
6. workflow проверяет `/ready`

После этого staging уже живет не от локальной сборки, а от registry image и воспроизводимого deploy flow.

## Что считается успешным запуском

Если всё поднялось правильно:

- `http://127.0.0.1:8000/ready` отвечает JSON вроде:

```json
{"status":"ready","ready":true,"model_loaded":true,"model_name":"LightGBM","postgres_configured":true,"s3_configured":true,"missing_dependencies":[]}
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

Это удобно для детерминированного запуска batch-задач и дальнейшего backfill.

Повторный запуск на тот же `run-date` не скипается:

- batch scoring выполняется заново;
- файлы в S3/MinIO для этой даты перезаписываются;
- записи в Postgres для той же даты сначала заменяются, поэтому дубликаты не создаются.

То есть идемпотентность здесь обеспечивается не пропуском rerun, а безопасным overwrite/update поведением.

## Airflow

Для batch-сервиса в проект добавлен Airflow DAG:

- `airflow/dags/batch_scoring_dag.py`

Что он делает:

- запускает batch scoring по расписанию `@daily`;
- использует `DockerOperator`;
- передаёт в контейнер логическую дату запуска как `{{ ds }}`;
- запускает batch-контейнер в docker-сети `booking_cancellation_default`;
- работает с `Postgres` и `MinIO` через контейнерные адреса `postgres` и `minio`;
- получает секреты и параметры batch-run из `Airflow Variables`;
- поддерживает catchup и backfill.

Текущая структура DAG:

1. `check_batch_not_processed` — логирует, были ли для `run-date` уже записаны outputs, но rerun не блокирует;
2. `run_batch_scoring` — запускает основной batch scoring в контейнере;
3. `verify_batch_outputs` — проверяет, что результаты и `_SUCCESS` marker появились в S3.

### Поднять Airflow локально

Теперь отдельный старт Airflow не обязателен: стандартный локальный bootstrap уже поднимает `Postgres`, `MinIO` и `Airflow` одной командой:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\bootstrap_local.ps1
```

Why `5433` instead of `5432`:
This avoids accidentally connecting host-side Python commands to some other local Postgres already listening on `localhost:5432`. The app on your machine should talk to Docker Postgres through `localhost:5433`, while Airflow tasks inside Docker still use the internal Docker hostname `postgres:5432`.

Если нужен только Airflow без повторной инициализации БД и без обучения модели, можно использовать отдельный скрипт:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\bootstrap_airflow.ps1
```

Он тоже сначала попытается использовать локальный image tar, а к сборке перейдёт только с флагом `-AllowBuild`.

После старта:

- Airflow UI: `http://127.0.0.1:8081`
- логин: значения `AIRFLOW_ADMIN_USERNAME` / `AIRFLOW_ADMIN_PASSWORD` из `.env`

### Variables в Airflow UI

Для работы DAG секреты теперь задаются через `Airflow UI`:

1. Откройте `Admin -> Variables`
2. Создайте переменные:
   - `POSTGRES_DB`
   - `POSTGRES_USER`
   - `POSTGRES_PASSWORD`
   - `S3_BUCKET`
   - `S3_ACCESS_KEY`
   - `S3_SECRET_KEY`
3. Рекомендуемые дополнительные переменные:
   - `S3_REGION`
   - `S3_ARTIFACTS_PREFIX`
   - `S3_BATCH_OUTPUTS_PREFIX`
   - `S3_AUTO_CREATE_BUCKET`
   - `S3_USE_PATH_STYLE`
   - `DEFAULT_BATCH_RISK_SHARE`

Минимальные значения для локального запуска:

- `POSTGRES_DB=booking_cancellation`
- `POSTGRES_USER=postgres`
- `POSTGRES_PASSWORD=<ваш пароль из .env>`
- `S3_BUCKET=booking-cancellation-artifacts`
- `S3_ACCESS_KEY=booking_minio`
- `S3_SECRET_KEY=<ваш S3 пароль из .env>`
- `S3_REGION=us-east-1`
- `S3_ARTIFACTS_PREFIX=artifacts`
- `S3_BATCH_OUTPUTS_PREFIX=batch-runs`
- `S3_AUTO_CREATE_BUCKET=true`
- `S3_USE_PATH_STYLE=true`
- `DEFAULT_BATCH_RISK_SHARE=0.3`

### Backfill

У DAG включён `catchup=True`, поэтому прошлые логические даты можно дообработать.

Пример backfill за три дня:

```powershell
docker compose -f docker-compose.local.yml exec airflow-standalone airflow dags backfill booking_batch_scoring --start-date 2026-04-14 --end-date 2026-04-16
```

Идемпотентность обеспечивается двумя слоями:

- output paths partitioned by `run_date`;
- rerun на ту же дату перезаписывает те же файлы в S3;
- перед записью в Postgres старые записи для того же `run_date` удаляются и заменяются новыми, поэтому дубликаты не появляются.

## API

- `GET /health` — простой liveness и факт загрузки модели;
- `GET /ready` — readiness для deploy/smoke-check: конфиг присутствует, модель загружена, сервис готов принимать трафик;
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

### Быстрый production-like smoke test

Для проверки всего пути `postgres -> minio -> init-db -> seed-model -> app` одной командой:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\smoke_prod_stack.ps1
```

Скрипт:

- создает временный `.env.smoke` из `.env.prod.example`;
- подставляет тестовые секреты;
- запускает `docker-compose.prod.yml` в отдельном compose project;
- использует отдельные host-порты, чтобы не конфликтовать с вашим обычным локальным стеком;
- ждет `GET /ready`;
- проверяет `GET /frontend-api/health` и главную страницу `/`;
- печатает итоговый статус и по умолчанию удаляет smoke-окружение после проверки.

Если хотите оставить стек поднятым и походить по UI руками:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\smoke_prod_stack.ps1 -KeepStack
```

Тогда после успешного запуска проверяйте:

- `http://127.0.0.1:8010/ready`
- `http://127.0.0.1:8010/docs`
- `http://127.0.0.1:8010/`
- `http://127.0.0.1:9101`

Потом прибрать это окружение можно одной командой:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\cleanup_smoke.ps1
```

## Docker

`Dockerfile` нужен для контейнеризации приложения: он собирает воспроизводимый образ с Python, зависимостями и кодом сервиса.

`docker-compose.local.yml` нужен для удобного локального старта всей инфраструктуры:

- `Postgres`
- `MinIO`
- `Airflow`

Это позволяет поднимать проект и batch-окружение одной командой.

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
5. проверить `GET /ready`.
