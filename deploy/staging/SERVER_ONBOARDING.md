# Staging Server Onboarding

Этот файл нужен на тот момент, когда у проекта появится первый удаленный staging-сервер.

## Что Это За Сервер

Staging-сервер — это отдельная Linux-машина, похожая на production, но не боевая. На нее GitHub Actions будет выкатывать Docker image из `ghcr.io`.

Минимально подойдет:

- Ubuntu 22.04/24.04 или Debian 12;
- 2 vCPU;
- 4 GB RAM;
- 20+ GB диска;
- открытый SSH-доступ;
- установленный Docker и Docker Compose plugin.

## Что Подготовить На Сервере Один Раз

Под пользователем деплоя:

```bash
mkdir -p /opt/booking-cancellation
mkdir -p /opt/booking-cancellation/airflow/logs
mkdir -p /opt/booking-cancellation/airflow/dags
```

Проверки:

```bash
docker --version
docker compose version
```

Если нужен доступ через firewall, заранее откройте:

- `22` для SSH;
- `8000` для приложения;
- `8081` для Airflow, если он нужен снаружи;
- `9001` для MinIO console, если она нужна снаружи.

## Какие GitHub Secrets Нужны

В `Settings -> Secrets and variables -> Actions`:

- `STAGING_HOST`
- `STAGING_SSH_USER`
- `STAGING_SSH_KEY`
- `STAGING_SSH_PORT`
- `STAGING_APP_DIR`
- `GHCR_READ_USER`
- `GHCR_READ_TOKEN`
- `STAGING_POSTGRES_DB`
- `STAGING_POSTGRES_USER`
- `STAGING_POSTGRES_PASSWORD`
- `STAGING_POSTGRES_PORT`
- `STAGING_POSTGRES_SSLMODE`
- `STAGING_S3_BUCKET`
- `STAGING_S3_ACCESS_KEY`
- `STAGING_S3_SECRET_KEY`
- `STAGING_S3_REGION`
- `STAGING_MINIO_ROOT_USER`
- `STAGING_MINIO_ROOT_PASSWORD`
- `STAGING_AIRFLOW_ADMIN_USERNAME`
- `STAGING_AIRFLOW_ADMIN_PASSWORD`

## Какие GitHub Vars Полезно Завести

В environment `staging`:

- `STAGING_DEFAULT_HIGH_RISK_THRESHOLD=0.7`
- `STAGING_DEFAULT_BATCH_RISK_SHARE=0.3`
- `STAGING_RANDOM_STATE=42`
- `STAGING_TEST_SIZE=0.2`
- `STAGING_S3_ARTIFACTS_PREFIX=artifacts`
- `STAGING_S3_BATCH_OUTPUTS_PREFIX=batch-runs`
- `STAGING_S3_AUTO_CREATE_BUCKET=true`
- `STAGING_S3_USE_PATH_STYLE=true`
- `STAGING_MINIO_API_PORT=9000`
- `STAGING_MINIO_CONSOLE_PORT=9001`
- `STAGING_AIRFLOW_PORT=8081`
- `STAGING_APP_PORT=8000`

## Как Пройдет Первый Деплой

1. `Docker Publish` публикует image в `ghcr.io`.
2. `Deploy Staging` подключается к серверу по SSH.
3. Workflow копирует `docker-compose.staging.yml` и `airflow/dags`.
4. Workflow рендерит `.env.staging` на сервере.
5. Workflow выполняет:

```bash
docker compose --env-file .env.staging -f docker-compose.staging.yml pull
docker compose --env-file .env.staging -f docker-compose.staging.yml up -d
```

6. После старта workflow проверяет:

```bash
curl --fail http://127.0.0.1:8000/ready
```

## Что Проверить После Первого Деплоя

На сервере:

```bash
cd /opt/booking-cancellation
docker compose --env-file .env.staging -f docker-compose.staging.yml ps
docker compose --env-file .env.staging -f docker-compose.staging.yml logs --tail 100 app
```

Снаружи:

- `http://<server-ip>:8000/ready`
- `http://<server-ip>:8000/docs`
- `http://<server-ip>:8000/`

## Что Уже Можно Проверять Без Сервера

До появления staging-машины локально и в CI уже можно проверять:

- сборку production image;
- запуск `init-db`, `seed-model` и `app`;
- `GET /ready`;
- отдачу встроенного frontend;
- публикацию image в `ghcr.io`;
- корректность staging workflow и compose-файлов.
