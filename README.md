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

## API

- `GET /health` - healthcheck и проверка загрузки модели.
- `POST /predict` - предсказание вероятности отмены для одного бронирования.
- `POST /predict/batch` - batch scoring для списка бронирований.