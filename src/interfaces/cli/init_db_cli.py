from src.config import settings
from src.infrastructure.db.connection import ensure_database_schema


def main():
    if not settings.postgres_enabled:
        raise ValueError(
            "Postgres is not configured. Fill POSTGRES_* variables in your local environment first."
        )

    ensure_database_schema()
    print("Postgres schema initialized successfully.")


if __name__ == "__main__":
    main()
