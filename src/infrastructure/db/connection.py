from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.config import settings
from src.infrastructure.db.models import Base


def get_engine():
    if not settings.postgres_url:
        raise ValueError("Postgres is not configured. Set POSTGRES_* environment variables.")

    return create_engine(settings.postgres_url, pool_pre_ping=True)


def get_session_factory():
    engine = get_engine()
    return sessionmaker(bind=engine, autoflush=False, autocommit=False)


def ensure_database_schema():
    engine = get_engine()
    Base.metadata.create_all(bind=engine)
