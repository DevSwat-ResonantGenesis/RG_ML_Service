import os

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.pool import NullPool

from .config import DATABASE_URL


Base = declarative_base()

DB_POOL_SIZE = int(os.getenv("ML_DB_POOL_SIZE", "1"))
DB_MAX_OVERFLOW = int(os.getenv("ML_DB_MAX_OVERFLOW", "0"))
DB_POOL_TIMEOUT = int(os.getenv("ML_DB_POOL_TIMEOUT", "30"))
DB_POOL_RECYCLE = int(os.getenv("ML_DB_POOL_RECYCLE", "1800"))
DB_POOL_CLASS = (os.getenv("ML_DB_POOL_CLASS", "queue").strip().lower() or "queue")

_engine_kwargs = {
    "echo": False,
}

if DB_POOL_CLASS in ("null", "none"):
    _engine_kwargs["poolclass"] = NullPool
else:
    _engine_kwargs.update(
        {
            "pool_size": DB_POOL_SIZE,
            "max_overflow": DB_MAX_OVERFLOW,
            "pool_timeout": DB_POOL_TIMEOUT,
            "pool_recycle": DB_POOL_RECYCLE,
            "pool_pre_ping": True,
            "pool_use_lifo": True,
        }
    )

engine = create_async_engine(DATABASE_URL, **_engine_kwargs)

async_session = sessionmaker(
    engine,
    expire_on_commit=False,
    class_=AsyncSession,
)


async def get_session():
    async with async_session() as session:
        yield session
