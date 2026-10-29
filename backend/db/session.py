import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# backend/.env — the same file main.py already loads.
ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(ENV_PATH)

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError(
        f"DATABASE_URL is not set. Add it to {ENV_PATH}, e.g.\n"
        "  DATABASE_URL=postgresql+psycopg://user:password@localhost:5432/supplychain"
    )

def _engine_kwargs() -> dict:
    """
    Pool settings for a process that holds a long-running background worker
    alongside request handlers. SQLite (used for local verification) takes none
    of these -- it has no real pool.
    """
    if DATABASE_URL.startswith("sqlite"):
        return {}
    return {
        "pool_size": int(os.getenv("DB_POOL_SIZE", "5")),
        "max_overflow": int(os.getenv("DB_MAX_OVERFLOW", "10")),
        # Don't let a checkout block forever if the pool is exhausted -- fail loudly.
        "pool_timeout": 30,
        # Recycle before typical idle-connection reapers (PgBouncer, cloud PG) cut in.
        "pool_recycle": 1800,
        "pool_pre_ping": True,
    }


engine = create_engine(DATABASE_URL, future=True, **_engine_kwargs())
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def pool_status() -> dict:
    """Pool gauges, surfaced by /health so connection leaks are visible."""
    pool = engine.pool
    stats = {"dialect": engine.dialect.name}
    for attr in ("size", "checkedin", "checkedout", "overflow"):
        getter = getattr(pool, attr, None)
        if callable(getter):
            try:
                stats[attr] = getter()
            except Exception:  # noqa: BLE001 - gauges are best-effort
                pass
    return stats


def get_session():
    """Yield a session and always close it. Usable as a FastAPI dependency later."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
