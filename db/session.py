import os

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

_engine = None
_SessionLocal = None


def get_session() -> Session:
    global _engine, _SessionLocal
    if _SessionLocal is None:
        database_url = os.environ["DATABASE_URL_POOLED"]
        _engine = create_engine(database_url, pool_pre_ping=True)
        _SessionLocal = sessionmaker(bind=_engine)
    return _SessionLocal()
