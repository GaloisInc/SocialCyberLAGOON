
from lagoon.config import get_config

import asyncio
import contextlib
import sqlalchemy.orm
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy_utils import database_exists, create_database
import threading

_lock = threading.Lock()
_engine = None
_sessionmaker = None
def get_engine(*, admin=False, kwargs=None):
    """Returns the sqlalchemy.Engine instance.

    Args:
        kwargs: Passed to `sqlalchemy.create_engine`. Only for development tools.
    """
    global _engine, _sessionmaker

    cfg = get_config()

    if kwargs is not None:
        assert _engine is None, 'Cannot specify kwargs after engine created. For dev only'
    else:
        kwargs = {}

    if admin:
        assert _engine is None, 'Cannot specify admin after engine created. Dev only'

    with _lock:
        if _engine is None:
            u = cfg['db']['user']
            p = cfg['db']['password']
            h = cfg['db']['host']
            po = cfg['db']['port']
            db = cfg['db']['db']
            if admin:
                db = 'postgres'
            _engine = sqlalchemy.create_engine(f'postgresql://{u}:{p}@{h}:{po}/{db}',
                    # SQLAlchemy 2.0 https://docs.sqlalchemy.org/en/14/core/engines.html#sqlalchemy.create_engine.params.future
                    future=True,
                    **kwargs)

            _sessionmaker = sqlalchemy.orm.sessionmaker(_engine)
    return _engine


def get_session():
    """Returns a sqlalchemy.orm.Session object, to be used in a context manager.

    Autocommit is on, meaning that the session will be committed if no error is
    raised.
    """
    if _engine is None:
        # Ensure engine exists
        get_engine()
    return _sessionmaker.begin()


_engineaio = None
_lockaio = asyncio.Lock()
_sessionmakeraio = None
async def get_engine_async():
    """Returns an asynchronous engine.
    """
    global _engineaio, _sessionmakeraio
    async with _lockaio:
        if _engineaio is None:
            cfg = get_config()
            u = cfg['db']['user']
            p = cfg['db']['password']
            h = cfg['db']['host']
            po = cfg['db']['port']
            db = cfg['db']['db']
            _engineaio = create_async_engine(
                    f'postgresql+asyncpg://{u}:{p}@{h}:{po}/{db}',
                    future=True)
            _sessionmakeraio = sqlalchemy.orm.sessionmaker(_engineaio,
                    class_=AsyncSession)
    return _engineaio


@contextlib.asynccontextmanager
async def get_session_async():
    """Returns a sqlalchemy.ext.asyncio.AsyncSession manager for use in a
    context manager.
    """
    if _engineaio is None:
        await get_engine_async()
    async with _sessionmakeraio() as session:
        async with session.begin():
            yield session

