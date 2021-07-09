
from lagoon.config import get_config

import sqlalchemy.orm
from sqlalchemy_utils import database_exists, create_database
import threading

_engine = None
_lock = threading.Lock()
_sessionmaker = None
def get_engine():
    """Returns the sqlalchemy.Engine instance.
    """
    global _engine, _sessionmaker

    cfg = get_config()

    with _lock:
        if _engine is None:
            u = cfg['db']['user']
            p = cfg['db']['password']
            h = cfg['db']['host']
            po = cfg['db']['port']
            db = cfg['db']['db']
            _engine = sqlalchemy.create_engine(f'postgresql://{u}:{p}@{h}:{po}/{db}',
                    # SQLAlchemy 2.0 https://docs.sqlalchemy.org/en/14/core/engines.html#sqlalchemy.create_engine.params.future
                    future=True)

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

