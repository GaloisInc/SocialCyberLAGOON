"""Main command line interface for LAGOON.

This tool consists of many sub-commands; see below modules (or commandline
`--help`) for more information.
"""

import typer

app = typer.Typer()

from .alembic import main as alembic_cmd
app.command('alembic')(alembic_cmd)

from .db import app as db_app
app.add_typer(db_app, name='db')

from .dev import app as dev_app
app.add_typer(dev_app, name='dev')

from .ingest import app as ingest_app
app.add_typer(ingest_app, name='ingest')

@app.command()
def hello():
    "Prints hello"
    print('hello')


@app.command()
def shell():
    """Opens a shell with DB access.
    """
    from lagoon.db.connection import get_session
    import lagoon.db.schema as sch
    import sqlalchemy as sa

    print(f'Can get db session as `get_session()` / schema module as `sch` / sqlalchemy as `sa`')
    print(f'...Try something like `s = get_session().__enter__(); s.execute(sa.select(sch.Entity).limit(1)).scalar()`')

    import IPython; IPython.embed()

