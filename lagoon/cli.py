"""Command line interface for LAGOON.
"""

import typer

app = typer.Typer()

from .db.cli import app as db_app
app.add_typer(db_app, name='db')

@app.command()
def hello():
    print('hello')


if __name__ == '__main__':
    app()

