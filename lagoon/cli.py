"""Command line interface for LAGOON.
"""

import typer

app = typer.Typer()

@app.command()
def hello():
    print('hello')


if __name__ == '__main__':
    app()

