"""
Import data from Have I Been Pwned.
Two batches can be created -- load_breaches() and load_peps()
"""

import typer

app = typer.Typer()

@app.command()
def load_breaches():
    from lagoon.ingest.hibp import load_hibp
    load_hibp('breaches')

@app.command()
def load_pastes():
    from lagoon.ingest.hibp import load_hibp
    load_hibp('pastes')


if __name__ == '__main__':
    app()
