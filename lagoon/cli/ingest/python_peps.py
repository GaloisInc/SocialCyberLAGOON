"""
Import data from Python Enhancement Proposals (PEPs).
Two batches created:
    load_peps : Ingests PEPs and their authors (persons), and creates observations to link them
    link_peps: Creates observations to link PEPs with existing messages and git commits which mention that PEP
Path to data not required since PEPs are obtained by scraping the web
"""

import typer

app = typer.Typer()

@app.command()
def load():
    from lagoon.ingest.python_peps import load_peps, link_peps
    load_peps()
    link_peps()


if __name__ == '__main__':
    app()
