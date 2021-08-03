"""Import data from an OCEAN Pickle file.
"""

from pathlib import Path
import typer

app = typer.Typer()

@app.command()
def load(path: Path = typer.Argument(..., exists=True, file_okay=True)):
    from lagoon.ingest.ocean_pickle import load_pickle
    load_pickle(path)


if __name__ == '__main__':
    app()

