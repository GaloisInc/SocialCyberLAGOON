"""Import data from a git repository.
"""

from lagoon.ingest.git import load_git_repo

from pathlib import Path
import typer

app = typer.Typer()

@app.command()
def load(path: Path = typer.Argument(..., exists=True, file_okay=False)):
    load_git_repo(path)


if __name__ == '__main__':
    app()

