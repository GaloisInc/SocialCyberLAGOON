"""Add toxicity information based on word lists.
"""

from pathlib import Path
import typer

app = typer.Typer()

@app.command()
def compute():
    """Adds toxicity attributes to any message or git_commit in the database.
    """
    from lagoon.ingest.toxicity_badwords import compute_badwords
    compute_badwords()


if __name__ == '__main__':
    app()

