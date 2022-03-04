"""Add toxicity information based on NLP models.
"""

from pathlib import Path
import typer

app = typer.Typer()

@app.command()
def compute():
    """Adds toxicity NLP attributes to any message or git_commit in the database.
    """
    from lagoon.ingest.toxicity_nlp import compute_toxicity_nlp
    compute_toxicity_nlp()


if __name__ == '__main__':
    app()

