"""Import data from GitHub.
"""

from pathlib import Path
import typer

app = typer.Typer()

@app.command()
def load_pr_riverloop(path: Path):
    """Ingest a JSON-formatted dump of GitHub pull requests from RiverLoop.
    """
    from lagoon.ingest.github_pr import load_github_pr_riverloop
    load_github_pr_riverloop(path)


if __name__ == '__main__':
    app()

