
import typer

app = typer.Typer()

@app.command()
def check():
    """Look at entity fusion results, with special attention to the top
    most-fused entities.
    """
    import os, subprocess
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
            '..', 'fusion', 'check.py')
    subprocess.run(['streamlit', 'run', script])


@app.command()
def recache():
    """Only run the `FusedEntity` / `FusedObservation` re-computation part of
    fusion.
    """
    import lagoon.fusion.fused_cache
    lagoon.fusion.fused_cache.recache()


@app.command()
def run():
    """Re-runs the entire entity fusion process, including recache.
    """
    import lagoon.fusion
    lagoon.fusion.fusion_compute()
    import lagoon.fusion.fused_cache
    lagoon.fusion.fused_cache.recache()

