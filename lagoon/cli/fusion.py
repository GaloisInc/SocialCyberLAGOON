
import typer

app = typer.Typer()

@app.command()
def run():
    """Re-runs the entire entity fusion process.
    """
    import lagoon.fusion
    lagoon.fusion.fusion_compute()

