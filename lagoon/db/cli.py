
import typer

app = typer.Typer()

@app.command()
def down():
    """Shutdown postgres as needed.
    """


@app.command()
def up():
    """Launch a docker instance of postgres; ensures it's up to data with latest
    schema. Also checks env vars required.
    """

if __name__ == '__main__':
    app()

