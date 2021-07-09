
import typer
app = typer.Typer()

from .git import app as git_app
app.add_typer(git_app, name='git')

if __name__ == '__main__':
    app()

