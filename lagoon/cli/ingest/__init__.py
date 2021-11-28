
import typer
app = typer.Typer()

from .git import app as git_app
app.add_typer(git_app, name='git')

from .ocean_pickle import app as ocean_pickle_app
app.add_typer(ocean_pickle_app, name='ocean_pickle')

from .python_peps import app as python_peps_app
app.add_typer(python_peps_app, name='python_peps')

from .toxicity_badwords import app as toxicity_badwords_app
app.add_typer(toxicity_badwords_app, name='toxicity_badwords')

if __name__ == '__main__':
    app()

