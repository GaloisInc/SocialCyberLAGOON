
from lagoon.config import get_config

import docker
import subprocess
import typer
import os

app = typer.Typer()

DB_VERSION = 'postgres:13'

@app.command()
def up():
    """Spins up a developer instance and all necessary services (like postgres).
    """

    cfg = get_config()
    client = docker.from_env()

    name = _docker_db_name(cfg)

    if name in [c.name for c in client.containers.list()]:
        print('Database already running')
        return

    db_path = os.path.abspath(os.path.join(cfg['dev']['path'], 'db'))
    os.makedirs(db_path, exist_ok=True)

    assert cfg['db']['host'] == 'localhost', cfg['db']['host']
    assert cfg['db']['user'] == 'postgres', cfg['db']['user']
    client.containers.run(DB_VERSION,
            name=cfg['dev']['name'] + '-db',
            detach=True,
            remove=True,
            ports={'5432/tcp': cfg['db']['port']},
            environment={
                'POSTGRES_PASSWORD': cfg['db']['password'],
            },
            volumes={
                db_path: {'bind': '/var/lib/postgresql/data', 'mode': 'rw'},
            },
    )


@app.command()
def down():
    """Shuts down the development services.
    """
    cfg = get_config()
    client = docker.from_env()

    name = _docker_db_name(cfg)
    try:
        c = client.containers.get(name)
    except docker.errors.NotFound:
        print('Container not running?')
    else:
        c.stop()


def _docker_db_name(cfg):
    return cfg['dev']['name'] + '-db'


if __name__ == '__main__':
    app()

