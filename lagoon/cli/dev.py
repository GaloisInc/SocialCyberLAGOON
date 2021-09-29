"""The difference between `dev` and `db` is that `dev` invokes docker.
"""

from lagoon.config import get_config

import pathlib
import subprocess
import typer
import os

app = typer.Typer()

DB_VERSION = 'postgres:13'

@app.command()
def backup_restore(fpath: pathlib.Path=typer.Argument(..., exists=True)):
    """Restores a DB backup file. See also :meth:`backup_to`.
    """
    import subprocess

    cfg = get_config()
    container = _docker_get_container(cfg)

    sure = input('Are you sure? This will wipe your DB. (y/N) ')
    if sure.lower() != 'y':
        print('Aborting.')
        return

    pg_env = {
            'PGHOST': cfg['db']['host'],
            'PGPORT': '5432',  # cfg['db']['port'],  -- port forwarded
            'PGUSER': cfg['db']['user'],
            'PGPASSWORD': cfg['db']['password'],
            'PATH': os.environ['PATH'],
    }
    print(pg_env)

    print('Restoring.')
    subprocess.check_call(
            ['docker', 'exec', container.id,
                'dropdb', '-U', pg_env['PGUSER'], '-f', cfg['db']['db'],
                '--if-exists'],
            env=pg_env)
    subprocess.check_call(
            ['docker', 'exec', container.id,
                'createdb', '-U', pg_env['PGUSER'],
                '-T', 'template0', cfg['db']['db']],
            env=pg_env)
    # docker-py's API is borked when we need stdin:
    # https://github.com/docker/docker-py/issues/2255
    subprocess.check_call(
            ['docker', 'exec', '-i', container.id,
                'pg_restore', '-U', pg_env['PGUSER'], '-d', cfg['db']['db']],
                #'psql', '-U', pg_env['PGUSER'], cfg['db']['db']],
            stdin=open(fpath, 'rb'),
            env=pg_env)


@app.command()
# Note that `exists=False` doesn't work
def backup_to(fpath: pathlib.Path=typer.Argument(..., file_okay=False, dir_okay=False)):
    """Creates a DB backup file. See also :meth:`backup_restore`.
    """
    cfg = get_config()
    container = _docker_get_container(cfg)

    cmd = ['pg_dump', '-Fc', cfg['db']['db']]
    _, output = container.exec_run(cmd,
            stream=True,
            demux=True,
            environment={
                'PGHOST': cfg['db']['host'],
                'PGPORT': '5432',  # cfg['db']['port'],  -- port forwarded
                'PGUSER': cfg['db']['user'],
                'PGPASSWORD': cfg['db']['password'],
            })

    with open(fpath, 'wb') as f:
        for chunkout, chunkerr in output:
            if chunkout is not None:
                f.write(chunkout)
            if chunkerr is not None:
                print(chunkerr.decode())


@app.command()
def up():
    """Spins up a developer instance and all necessary services (like postgres).
    """
    import docker

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
            name=name,
            detach=True,
            remove=True,
            ports={'5432/tcp': cfg['db']['port']},
            environment={
                'POSTGRES_PASSWORD': cfg['db']['password'],
            },
            volumes={
                db_path: {'bind': '/var/lib/postgresql/data', 'mode': 'rw'},
            },
            # If you start seeing "psycopg2.errors.DiskFull" on multithreaded
            # applications, this is why
            # https://stackoverflow.com/a/56754077/160205
            shm_size='4g',
    )


@app.command()
def down():
    """Shuts down the development services.
    """
    import docker

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


def _docker_get_container(cfg):
    import docker
    client = docker.from_env()
    name = _docker_db_name(cfg)
    container = {c.name: c for c in client.containers.list()}[name]
    return container


if __name__ == '__main__':
    app()

