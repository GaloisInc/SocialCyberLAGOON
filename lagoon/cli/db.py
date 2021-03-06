
from lagoon.config import get_config

import os
import shutil
import subprocess
import tempfile
import threading
import time
import typer
import webbrowser

app = typer.Typer()

@app.command()
def pgadmin():
    """Launch pgAdmin pointed at the database. This is the main command used to
    visually debug postgres installations.
    """
    cfg = get_config()

    import docker
    client = docker.from_env()

    host = cfg['db']['host']
    port = cfg['db']['port']
    if host == 'localhost' and cfg.get('dev'):
        print('Resolving docker IP address information...')
        c = client.containers.get(cfg['dev']['name'] + '-db')
        bridge = c.attrs['NetworkSettings']['Networks']['bridge']
        host = bridge['IPAddress']
        # Since we're using the docker container's IP address, we must also use
        # the default postgres port
        port = 5432

    with tempfile.NamedTemporaryFile('w+') as f, \
            tempfile.NamedTemporaryFile('w+', delete=False) as f_exec:
        f.write(rf'''{{"Servers":{{
                "1": {{
                    "Name": "Lagoon DB",
                    "Group": "Server group 1",
                    "Host": "{host}",
                    "Port": {port},
                    "Username": "postgres",
                    "SSLMode": "prefer",
                    "MaintenanceDB": "postgres"
                }}
        }}}}''')
        f.flush()
        os.chmod(f.name, 0o644)

        f_exec.write(rf'''#!/bin/sh

                echo '{host}:{port}:*:postgres:{cfg['db']['password']}' > /tmp/pgpassfile
                chmod 600 /tmp/pgpassfile
                /entrypoint.sh
                ''')
        # Executables must not have an open write pointer; so, we must close
        # this one.
        f_exec.close()

        try:
            os.chmod(f_exec.name, 0o755)

            def open_browser():
                time.sleep(10)
                webbrowser.open_new('http://localhost:9450')
            threading.Thread(target=open_browser, daemon=True).start()

            c = client.containers.run('dpage/pgadmin4',
                    entrypoint='/pgadmin_workaround.sh',
                    detach=True,
                    remove=True,
                    tty=True,
                    ports={'80/tcp': '9450'},
                    environment={
                        'PGADMIN_DEFAULT_EMAIL': 'lagoon@example.com',
                        'PGADMIN_DEFAULT_PASSWORD': 'lagoon',
                        'PGPASSFILE': '/tmp/pgpassfile',
                    },
                    volumes={
                        f.name: {'bind': '/pgadmin4/servers.json', 'mode': 'ro'},
                        f_exec.name: {'bind': '/pgadmin_workaround.sh', 'mode': 'ro'},
                    },
            )

            print(f'Container {c} launched; ctrl+c to exit')
            print(f"UI will open in browser; Username is 'lagoon@example.com'; "
                    f"password is 'lagoon'")
            print(f"TODO fix this; for now, must manually type DB password: '{cfg['db']['password']}'")
            try:
                while True:
                    time.sleep(1)
            finally:
                c.kill()
        finally:
            os.remove(f_exec.name)


@app.command()
def copy_to(name: str):
    """Copies the entire database to a new database with the specified name.
    Useful for testing migrations. See also: :meth:`copy_from`
    """
    import lagoon.config
    db_src = lagoon.config.get_config()['db']['db']
    db_dst = name

    _copy_db(db_src, db_dst)


@app.command()
def copy_from(name: str):
    """Overwrites the current database based on the database with the specified
    name. Useful for testing. See also: :meth:`copy_to`
    """
    import lagoon.config
    db_src = name
    db_dst = lagoon.config.get_config()['db']['db']
    _copy_db(db_src, db_dst)


def _copy_db(db_src, db_dst, no_prompt=False):
    """Destroy `db_dst`, replacing it with a copy of `db_src`.
    """
    import lagoon.db.connection
    import sqlalchemy as sa

    engine = lagoon.db.connection.get_engine(admin=True,
            kwargs={'isolation_level': 'AUTOCOMMIT'})

    with engine.connect() as connection:
        def terminate_backends(db):
            text = sa.text(f'''
                SELECT pg_terminate_backend(pg_stat_activity.pid)
                FROM pg_stat_activity
                WHERE pg_stat_activity.datname = :database
                AND pid <> pg_backend_pid();
                ''')
            connection.execute(text, {'database': db})

        cur = connection.execute(
                sa.text('''SELECT datname FROM pg_database WHERE datname = :database'''),
                {'database': db_dst})
        results = cur.all()
        if results:
            if not no_prompt:
                sure = input(f'Database {db_dst} exists; proceeding will delete it! (y/N) ')
                if sure.lower() != 'y':
                    print('Aborting.')
                    return

            terminate_backends(db_dst)
            connection.execute(sa.text(f'''DROP DATABASE {db_dst}'''))

        terminate_backends(db_src)
        connection.execute(
                sa.text(f'''CREATE DATABASE {db_dst} WITH TEMPLATE {db_src}'''))


@app.command()
def reset(reset_alembic: bool=typer.Option(False),
        new_revision: bool=typer.Option(False)):
    """Deletes entire database and reloads from the beginning, with the latest
    alembic version.

    Args:
        reset_alembic: If True, force checkout of the alembic/versions
                directory. Useful for regenerating the DB at a state before the
                current commit.
        new_revision: Requires reset_alembic. If True, also create a new alembic
                revision automatically.
    """
    # Delayed import to keep CLI fast
    import lagoon.db.connection
    engine = lagoon.db.connection.get_engine()
    path = os.path.dirname(os.path.abspath(__file__))
    from sqlalchemy_utils import database_exists, drop_database, create_database

    if database_exists(engine.url):
        print(f'Considering reset of: {engine.url.render_as_string()}')
        sure = input('Are you sure? This will purge everything. (y/N) ')
        if sure.lower() != 'y':
            print('Aborting.')
            return

        print('Resetting.')
        drop_database(engine.url)
    else:
        print('Creating.')

    alembic_versions = os.path.join(path, '../db/alembic/versions')
    if reset_alembic:
        if os.path.lexists(alembic_versions):
            shutil.rmtree(alembic_versions)
        subprocess.call(['git', 'checkout', '--', alembic_versions])
    else:
        assert not new_revision, 'new_revision requires reset_alembic'

    create_database(engine.url)

    # Next alembic commands require this folder to exist, even if empty
    os.makedirs(alembic_versions, exist_ok=True)
    subprocess.check_call(['alembic',
            '--config', os.path.join(path, '../db/alembic/alembic.ini'),
            'upgrade', 'head'])

    if new_revision:
        # Now that database is at latest version, let alembic generate a diff.
        subprocess.check_call(['alembic',
                '--config', os.path.join(path, '../db/alembic/alembic.ini'),
                'revision', '--autogenerate'])
        subprocess.check_call(['alembic',
                '--config', os.path.join(path, '../db/alembic/alembic.ini'),
                'upgrade', 'head'])


if __name__ == '__main__':
    app()

