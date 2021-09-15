
This is the LAGOON (acronym...) project source code.

# Getting started with development

Note that `./lagoon_cli.py` is a CLI for running common LAGOON functions.

1. Run `pip install -r` on both `requirements*.txt` files to ensure your Python environment has LAGOON's dependencies.
2. Also ensure you have Docker installed.
3. Run `./lagoon_cli.py dev up` to launch an appropriately configured Postgres DB (and any other services required by LAGOON).
4. Either use a pre-populated database or build one from scratch (see two sections below).
5. Run `./lagoon_cli.py ui` to browse around visually.
6. Run `./lagoon_cli.py shell` to interact with the database in a CLI.


## Using a pre-populated database

This method is preferred, as it saves a lot of time.

1. Retrieve a backup of the database, named like `lagoon-db-backup-DATE` in [Google Drive](https://drive.google.com/drive/folders/0AKhGiIfGF_XOUk9PVA).
2. Run `./lagoon_cli.py dev backup-restore path/to/backup` to restore the database.

## Building a database from scratch

1. Run `./lagoon_cli.py db reset` to delete / create / set up the database.
2. Clone e.g. [the CPython repository](https://github.com/python/cpython) somewhere.
3. Run `./lagoon_cli.py ingest git load <path/to/cpython>` to extract information from git into the LAGOON database. This took just under two hours on my laptop.
4. Run `./lagoon_cli.py ingest ocean_pickle load ~/Downloads/python.pck` to extract information from OCEAN data.
5. Run `./lagoon_cli.py fusion run` to fuse entities.

# Documentation

System documentation may be built with the following commands:

```sh
$ cd docs
$ make html
$ open _build/html/index.html
```

# Troubleshooting

## Docker crashes

If the Postgres docker container holding the database crashes, no worries. The actual database files are stored in the folder `../deploy/dev/db`, so as long as that still exists, the database is not truly deleted. If the container crashes, do `docker stop <container_id>`, and then `./lagoon_cli.py dev up` again. May also want to restart VSCode.

## Upgrading versions

Sometimes, the database might get upgraded. To upgrade your database to the latest version, run:

```sh
$ ./lagoon_cli.py alembic -- upgrade head
```

# Postgres: using pgadmin

pgadmin is a popular tool for investigating PostgreSQL installations. To launch
an instance of it pointing at the development database, call:

```sh
$ ./lagoon_cli.py db pgadmin
```

It may take up to a minute to actually open a browser tab.

