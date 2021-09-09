
This is the LAGOON (acronym...) project source code.

To build the docs:

```sh
$ pip install -r requirements-dev.txt
$ cd docs
$ make html
$ open _build/html/index.html
```

# Getting started with development

Note that `./lagoon_cli.py` is a CLI for running common LAGOON functions.

1. Run `pip install -r` on both `requirements*.txt` files to ensure your Python environment has LAGOON's dependencies.
1. Also ensure you have Docker installed.
2. Run `./lagoon_cli.py dev up` to launch an appropriately configured Postgres DB (and any other services required by LAGOON).
3. (May run `./lagoon_cli.py db pgadmin` to launch PgAdmin pointed at this database)
4. Run `./lagoon_cli.py db reset` to delete / create / set up the database.
5. Clone e.g. [the CPython repository](https://github.com/python/cpython) somewhere.
6. Run `./lagoon_cli.py ingest git load <path/to/cpython>` to extract information from git into the LAGOON database. This took just under two hours on my laptop.
7. Run `./lagoon_cli.py ingest ocean_pickle load ~/Downloads/python.pck` to extract information from OCEAN data.
8. Run `./lagoon_cli.py fusion run` to fuse entities.
9. Run `./lagoon_cli.py ui` to browse around.

# Alternative method to populating the database

Steps 5, 6, 7 and 8 above may take a lot of time and RAM resources. As an alternative, one can populate the whole database from a tarball. Do Steps 1-4 above (3 is required), and then:

5. Save an appropriate tarball of a pre-populated database to your local machine.
6. Run `docker cp <path/to/tarball> <ID of pgadmin docker container>:/var/lib/pgadmin/storage/lagoon_example.com/<something>.tar`
7. Navigate to `localhost:9450`,  expand `lagoon_db` in the left sidebar, right click on `lagoon_db` and click `Restore`.
8. Choose `<something>.tar`.
9. Once done, one can stop the `pgadmin` docker container. The database will stay populated.

# Docker crashes

If the Postgres docker container holding the database crashes, no worries. The actual database files are stored in the folder `../deploy/dev/db`, so as long as that still exists, the database is not truly deleted. If the container crashes, do `docker stop <container_id>`, and then `./lagoon_cli.py dev up` again. May also want to restart VSCode.

## Upgrading versions

Sometimes, the database might get upgraded. To upgrade your database to the latest version, run:

```sh
$ ./lagoon_cli.py alembic -- upgrade head
```

