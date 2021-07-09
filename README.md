
This is the LAGOON (acronym...) project source code.

To build the docs:

```sh
$ pip install -r requirements-dev.txt
$ cd docs
$ make html
$ open _build/index.html
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

