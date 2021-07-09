
import os
import subprocess
from typing import List

_path = os.path.dirname(os.path.abspath(__file__))

def main(args: List[str]):
    """Runs the `alembic` command with the required parameters for LAGOON's
    configuration.

    Usually ran as `./lagoon_cli.py alembic -- <args>`

    To see alembic's help, run `./lagoon_cli.py alembic -- --help`

    Most common commands:

        * downgrade -1 : Downgrade the database by one revision.

        * history : Show all revisions.

        * revision --autogenerate : Generate a new revision, automatically
            detecting as many changes as possible. Should be manually checked.

        * upgrade head : Upgrade database to most recent revision.

    To delete an alembic revision, delete the corresponding file in
    `lagoon/db/alembic/versions/`.

    """
    import subprocess
    subprocess.check_call(['alembic',
            '--config', os.path.join(_path, '../db/alembic/alembic.ini'),
            *args
    ])

