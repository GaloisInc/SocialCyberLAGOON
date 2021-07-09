#! /usr/bin/env python3

"""Script which automatically adds the folder containing this script to
PYTHONPATH and then runs lagoon's CLI.

Note that this file is NOT `lagoon.py` to keep the `lagoon` name referring
to the module.
"""

import os, sys
_path = os.path.dirname(os.path.abspath(__file__))

os.environ['PYTHONPATH'] = _path + ':' + os.environ.get('PYTHONPATH', '')
from lagoon.cli import app
app()

