"""Methods for dealing with LAGOON-wide configuration (DB access)
"""

def get_config():
    """Retrieves global (singleton) config object.

    Retrieves the default config; one day will retrieve the user's desired
    config.
    """
    DEFAULT = {
            'db': {
                'user': 'postgres',
                'password': 'lagoon345',
                'host': 'localhost',
                'port': 9454,
                'db': 'lagoon_db',
            },
            'dev': {
                'name': 'lagoon-dev',
                'path': './deploy/dev',
            },
    }
    return DEFAULT

