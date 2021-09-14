"""Timing tests for data fetching.
"""

from .data import _fetch_batch, _get_list_person

from lagoon.db.connection import get_session
import lagoon.db.schema as sch

import arrow
import time

import contextlib
import io
import pstats
import cProfile
@contextlib.contextmanager
def profiled(callers=False, top=100):
    pr = cProfile.Profile()
    pr.enable()
    yield
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(top)
    # uncomment this to see who's calling what
    if callers:
        ps.print_callers()
    print(s.getvalue())


def main():
    print('getting person list...')
    ent_ids = _get_list_person()

    n = 10
    print(f'starting {n} fetches...')
    with profiled():
        a = time.monotonic()
        for _ in range(n):
            batch = _fetch_batch(ent_ids, 2,
                    time_min=arrow.get('1998-01-01').timestamp(),
                    time_max=arrow.get('2020-12-31').timestamp(),
                    w_size=0.5 * 3600 * 24 * 365.25,
                    batch_size=16)
        b = time.monotonic()
    print(f'Fetching {n}*{len(batch)} took {(b - a)/n:.3f}s/batch')
    print(f'Ignored {_fetch_batch.examples_ignored} rows in last batch')


if __name__ == '__main__':
    main()

