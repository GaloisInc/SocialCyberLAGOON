#! /usr/bin/env python3

"""This script is responsible for going through all "Person" entities, and
reporting the year-by-year breakdown of the number of commits they were involved
with, as well as an estimate of their toxicity based on various word lists.

Used for the 2021-09 report regarding a characterization.
"""

import os
import sys
import typer

app = typer.Typer()

@app.command()
def get_committer_activity():
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from lagoon.db.connection import get_session
    import lagoon.db.schema as sch
    import arrow
    import pandas as pd
    import sqlalchemy as sa
    import tqdm

    E = sch.FusedEntity
    O = sch.FusedObservation

    rows = []

    with get_session() as sess:

        date_min = sess.query(O).order_by(O.time).limit(1).scalar().time
        date_max = sess.query(O).order_by(O.time.desc()).limit(1).scalar().time

        print(f'Scanning from {date_min} to {date_max}', file=sys.stderr)

        query = sess.query(E).where(E.type == sch.EntityTypeEnum.person)
        for p in tqdm.tqdm(query, total=query.count()):
            emails = []
            for fe in p.fusions:
                if not fe.other.attrs['email']:
                    continue
                emails.append(fe.other.attrs['email'])
            obj = {
                'name': p.name,
                'email_list': ','.join(emails),
            }
            rows.append(obj)

            ## Fill in computed_badwords information based on `message_from`
            # relations.
            q = (sess.query(E, O.time)
                    .join(O, E.obs_as_dst)
                    .where(O.src == p)
                    .where(O.type == sch.ObservationTypeEnum.message_from)
                    )
            for msg, msg_t in q:
                year = arrow.get(msg_t).year
                for k, v in msg.attrs.items():
                    if not k.startswith('computed_badwords'):
                        continue

                    rowk = f'{k}_{year}'
                    obj[rowk] = obj.get(rowk, 0) + v

            ## Fill in commit information
            q = (sess.query(O.time.label('time'))
                    .where(O.type.in_([sch.ObservationTypeEnum.created,
                        sch.ObservationTypeEnum.committed]))
                    )
            # OR clause is expensive in postgres...
            q = q.where(O.dst == p).union(q.where(O.src == p))
            q = q.subquery()

            t = sa.func.extract('year', q.c.time)
            q = sess.query(t, sa.func.count()).group_by(t)
            res = q.all()
            for year, count in res:
                obj[f'commits_{int(year)}'] = count

    print(pd.DataFrame(rows).to_csv())


if __name__ == '__main__':
    app()
