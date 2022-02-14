
import lagoon.db.schema as sch

import arrow

def clean_for_ingest(session):
    """Run this before any new ingest process. Because of how fusion works, we
    must re-process all fusion after an ingest.
    """
    session.query(sch.EntityFusion).delete()
    print(f'Entity fusion cleared -- remember to re-run `lagoon_cli.py fusion run`!')


def date_field_resolve(*dates):
    """Try to resolve each value in `dates`, in order. Return the first one that
    resolves correctly.
    """
    for fmt_field in dates:
        for fmt in [None, ['ddd, DD MMM YYYY HH.mm.ss Z']]:
            args = []
            if fmt is not None:
                args.append(fmt)
            try:
                message_time = arrow.get(fmt_field, *args).datetime
            except (TypeError, arrow.parser.ParserError):
                continue
            else:
                return message_time
    else:
        raise ValueError(f"Bad dates: {dates}")

