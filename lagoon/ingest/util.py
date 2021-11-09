
import lagoon.db.schema as sch

def clean_for_ingest(session):
    """Run this before any new ingest process. Because of how fusion works, we
    must re-process all fusion after an ingest.
    """
    session.query(sch.EntityFusion).delete()
    print(f'Entity fusion cleared -- remember to re-run `lagoon_cli.py fusion run`!')

