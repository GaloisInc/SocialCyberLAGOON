
import lagoon.db.connection
import lagoon.db.schema as sch

import sqlalchemy as sa

def main():
    sess = lagoon.db.connection.get_session
    with sess() as s:
        # OCEAN only exists for CPython, so only include those uses.
        b_r = [b.id for b in s.query(sch.Batch)
                .where(sch.Batch.resource.startswith(
                    'ingest-git-github.com/python/cpython'))
                .all()]
        b_m = [b.id for b in s.query(sch.Batch)
                .where(sch.Batch.resource.startswith('ocean'))
                .all()]

        pr = sa.orm.aliased(sch.Entity)
        prf = sa.orm.aliased(sch.EntityFusion)
        pf = sa.orm.aliased(sch.FusedEntity)
        uc = (s.query(pf)
                .select_from(pr)
                .where(pr.batch_id.in_(b_r))
                .where(pr.type == sch.EntityTypeEnum.person)
                .join(prf, prf.id_other == pr.id)
                .join(pf, pf.id == prf.id_lowest)
                .distinct()
                .count())
        #print(f'Unique committers: {uc}')

        # pr = person repository, pm = person mailing list
        pm = sa.orm.aliased(sch.Entity)
        pmf = sa.orm.aliased(sch.EntityFusion)

        q = (s.query(pf)
                .select_from(pr)
                .where(pr.type == sch.EntityTypeEnum.person)
                .where(pr.batch_id.in_(b_r))
                .join(prf, prf.id_other == pr.id)
                .join(pf, prf.id_lowest == pf.id)
                .join(pmf, pmf.id_lowest == pf.id)
                .join(pm, pm.id == pmf.id_other)
                .where(pm.batch_id.in_(b_m))
                .where(pm.type == sch.EntityTypeEnum.person)
                .distinct()
                .count())
        print(f'Matched committers: {q} / {uc} = {q / uc:.4f}')


if __name__ == '__main__':
    main()

