'''Ingest PEP-authorship correlations into UI to highlight them.

DOES NOT CLEAR OUT FUSIONS!!! RUN THIS AFTER FUSION AND RUN IT WITH --clean
TO CLEAN UP WHEN DONE.
'''

from lagoon.db.connection import get_session
import lagoon.db.schema as sch

import arrow
import dataclasses
import sqlalchemy as sa
import typer

app = typer.Typer()

@dataclasses.dataclass
class _PersonInfo:
    db_id: int
    local_id: int
    name: str

    db_fork_id: int = -1
    pep_count: int = 0


@app.command()
def main(clean: bool=typer.Option(False)):
    resource = f'oneoffs/20211116-ingest'

    with get_session() as sess:
        (sess.query(sch.EntityFusion)
                .filter(sch.EntityFusion.id_lowest.in_(
                    sa.select(sch.EntityFusion.id_lowest)
                    .join(sch.Entity, sch.Entity.id == sch.EntityFusion.id_lowest)
                    .join(sch.Batch, sch.Batch.id == sch.Entity.batch_id)
                    .where(sch.Batch.resource == resource)
                    .cte()))
                .delete(synchronize_session=False))
        sch.Batch.cls_reset_resource(resource, session=sess)
        if clean:
            # All done!
            return

        peps = (sess.query(sch.FusedEntity)
                .where(sch.FusedEntity.type == sch.EntityTypeEnum.pep)
                .order_by(sch.FusedEntity.name)
                )
        author_db_to_local = {}
        author_info = []  # local -> _PersonInfo
        pep_collabs = []
        pep_dates = []
        def get_author(oid):
            if oid not in author_db_to_local:
                author_db_to_local[oid] = len(author_info)
                author_info.append(_PersonInfo(
                        local_id=len(author_info),
                        db_id=oid,
                        name=sess.query(sch.FusedEntity).where(sch.FusedEntity.id == oid).scalar().name,
                ))
            return author_info[author_db_to_local[oid]]
        print(f'Loading PEPs...')
        for p in peps.all():
            collabs = _pep_group_collabs(sess, p.id)
            collabs = [get_author(i) for i in collabs]
            for c in collabs:
                c.pep_count += 1
            pep_collabs.append(collabs)
            date = arrow.get(p.attrs['created'], 'DD-MMM-YYYY')
            pep_dates.append(date)
        print(f'Loaded {len(pep_collabs)} PEPs')

        # Start loading stuff in the database
        batch = sch.Batch(resource=resource)
        sess.add(batch)

        # Fork authors (prefix with something to make them searchable)
        # author_idx : db idx
        for info in author_info:
            if info.pep_count < 3:
                continue

            a = sch.Entity(type=sch.EntityTypeEnum.person,
                    name='P-' + info.name,
                    attrs={})
            a.batch = batch
            sess.add(a)
            sess.flush()
            info.db_fork_id = a.id

            # Fuse it with itself
            sess.add(sch.EntityFusion(id_lowest=a.id, id_other=a.id,
                    comment='oneoffs/20211116-ingest'))

        # Link authors within each pep
        for date, collabs in zip(pep_dates, pep_collabs):
            if True:
                # Many-to-many, ugly ugly
                for c1 in collabs:
                    for c2 in collabs:
                        if c1 is c2:
                            continue
                        if c1.db_fork_id < 0 or c2.db_fork_id < 0:
                            continue

                        sess.add(sch.Observation(type=sch.ObservationTypeEnum.message_ref,
                                batch=batch,
                                src_id=c1.db_fork_id,
                                dst_id=c2.db_fork_id,
                                time=date.datetime))
            else:
                # Ring, doesn't show anything useful
                col = [c for c in collabs if c.db_fork_id >= 0]
                for c1, c2 in zip(col, col[1:] + [col[0]]):
                    sess.add(sch.Observation(type=sch.ObservationTypeEnum.message_ref,
                            batch=batch,
                            src_id=c1.db_fork_id,
                            dst_id=c2.db_fork_id,
                            time=date.datetime))


def _entity_obs_hops(sess, entity_id, *args, **kwargs):
    ent = sess.query(sch.FusedEntity).where(sch.FusedEntity.id == entity_id).scalar()
    result = ent.obs_hops(*args, **kwargs)

    # Normally, when a session exits, all information is wiped. If we expunge
    # the object first, it isn't. So, expunge all objects we might want to query
    # downstream.
    for r in result:
        for ro in [r, r.src, r.dst]:
            if ro in sess:
                sess.expunge(ro)

    return result


def _pep_group_collabs(sess, p_id):
    collabs = set()
    neighborhood = _entity_obs_hops(sess, p_id, 1)
    for obs in neighborhood:
        if (obs.type.name == sch.ObservationTypeEnum.created.name
                and obs.dst.id == p_id):
            collabs.add(obs.src.id)
        elif (obs.type.name == sch.ObservationTypeEnum.message_ref.name
                and obs.dst.id == p_id):
            talker = (sess.query(sch.FusedObservation)
                    .where(sa.orm.with_parent(obs.src, sch.FusedEntity.obs_as_dst))
                    .where(sch.FusedObservation.type == sch.ObservationTypeEnum.message_from)
                    ).all()
            # Not all have FROM...
            #assert len(talker) >= 1, f'{obs.src.id} -- {repr(talker)}'
            for talk in talker:
                collabs.add(talk.src.id)
    return collabs


if __name__ == '__main__':
    app()

