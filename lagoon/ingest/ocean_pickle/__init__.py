"""Import data from an OCEAN .pck file.

Data graph (entities are nodes, observations are labeled edges):

.. mermaid::

    flowchart LR
    person[person<br/><div style='text-align:left'>+name<br/>+email</div>]
    message[message<br/><div style='text-align:left'>+subject<br/>+body_text<br/>+origin_filename<br/>+time<br/> \
        +flagged_abuse<div/>]
    person -- message_from --> message
    message -- message_to --> person
    message -- message_cc --> person
    message -- message_ref --> message
"""

from lagoon.db.connection import get_session
import lagoon.db.schema as sch
from lagoon.ingest.util import clean_for_ingest, date_field_resolve

import arrow
import collections
import os
from pathlib import Path
import pickle
import re
import sqlalchemy as sa
import tqdm

def load_pickle(path: Path):
    """Ingests an OCEAN pickle file.
    """
    # Before investing significant time processing, ensure server is up
    with get_session() as sess:
        _b = sess.execute(sa.select(sch.Batch).limit(1))

    data = pickle.load(open(path, 'rb'))
    required_cols = ['from_name', 'from_email', 'raw_from_string',
            'to_name', 'to_email', 'raw_to_string',
            'cc_name', 'cc_email', 'raw_cc_string',
            'subject',
            'date', 'raw_date_string',
            'message_id',
            'in_reply_to', 'refs',
            'body_text', 'flagged_abuse',
            'filename',
            # time_stamp is when it was imported... not important for us.
            #'time_stamp',
            ]
    # Will raise error if any columns not found
    data = data[required_cols]

    entities = collections.defaultdict(lambda: {})
    flush_count = [0]
    with get_session() as sess:
        clean_for_ingest(sess)

        resource = f'ocean-{os.path.basename(path)}'
        sch.Batch.cls_reset_resource(resource, session=sess)

        batch = sch.Batch(resource=resource)
        sess.add(batch)

        # The `db_get_*()` functions return the id for the chosen Entity, as an
        # optimization
        def db_get_message(id):
            rid = entities['message'].get(id)
            if rid is None:
                rid = sch.Entity(name=f'Message {id}',
                        type=sch.EntityTypeEnum.message, attrs={},
                        batch=batch)
                sess.add(rid)
                flush_count[0] += 1
                entities['message'][id] = rid
            return rid
        def db_get_user(name, email):
            id = f'{name} <{email}>'
            rid = entities['user'].get(id)
            if rid is None:
                rid = sch.Entity(name=id, type=sch.EntityTypeEnum.person,
                        batch=batch,
                        attrs={
                            'name': name,
                            'email': email,
                        })
                sess.add(rid)
                flush_count[0] += 1
                entities['user'][id] = rid
            return rid

        for m_idx, m in tqdm.tqdm(data.iterrows(), desc='importing messages',
                total=len(data)):

            # No date --> useless
            if m['raw_date_string'] is None:
                continue

            def user_resolve(prefix):
                if m[f'{prefix}_name'] is None:
                    return None
                name = m[f'{prefix}_name']
                email = m[f'{prefix}_email']
                return db_get_user(name, email)
            frm = user_resolve('from')
            to = user_resolve('to')
            cc = user_resolve('cc')

            try:
                message_time = date_field_resolve(m['date'], m['raw_date_string'])
            except:
                raise ValueError(f"Bad date: {m['message_id']} {m['date']} {m['raw_date_string']}")

            def fixnull(v):
                "Some ocean data has \x00 bytes... remove those"
                if not isinstance(v, str):
                    return v
                return v.replace('\x00', '<NULL>')
            message = db_get_message(m['message_id'])
            message.attrs.update({
                'origin_filename': fixnull(m['filename']),
                'subject': fixnull(m['subject']),
                'body_text': fixnull(m['body_text']),
                'flagged_abuse': m['flagged_abuse'],
                'time': message_time.timestamp(),  # float for JSON
            })

            if frm is not None:
                message.obs_as_dst.append(sch.Observation(src=frm, batch=batch,
                    type=sch.ObservationTypeEnum.message_from,
                    time=message_time))
            if to is not None:
                message.obs_as_src.append(sch.Observation(dst=to, batch=batch,
                    type=sch.ObservationTypeEnum.message_to,
                    time=message_time))
            if cc is not None:
                message.obs_as_src.append(sch.Observation(dst=cc, batch=batch,
                    type=sch.ObservationTypeEnum.message_cc,
                    time=message_time))
            for r in m['refs']:
                message.obs_as_src.append(sch.Observation(
                    dst=db_get_message(r['ref']), batch=batch,
                    type=sch.ObservationTypeEnum.message_ref,
                    time=message_time))

            if flush_count[0] > 10000:
                sess.flush()
                flush_count[0] = 0

        print(f'Finished with batch {batch.id}; committing')

