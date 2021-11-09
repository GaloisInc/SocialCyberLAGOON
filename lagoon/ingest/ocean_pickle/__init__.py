"""Import data from an OCEAN .pck file.

Data graph (entities are nodes, observations are labeled edges):

.. mermaid::

    flowchart LR
    person[person<br/><div style='text-align:left'>+name<br/>+email</div>]
    message[message<br/><div style='text-align:left'>+subject<br/>+body_text<br/>+origin_filename<br/>+time<br/> \
        +flagged_abuse<br/> \
        +badwords_ex_googleInstantB_any<br/> \
        +badwords_ex_mrezvan94Harassment_Sexual<br/> \
        +badwords_ex_mrezvan94Harassment_Racial<br/> \
        +badwords_ex_mrezvan94Harassment_Appearance<br/> \
        +badwords_ex_mrezvan94Harassment_Intelligence<br/> \
        +badwords_ex_mrezvan94Harassment_Politics<br/> \
        +badwords_ex_mrezvan94Harassment_Generic<br/> \
        +badwords_ex_swearing_any \
        +computed_badwords_googleInstantB_any<br/> \
        +computed_badwords_mrezvan94Harassment_Sexual<br/> \
        +computed_badwords_mrezvan94Harassment_Racial<br/> \
        +computed_badwords_mrezvan94Harassment_Appearance<br/> \
        +computed_badwords_mrezvan94Harassment_Intelligence<br/> \
        +computed_badwords_mrezvan94Harassment_Politics<br/> \
        +computed_badwords_mrezvan94Harassment_Generic<br/> \
        +computed_badwords_swearing_any</div>]
    person -- message_from --> message
    message -- message_to --> person
    message -- message_cc --> person
    message -- message_ref --> message
"""

from lagoon.db.connection import get_session
import lagoon.db.schema as sch
from lagoon.ingest.util import clean_for_ingest

import arrow
import collections
import csv
import multiprocessing
import os
from pathlib import Path
import pickle
import re
import sqlalchemy as sa
import tqdm

_file = os.path.dirname(os.path.abspath(__file__))

def load_pickle(path: Path):
    """Ingests an OCEAN pickle file.
    """
    # Before investing significant time processing, ensure server is up
    with get_session() as sess:
        _b = sess.execute(sa.select(sch.Batch).limit(1))

    # Load bad words
    bad_words = {
            'googleInstantB': _load_word_list(os.path.join(_file,
                'badwords_googleInstantBCensoredWords.txt')),
            'mrezvan94Harassment': _load_harassment_corpus(os.path.join(
                _file, 'badwords_mrezvan94Harassment.csv')),
            'swearing': _load_word_list(os.path.join(_file,
                'badwords_swearing.txt')),
    }

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
    def db_get_message(id, origin=None):
        r = entities['message'].get(id)
        if r is None:
            r = entities['message'][id] = sch.Entity(name=f'Message {id}',
                    type=sch.EntityTypeEnum.message, attrs={})
        if origin is not None:
            r.attrs['origin_filename'] = origin
        return r
    def db_get_user(name, email):
        id = f'{name} <{email}>'
        r = entities['user'].get(id)
        if r is None:
            r = entities['user'][id] = sch.Entity(name=id,
                    type=sch.EntityTypeEnum.person,
                    attrs={
                        'name': name,
                        'email': email,
                    })
        return r

    # Processing bad words takes significant time. Speed up via parallelism.
    exit_flag = multiprocessing.Value('i', 0)
    badwords_queue_written = 0
    badwords_queue_in = multiprocessing.Queue()
    badwords_queue_out = multiprocessing.Queue()
    badwords_procs = [
            multiprocessing.Process(target=_proc_badword, args=(exit_flag,
                badwords_queue_in, badwords_queue_out, bad_words))
            for _ in range(os.cpu_count())]
    [p.start() for p in badwords_procs]

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
            message_time = _date_field_resolve(m['date'], m['raw_date_string'])
        except:
            raise ValueError(f"Bad date: {m['message_id']} {m['date']} {m['raw_date_string']}")

        message = db_get_message(m['message_id'], origin=m['filename'])
        message.attrs['subject'] = m['subject']
        message.attrs['body_text'] = m['body_text']
        message.attrs['flagged_abuse'] = m['flagged_abuse']
        message.attrs['time'] = message_time.timestamp()  # float for JSON

        # Some ocean data has \x00 bytes... remove those
        message.attrs = {k: v if not isinstance(v, str) else v.replace('\x00', '<NULL>')
                for k, v in message.attrs.items()}

        badwords_queue_in.put((m['message_id'], m['body_text']))
        badwords_queue_written += 1

        if frm is not None:
            message.obs_as_dst.append(sch.Observation(src=frm,
                type=sch.ObservationTypeEnum.message_from,
                time=message_time))
        if to is not None:
            message.obs_as_src.append(sch.Observation(dst=to,
                type=sch.ObservationTypeEnum.message_to,
                time=message_time))
        if cc is not None:
            message.obs_as_src.append(sch.Observation(dst=cc,
                type=sch.ObservationTypeEnum.message_cc,
                time=message_time))
        for r in m['refs']:
            message.obs_as_src.append(sch.Observation(
                dst=db_get_message(r['ref']),
                type=sch.ObservationTypeEnum.message_ref,
                time=message_time))

    # Wait for processing to finish
    exit_flag.value = 1
    exited = 0
    progress_out = tqdm.tqdm(desc='Waiting for toxicity processing to finish',
            total=badwords_queue_written)
    while exited != len(badwords_procs):
        try:
            msg_id, msg_attrs = badwords_queue_out.get(timeout=1)
        except multiprocessing.queues.Empty:
            break
        if msg_id == '<<DONE>>':
            exited += 1
            continue

        progress_out.update(1)
        msg = db_get_message(msg_id)
        msg.attrs.update(msg_attrs)
    progress_out.close()
    [p.join() for p in badwords_procs]
    print('Writing to database...')

    with get_session() as sess:
        clean_for_ingest(sess)

        resource = f'ocean-{os.path.basename(path)}'
        sch.Batch.cls_reset_resource(resource, session=sess)

        batch = sch.Batch(resource=resource)
        for egroup, edict in entities.items():
            for ename, e in edict.items():
                batch.entities.append(e)
                for o in e.obs_as_src:
                    batch.observations.append(o)
                for o in e.obs_as_dst:
                    batch.observations.append(o)
        sess.add(batch)

        sess.flush()
        print(f'Finished with batch {batch.id}')


def _date_field_resolve(*dates):
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


def _load_word_list(fpath):
    """Returns a regex representing all words in file.
    """
    with open(fpath) as f:
        words = f.read().split('\n')
    r = _word_list_to_re(words)
    return {'any': r}


def _load_harassment_corpus(fpath):
    """From https://github.com/Mrezvan94/Harassment-Corpus, per Laurent

    Tied to https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0240924
    """
    csv_reader = csv.reader(open(fpath))
    cols = None
    words = {}
    for row in csv_reader:
        if cols is None:
            cols = row
            for c in cols:
                words[c] = []
            continue

        for col, val in zip(cols, row):
            val = val.lower().strip()
            if not val:
                continue
            words[col].append(val)

    r = {k: _word_list_to_re(v) for k, v in words.items()}
    return r


def _proc_badword(exit_flag, queue_in, queue_out, bad_words):
    while True:
        try:
            msg_id, msg_text = queue_in.get(timeout=1)
        except multiprocessing.queues.Empty:
            if exit_flag.value:
                break
            continue

        result = {}
        for bw_k, bw_res in bad_words.items():
            for k, v in bw_res.items():
                matches = list(v.finditer(msg_text))
                if len(matches) == 0:
                    # Just don't print these. Add a bunch of space to database,
                    # don't give value.
                    continue
                result[f'computed_badwords_{bw_k}_{k}'] = len(matches)
                matches_unique = list(set([m.group(0).lower() for m in matches]))
                result[f'badwords_ex_{bw_k}_{k}'] = matches_unique
        queue_out.put((msg_id, result))

    queue_out.put(('<<DONE>>', None))


def _word_list_to_re(words):
    """Also filters out OK words"""
    words = set([w.lower().strip() for w in words if w.strip()])
    _WORDS_OK = [
            'are',
            'bigger',
            'failure',
            'guido',
            'harder',
            'insertions',
            'native',
            'primitive',
    ]
    words.difference_update(_WORDS_OK)
    words = [re.escape(w) for w in words]

    words = [w.lower() for w in words]
    r = re.compile(r'\b' + r'\b|\b'.join(words) + r'\b', flags=re.I)
    return r

