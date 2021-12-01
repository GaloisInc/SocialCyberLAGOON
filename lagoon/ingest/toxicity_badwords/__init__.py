"""Annotate `EntityTypeEnum.message` and `EntityTypeEnum.git_commit` with
the following attributes:

.. mermaid::

    flowchart LR
    message["message | git_commit<br/><div style='text-align:left'> \
        +badwords_ex_googleInstantB_any<br/> \
        +badwords_ex_mrezvan94Harassment_Sexual<br/> \
        +badwords_ex_mrezvan94Harassment_Racial<br/> \
        +badwords_ex_mrezvan94Harassment_Appearance<br/> \
        +badwords_ex_mrezvan94Harassment_Intelligence<br/> \
        +badwords_ex_mrezvan94Harassment_Politics<br/> \
        +badwords_ex_mrezvan94Harassment_Generic<br/> \
        +badwords_ex_swearing_any<br/> \
        +computed_badwords_googleInstantB_any<br/> \
        +computed_badwords_mrezvan94Harassment_Sexual<br/> \
        +computed_badwords_mrezvan94Harassment_Racial<br/> \
        +computed_badwords_mrezvan94Harassment_Appearance<br/> \
        +computed_badwords_mrezvan94Harassment_Intelligence<br/> \
        +computed_badwords_mrezvan94Harassment_Politics<br/> \
        +computed_badwords_mrezvan94Harassment_Generic<br/> \
        +computed_badwords_swearing_any</div>"]
"""

from lagoon.db.connection import get_session
import lagoon.db.schema as sch

import csv
import multiprocessing
import os
import re
import sqlalchemy as sa
import time
import tqdm

_file = os.path.dirname(os.path.abspath(__file__))

def compute_badwords():
    """Compute `computed_badwords_` toxicity figures.
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

    # Processing bad words takes significant time. Speed up via parallelism.
    exited = 0
    exit_flag = multiprocessing.Value('i', 0)
    badwords_queue_written = 0
    badwords_queue_in = multiprocessing.Queue()
    badwords_queue_out = multiprocessing.Queue()
    badwords_procs = [
            multiprocessing.Process(target=_proc_badword, args=(exit_flag,
                badwords_queue_in, badwords_queue_out, bad_words))
            for _ in range(os.cpu_count())]
    [p.start() for p in badwords_procs]

    with get_session() as sess:
        resource = f'toxicity_badwords'
        sch.Batch.cls_reset_resource(resource, session=sess)
        batch = sch.Batch(resource=resource)
        sess.add(batch)
        sess_added = [0]

        def fetch_badwords_queue_out(timeout=1.):
            nonlocal exited
            try:
                msg_id, msg_attrs = badwords_queue_out.get(timeout=timeout)
            except multiprocessing.queues.Empty:
                return False

            if msg_id == '<<DONE>>':
                exited += 1
            elif msg_attrs:  # Don't save if empty
                sess.add(sch.ComputedAttrs(obj_id=msg_id, batch=batch,
                        attrs=msg_attrs))
                sess_added[0] += 1
                if sess_added[0] >= 10000:
                    sess.flush()
                    sess_added[0] = 0
            return True

        objs = sess.query(sch.Entity).where(sch.Entity.type.in_([
                sch.EntityTypeEnum.message, sch.EntityTypeEnum.git_commit]))
        objs_count = objs.count()
        for obj in tqdm.tqdm(objs, total=objs_count):
            m = obj.attrs
            try:
                if obj.type == sch.EntityTypeEnum.message:
                    text_field = (
                            (m.get('subject') or '')
                            + ' !!! '
                            + (m.get('body_text') or ''))
                elif obj.type == sch.EntityTypeEnum.git_commit:
                    text_field = m['message']
                else:
                    raise NotImplementedError(obj.type)
            except (KeyError, TypeError):
                raise ValueError(f'While looking at {obj.id}: {obj.attrs}')

            badwords_queue_in.put((obj.id, obj.type, text_field))
            badwords_queue_written += 1

            # Clear cache as we can
            while (fetch_badwords_queue_out(timeout=1e-3)
                    or badwords_queue_in.qsize() > 100):
                pass

        # Wait for processing to finish
        exit_flag.value = 1
        while exited != len(badwords_procs):
            fetch_badwords_queue_out()

        [p.join() for p in badwords_procs]

        sess.flush()
        print(f'Finished with batch {batch.id}')


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


def _message_split(message: str):
    """OCEAN messages have all sorts of quoted toxicity. To begin dealing with
    this, this function splits the message into 3 parts:

        'body': The main body of the message
        'quote': Quoted parts of the message
        'sign': Signature parts of the message
    """
    lines = {'body': [], 'quote': [], 'sign': []}
    last_line = None
    in_sign = False
    for line in message.split('\n'):
        line = line.strip()
        line_no_quote = re.sub(r'^(> ?)+', '', line)
        if re.search(r'^(On .*,.*wrote|Le [a-z]+\..*crit ):$',
                line_no_quote, flags=re.I) is not None:
            # Headers which should be skipped
            continue
        if len(lines['body']) == 0 and re.search(r'(schrieb|wrote):$', line, flags=re.I) is not None:
            # Another header
            continue
        if line.startswith('>'):
            # Note -- order here matters! We do quotes before signature / body as
            # some quotes occur after the signature.
            lines['quote'].append(line)
            continue

        # Strip off things that look like a footer
        if re.search(r'^(best|best wishes|cheers|regards|thank you|thanks|--),?$', line,
                flags=re.I) is not None:
            in_sign = True

        if in_sign:
            lines['sign'].append(line)
        else:
            lines['body'].append(line)

    return {k: '\n'.join(v).strip() for k, v in lines.items()}


def _proc_badword(exit_flag, queue_in, queue_out, bad_words):
    while True:
        try:
            msg_id, msg_type, msg_text = queue_in.get(timeout=1)
        except multiprocessing.queues.Empty:
            if exit_flag.value:
                break
            continue

        msg_parts = {'body': msg_text}
        if msg_type == sch.EntityTypeEnum.message:
            msg_parts = _message_split(msg_text)

        result = {}
        for msg_part, msg_text in msg_parts.items():
            for bw_k, bw_res in bad_words.items():
                for k, v in bw_res.items():
                    matches = list(v.finditer(msg_text))
                    if len(matches) == 0:
                        # Just don't print these. Add a bunch of space to database,
                        # don't give value.
                        continue
                    result[f'computed_badwords_{msg_part}_{bw_k}_{k}'] = len(matches)
                    matches_unique = list(set([m.group(0).lower() for m in matches]))
                    result[f'badwords_ex_{msg_part}_{bw_k}_{k}'] = matches_unique
        queue_out.put((msg_id, result))

    queue_out.put(('<<DONE>>', None))


def _word_list_to_re(words):
    """Also filters out OK words"""
    words = set([w.lower().strip() for w in words if w.strip()])
    _WORDS_OK = [
            'alla',
            'are',
            'asian',
            'backdoor',
            'bigger',
            'black',
            'colored',
            'destroy',
            'failure',
            'guido',
            'harder',
            'insertions',
            'native',
            'primitive',
    ]
    words.difference_update(_WORDS_OK)
    words = [re.escape(w) for w in words]
    r = re.compile(r'\b' + r'\b|\b'.join(words) + r'\b', flags=re.I)
    return r

