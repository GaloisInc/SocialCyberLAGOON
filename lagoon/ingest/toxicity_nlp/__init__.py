"""Annotate `EntityTypeEnum.message` and `EntityTypeEnum.git_commit` with
the following attributes:

.. mermaid::

    flowchart LR
    message["message | git_commit<br/><div style='text-align:left'> \
        +toxicity_nlp_classification<br/> \
        +toxicity_nlp_regression</div>"]
"""

from lagoon.ml.config import *
from lagoon.ingest.toxicity_badwords import _message_split
from lagoon.db.connection import get_session
from lagoon.db import schema as sch
import sqlalchemy as sa

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

import os
from tqdm import tqdm
import math


def process_attr_value(val) -> str:
    if val is None:
        return ''
    return str(val)

def process_message(obj):
    return _message_split(process_attr_value(obj.attrs.get('body_text','')))['body']

def process_commit(obj):
    return _message_split(process_attr_value(obj.attrs.get('message','')))['body']

def process_pr(obj):
    raise NotImplementedError


def compute_toxicity_nlp(chunk_size=128, low_precision=True):
    """
    chunk_size: How many nodes the transformer model will process at a time.
        Limited by device memory size.
    low_precision: If True, use int32 instead of int64, and float16 instead of float32.
        Note that float32 for transformers is only possible on GPU.
    """
    if low_precision and not torch.cuda.is_available():
        print("""WARNING: Cannot use low precision transformer computations when on CPU due to the error `"LayerNormKernelImpl" not implemented for 'Half'`. Low precision model will be disabled.""")
        low_precision = False
    
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased") # do not change this
    
    model_clf = AutoModelForSequenceClassification.from_pretrained(os.path.join(NLP_MODELS_FOLDER,'tox_classifier'), num_labels=2)
    if low_precision:
        model_clf.to(torch.float16)
    model_clf.to(DEVICE)
    model_clf.eval()

    model_reg = AutoModelForSequenceClassification.from_pretrained(os.path.join(NLP_MODELS_FOLDER,'tox_regression'), num_labels=1)
    if low_precision:
        model_reg.to(torch.float16)
    model_reg.to(DEVICE)
    model_reg.eval()

    def _run_type(objs, process_fn):
        """ Process all objects of a single EntityType """
        objs_chunk = []

        def _run_chunk():
            """ Process all objects in a chunk """
            if not objs_chunk:
                return

            strings_chunk = [process_fn(obj) for obj in objs_chunk]
            ids_chunk = [obj.id for obj in objs_chunk]
            assert len(ids_chunk) == len(set(ids_chunk)), f'Found duplicate IDs in {ids_chunk}'

            with torch.no_grad():
                encodings = tokenizer(strings_chunk, truncation=True, padding=True, return_tensors='pt')
                #NOTE: We have to ignore longer messages by setting truncation=True, otherwise it throws an error
                # padding=True is required to get a rectangular tensor
                if low_precision:
                    for k in encodings:
                        encodings[k] = encodings[k].to(torch.int32)
                encodings = encodings.to(DEVICE)

                result = model_clf(**encodings)
                result_softmax = torch.nn.functional.softmax(result.logits, dim=-1) # (not toxic, toxic)
                assert result_softmax.shape[0] == len(ids_chunk), f'IDs chunk has length {len(ids_chunk)}, softmax result has length {result_softmax.shape[0]}'

                result = model_reg(**encodings)
                result_sigmoid = torch.sigmoid(result.logits) # higher indicates toxicity
                assert result_sigmoid.shape[0] == len(ids_chunk), f'IDs chunk has length {len(ids_chunk)}, sigmoid result has length {result_sigmoid.shape[0]}'

            for i in range(len(ids_chunk)):
                sess.add(sch.ComputedAttrs(
                    obj_id = ids_chunk[i],
                    batch = batch,
                    attrs = {
                        'toxicity_nlp_classification': result_softmax[i].tolist(),
                        'toxicity_nlp_regression': result_sigmoid[i].item()
                    }
                ))
            sess.flush()

            objs_chunk.clear()

        objs_count = objs.count()
        with tqdm(total = objs_count) as pbar:
            for obj in objs.yield_per(128): # `arg` in objs.yield_per(arg) can be anything, it just manages memory by allocating memory for a certain number of objects to be yielded at a time instead of the whole `objs`` query. It is independent of `chunk_size``.
                objs_chunk.append(obj)
                if len(objs_chunk) >= chunk_size:
                    _run_chunk()
                    pbar.update(chunk_size)
            _run_chunk() # for the last non-fully-filled chunk
            pbar.update(objs_count%chunk_size)

    with get_session() as sess:
        # Before investing significant time processing, ensure server is up
        _b = sess.execute(sa.select(sch.Batch).limit(1))

        resource = 'toxicity_nlp'
        sch.Batch.cls_reset_resource(resource, session=sess)
        batch = sch.Batch(resource=resource)
        sess.add(batch)
        print('Computing NLP toxicity attrbutes for ...')

        ## Messages
        print('messages ...')
        objs = sess.query(sch.Entity).where(sch.Entity.type == sch.EntityTypeEnum.message).order_by(sch.Entity.id)
        _run_type(objs, process_message)

        ## Commits
        print('commits ...')
        objs = sess.query(sch.Entity).where(sch.Entity.type == sch.EntityTypeEnum.git_commit).order_by(sch.Entity.id)
        _run_type(objs, process_commit)

        sess.flush()
        print(f'Finished with batch {batch.id}')
