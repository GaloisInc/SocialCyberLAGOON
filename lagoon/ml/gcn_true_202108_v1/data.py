"""Data fetcher -- dynamically (and in parallel) fetch batches of training
data.
"""

from lagoon.db.connection import get_session
import lagoon.db.schema as sch

import arrow
import functools
import itertools
import math
import torch.multiprocessing as multiprocessing
import random
import sqlalchemy as sa
import torch
from typing import Any, List

class BatchManager:
    """Oversees fetching batches for train/testing.

    Args:
        data_badwords: set to `False` to disable badword count information
        data_cheat: Set to `True` to hide the real answer in the graph. Useful for
                checking architectures against data.
        data_type: set to `False` to disable type embeddings.

    Batch groups:
        0 - Train
        1 - Validation
        2 - Test
    """
    def __init__(self, batch_size, embed_size, window_size, hops,
            *, data_badwords=True, data_cheat=False, data_type=True,
            precache=10,
            device='cpu'):
        self._pool = multiprocessing.Pool(initializer=_setup_rng)
        self._entities = _get_list_person()
        self._time_max = _get_max_date().timestamp()
        self._time_min = _get_min_date().timestamp()
        self._w_size = window_size
        self._hops = hops
        self._batch_size = batch_size
        self._embed_size = embed_size
        self._data_badwords = data_badwords
        self._data_cheat = data_cheat
        self._data_type = data_type
        self._device = device

        self._batches_cache = [
                [self._batch_async(group) for _ in range(precache)]
                for group in [0, 1, 2]]


    def count_entity(self):
        return len(self._entities)


    def fetch_batch(self, model_type_embeddings, group_num):
        b = self._batches_cache[group_num]
        batch = self._batch_gather(b)
        b.append(self._batch_async(group_num))

        # Multiprocessing requires us to build sparsity tensor in main process.
        batch_in, batch_out = self._batch_collate(model_type_embeddings, batch)
        return batch_in, batch_out


    def prepare_batch_ins(self, model_type_embeddings, batch_ins):
        """For some inputs, re-assign the embedding portions which have
        gradients attached. Return copied version of batch_ins with these
        replacements performed.
        """
        ins = []
        for b in batch_ins:
            new_row = [v for v in b]
            ins.append(new_row)

            new_row[1] = new_row[1].clone()
            if self._data_type:
                # Anything that can be batched should be assigned on GPU for
                # backprop purposes
                idx_min, idx_max, indices = new_row[-1]['data_ent_type']
                new_row[1][:, idx_min:idx_max] = model_type_embeddings(indices)
        return ins


    def run_windows(self, model, model_type_embeddings,
            entity_id, window_starts: List[Any]):
        """Utility method for running an entity across multiple windows.

        Each window specified is the start time; inputs built around t+window,
        and target outputs computed based on `t+window:t+2*window`.

        Returns:
            tensor[window, (prediction, real_output)] -- always on CPU.
        """
        all_times = [arrow.get(t) for t in window_starts]
        all_data = [
                self._pool.apply_async(_fetch_batch_member_no_session,
                    args=(entity_id, self._hops),
                    kwds={
                        'window_in': t,
                        'window_pt': t.shift(seconds=self._w_size),
                        'window_out': t.shift(seconds=2*self._w_size),
                    })
                for t in all_times]

        # Wait for that to finish
        all_data = [d.get() for d in all_data]

        # Group inputs by batch, send to model
        results = torch.zeros(len(window_starts), 2)
        b = []
        bi = []
        def handle():
            if not b:
                return

            bin, bout = self._batch_collate(model_type_embeddings, b)
            with torch.no_grad():
                bin = self.prepare_batch_ins(model_type_embeddings, bin)
                r = model(bin).cpu()
            bit = torch.tensor(bi, dtype=torch.long)
            results[bit, :1] = r
            results[bit, 1:] = bout

            b.clear()
            bi.clear()
        for di, d in enumerate(all_data):
            if d is None:
                continue
            if len(b) >= self._batch_size:
                handle()
            b.append(d)
            bi.append(di)
        handle()
        return results


    def _batch_async(self, group_num):
        """Asynchronously fetches a batch of the specified type.

        # 90/5/5 splits.
        """
        i_v = len(self._entities) * 9 // 10
        i_t = i_v + len(self._entities) * 5 // 100
        if group_num == 0:
            ents = self._entities[:i_v]
        elif group_num == 1:
            ents = self._entities[i_v:i_t]
        elif group_num == 2:
            ents = self._entities[i_t:]
        else:
            raise NotImplementedError(group_num)
        return self._pool.apply_async(_fetch_batch,
                args=(ents, self._hops, self._time_min, self._time_max,
                    self._w_size, self._batch_size))


    def _batch_collate(self, model_type_embeddings, batch):
        """Given some batch, which is a list of `_fetch_batch_member` results
        that aren't `None`, return `batch_in, batch_out` on the right device.
        """
        batch_in = []
        for b in batch:
            adj_idx, entities, stats = b[0]
            adj_mat = torch.sparse_coo_tensor(adj_idx, torch.ones(len(adj_idx[0])),
                    (len(entities),) * 2)

            # Add identity
            entities_rev = {v[0]: k for k, v in entities.items()}
            ent_props_cpu = torch.zeros(len(entities), self._embed_size)
            ent_types = []

            # Establish slots within each node's embedding
            emb_idx = 0
            idx_type = emb_idx
            emb_idx += model_type_embeddings.embedding_dim
            idx_query = emb_idx
            emb_idx += 1
            idx_badwords = emb_idx
            emb_idx += 1
            idx_cheat = emb_idx
            emb_idx += 1

            assert emb_idx <= self._embed_size, emb_idx

            # Populate embeddings
            for i in range(len(entities)):
                e = entities[entities_rev[i]][1]
                if self._data_type:
                    for ti, t in enumerate(sch.EntityTypeEnum):
                        if e['type'] == t:
                            ent_types.append(ti)
                            break
                    else:
                        raise ValueError(e['type'])

                ent_props_cpu[i, idx_query] = -1
                if i == 0:
                    # Tag the network's "gathering" node
                    ent_props_cpu[i, idx_query] = 1

                ent_props_cpu[i, idx_badwords] = 0
                if self._data_badwords:
                    for k, v in e['attrs'].items():
                        if k.startswith('computed_badwords_'):
                            ent_props_cpu[i, idx_badwords] += v
                if self._data_cheat and i == len(entities) - 1:
                    ent_props_cpu[i, idx_cheat] = b[1][0]

            ent_props = ent_props_cpu.to(self._device)

            if self._data_type:
                # Must delay this assignment since they have a gradient
                # associated
                stats['data_ent_type'] = (idx_type,
                        idx_type + model_type_embeddings.embedding_dim,
                        torch.tensor(ent_types, dtype=torch.long, device=self._device))

            batch_in.append([adj_mat, ent_props, stats])
        batch_out = torch.tensor([b[1] for b in batch])

        # Transfer result tensors to correct device
        batch_in, batch_out = self._to_device(batch_in, batch_out)
        return batch_in, batch_out


    def _batch_gather(self, b):
        """Gather + pop"""
        while True:
            for p in b:
                if p.ready():
                    r = p.get()
                    b.remove(p)
                    return r
            b[0].wait(0.1)


    def _to_device(self, *args):
        if self._device is None:
            return args

        def c(v):
            if isinstance(v, (list, tuple)):
                return [c(vv) for vv in v]
            if not isinstance(v, torch.Tensor):
                return v
            return v.to(self._device)
        return [c(a) for a in args]



def _fetch_batch(entity_id_list, nhops, time_min, time_max, w_size, batch_size):
    """Given a list of available entities, and some time constraints, fetch a
    window of data.
    """
    # For debugging
    _fetch_batch.examples_ignored = 0
    batch = []
    with get_session() as sess:
        while len(batch) < batch_size:
            p = torch.randint(0, len(entity_id_list), ()).item()
            window_pt = time_min + (time_max - time_min) * torch.rand(()).item()
            window_in = window_pt - w_size
            window_out = window_pt + w_size

            window_pt, window_in, window_out = [
                    arrow.get(v)
                    for v in [window_pt, window_in, window_out]]

            p = _fetch_batch_member(sess, entity_id_list[p], nhops, window_in, window_pt,
                    window_out)
            if p is not None:
                batch.append(p)
            else:
                _fetch_batch.examples_ignored += 1
    return batch


def _fetch_batch_member_no_session(*args, **kwargs):
    with get_session() as sess:
        return _fetch_batch_member(sess, *args, **kwargs)


def _fetch_batch_member(sess, entity_id, nhops, window_in: arrow.Arrow,
        window_pt: arrow.Arrow, window_out: arrow.Arrow):
    """Fetch local neighborhood for a given entity id at a given point in
    time.
    """
    # If this person hasn't committed, don't bother tracking them.
    c = sess.query(sa.select(sch.FusedObservation)
            .where(sch.FusedObservation.src_id == entity_id)
            .where(sch.FusedObservation.time >= window_in.datetime)
            .where(sch.FusedObservation.time < window_pt.datetime)
            .join(sch.FusedEntity,
                (sch.FusedEntity.id == sch.FusedObservation.dst_id)
                & (sch.FusedEntity.type == sch.EntityTypeEnum.git_commit))
            .exists()
            ).scalar()
    if not c:
        return

    rec = (sess.query(sch.FusedEntity).where(sch.FusedEntity.id == entity_id)
            .scalar())
    neigh_all = rec.obs_hops(nhops, time_min=window_in.datetime, time_max=window_out.datetime)
    neigh_in = [n for n in neigh_all if arrow.get(n.time) < window_pt]
    neigh_out = [n for n in neigh_all if arrow.get(n.time) >= window_pt]

    # For input, build (adjacency, node props; target == 0)
    entities = {rec.id: (0, rec.asdict())}
    adj_idx = set()
    for o in neigh_in:
        if o.dst_id not in entities:
            entities[o.dst_id] = (len(entities), o.dst.asdict())
        if o.src_id not in entities:
            entities[o.src_id] = (len(entities), o.src.asdict())

        adj_idx.add((entities[o.src_id][0], entities[o.dst_id][0]))
        if True:
            # Symmetric! Ignore directionality
            adj_idx.add((entities[o.dst_id][0], entities[o.src_id][0]))
    for i in range(len(entities)):
        adj_idx.add((i, i))

    # NOTE: cannot return a sparse tensor! Instead, send what's needed for it.
    stats = {'commits': 0, 'computed_badwords': 0, 'ents': len(entities)}
    for _, e in entities.values():
        for k, v in e['attrs'].items():
            if k.startswith('computed_badwords_'):
                stats['computed_badwords'] += v

    # For target, find number of commits in next window
    # Note that just because we have input observations, we are not
    # guaranteed input commits.
    def count(neigh):
        count = 0
        for o in neigh:
            if o.src is not rec:
                continue
            if o.dst.type == sch.EntityTypeEnum.git_commit:
                count += 1
        return count
    n_commit_in = count(neigh_in)
    n_commit_out = count(neigh_out)

    stats['commits'] = n_commit_in
    stats['weight'] = 1 + stats['commits']

    # Prevent a ridiculous objective function
    output_beta = 0.1
    output = (output_beta + n_commit_out) / (output_beta + n_commit_in) - 1
    # Account for asymmetry -- can't get below -1.
    # Could cap at zero, for "no concern" style metric.
    output = min(0, output)

    inputs = (
            # Convert set to torch-expected input
            [[a[0] for a in adj_idx], [a[1] for a in adj_idx]],
            entities,
            stats)
    return inputs, (output,)


@functools.lru_cache()
def _get_list_person():
    """Random order -- non-deterministic. Increases accuracy over many runs,
    decreases accuracy on a single run."""
    with get_session() as sess:
        obs_count_src = {k: v for k, v in
                sess.query(sch.FusedObservation.src_id, sa.func.count(1))
                .group_by(sch.FusedObservation.src_id)}
        obs_count_dst = {k: v for k, v in
                sess.query(sch.FusedObservation.dst_id, sa.func.count(1))
                .group_by(sch.FusedObservation.dst_id)}
        people = [r[0] for r in sess.query(sch.FusedEntity.id)
                .where(sch.FusedEntity.type == sch.EntityTypeEnum.person)]

        people = [p for p in people
                if obs_count_src.get(p, 0) + obs_count_dst.get(p, 0) >= 100]

    people.sort()
    # Deterministic if desired; biases our sample though, so disabled by default
    #random.seed(1234)
    random.shuffle(people)

    return people


@functools.lru_cache()
def _get_min_date():
    with get_session() as sess:
        return arrow.get((sess.query(sch.FusedObservation)
                .order_by(sch.FusedObservation.time).limit(1)
                .scalar()).time)


@functools.lru_cache()
def _get_max_date():
    with get_session() as sess:
        return arrow.get((sess.query(sch.FusedObservation)
                .order_by(sch.FusedObservation.time.desc()).limit(1)
                .scalar()).time)


def _setup_rng():
    """Called on new fork; re-initialize RNGs."""
    torch.random.seed()

