"""Import data from a git repository.

Data graph (entities are nodes, observations are labeled edges):

.. mermaid::

    flowchart LR
    person[person<br/><div style='text-align:left'>+name<br/>+email</div>]
    git_commit[git_commit<br/><div style='text-align:left'>+message</div>]
    person -- committed --> git_commit
    person -- created --> git_commit
    git_commit -- modified --> file
"""

from lagoon.db.connection import get_session
import lagoon.db.schema as sch

import arrow
import collections
import git
from pathlib import Path
import tqdm

def load_git_repo(path: Path):
    """Ingests a git repository. No optimizations at the moment.
    """
    repo = git.Repo(path)

    # Call this early -- want it to fail before spending a bunch of time
    # importing commits.
    repo_name = _repo_get_name(repo)

    entities = collections.defaultdict(lambda: {})
    # Utility functions...
    def db_get_commit(o):
        id = f'{o.hexsha[:10]}'
        r = entities['commit'].get(id)
        if r is None:
            r = entities['commit'][id] = sch.Entity(name=id,
                type=sch.EntityTypeEnum.git_commit,
                attrs={
                    'message': s.message,
                })
        return r
    def db_get_file(o):
        id = o
        r = entities['file'].get(id)
        if r is None:
            r = entities['file'][id] = sch.Entity(name=id,
                    type=sch.EntityTypeEnum.file)
        return r
    def db_get_person(o):
        name = o.name
        email = o.email
        id = f'{name} <{email}>'
        r = entities['person'].get(id)
        if r is None:
            r = entities['person'][id] = sch.Entity(name=id,
                    type=sch.EntityTypeEnum.person,
                    attrs={
                        'name': name,
                        'email': email,
                    })
        return r

    # Loop and process
    seen = set()
    stack = [repo.commit(v) for v in repo.refs]
    all_commits = []
    while stack:
        s = stack.pop()
        if s.hexsha in seen:
            continue
        seen.add(s.hexsha)
        stack.extend(s.parents)

        all_commits.append(s)

    for s in tqdm.tqdm(all_commits, desc='importing commits'):
        # Scrape information / entities
        # s.author, s.committer, s.authored_datetime,
        # s.message, s.stats.files

        commit_time = arrow.get(s.authored_date).datetime
        db_commit = db_get_commit(s)
        db_author = db_get_person(s.author)
        db_committer = db_get_person(s.committer)

        db_author.obs_as_src.append(sch.Observation(dst=db_commit,
                type=sch.ObservationTypeEnum.created,
                time=commit_time))
        db_committer.obs_as_src.append(sch.Observation(dst=db_commit,
                type=sch.ObservationTypeEnum.committed,
                time=arrow.get(s.committed_date).datetime))

        for fpath in s.stats.files:
            db_f = db_get_file(fpath)
            db_f.obs_as_dst.append(sch.Observation(src=db_commit,
                type=sch.ObservationTypeEnum.modified,
                time=commit_time))

    # Now that we have a web, commit it to DB
    with get_session() as sess:
        resource = f'ingest-git-{repo_name}'
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



def _repo_get_name(repo):
    """Returns the repository's name as a short, schema-less URL
    """
    url = repo.remotes['origin'].url
    if '@' in url.split('/', 1)[0]:
        # git
        url = url.split('@', 1)[1]
        assert ':' in url, url
        url = '/'.join(url.split(':', 1))
        return url
    elif url.startswith('https://'):
        return url[len('https://'):]
    else:
        raise NotImplementedError(f'Should return pseudo-url like '
                f'`hostname/path/to/repo` from: {url}')

