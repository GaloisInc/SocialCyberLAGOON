"""Handles import of a JSON-formatted dump of GitHub pull requests from
RiverLoop.

TODO
* (done) Ensure commits are fusable -- no hash collisions between lxml and CPython.
* (mostly done) Pull PRs, try to link on GitHub username if possible...
* (done) Mark which repo is which in UI for LAGOON. E.g., show source batch(es)?
* (done) Query for 'scoder' should find 'stefan behnel' (and scoder)

# Riverloop's schema

Backed out via the `genson` tool.

* ['repository']
    * ['pullRequests']['edges'][]['node']
        * ['author']?['email']
        * ['bodyText']
        * ['number']
        * ['mergeCommit' | 'closedAt'] TODO AFTER PATCH
        * ['publishedAt']
        * ['commits']['nodes'][]['commit']['oid']
        * ['reviews']['nodes'][]
        * ['comments']['edges'][]['node'](['author' | 'bodyText' | 'publishedAt'])

"""

from lagoon.db.connection import get_session
import lagoon.db.schema as sch
from lagoon.ingest.util import clean_for_ingest, date_field_resolve

import arrow
import collections
import json
from pathlib import Path
import tqdm


def load_github_pr_riverloop(path: Path):
    """Ingests a JSON-formatted dump of GitHub pull requests from RiverLoop.

    The repository name will be taken from the first record in the JSON file.
    That record is expected to have a "name" field, whose value specifies the
    batch name.
    """

    with open(path) as f:
        if f.read(1) == '[':
            f.seek(0)
            data = json.load(f)
        else:
            # Multiline json
            f.seek(0)
            data = []
            for fline in f:
                data.append(json.loads(fline.strip()))

    repo_name = data[0]['repository']['name']

    with get_session() as sess:
        clean_for_ingest(session=sess)

        resource = f'ingest-github-pr-riverloop-{repo_name}'
        sch.Batch.cls_reset_resource(resource, session=sess)

        batch = sch.Batch(resource=resource)
        sess.add(batch)

        ET = sch.EntityTypeEnum
        OT = sch.ObservationTypeEnum
        entities = collections.defaultdict(dict)
        entities_new = 0
        def entity_lookup(db_type, db_attrs):
            nonlocal entities_new
            if db_type == ET.person:
                parts = []
                if db_attrs['name']:
                    parts.append(db_attrs['name'])
                if db_attrs['email']:
                    parts.append(f"<{db_attrs['email']}>")
                if db_attrs['github_user']:
                    parts.append(f"<GH {db_attrs['github_user']}>")
                id = ' '.join(parts)
            elif db_type == ET.git_commit:
                id = db_attrs['commit_sha']
            elif db_type == ET.github_review:
                id = f'{repo_name} {db_attrs["name"]}'
            elif db_type == ET.github_pullrequest:
                id = f'{repo_name} PR {db_attrs["github_pr_number"]}'
            elif db_type == ET.message:
                id = f'Message <{db_attrs["name"]}> hash {hash(db_attrs["message"])}'
            else:
                raise NotImplementedError(db_type)

            # Unified creation
            r = entities[db_type].get(id)
            if r is None:
                entities_new += 1
                r = entities[db_type][id] = sch.Entity(name=id,
                        type=db_type,
                        attrs=db_attrs,
                        batch=batch)
                sess.add(r)
            else:
                for k, v in db_attrs.items():
                    if not r.attrs.get(k):  # Overwrite any falsey value
                        r.attrs[k] = v
                    elif v:
                        # Ensure same if not missing
                        if v != r.attrs[k]:
                            raise ValueError(f'{v} != {r.attrs[k]}')
                r.attrs.update(db_attrs)
            if entities_new > 1000:
                sess.flush()
                entities_new = 0
            return r
        def entity_author_lookup(author_block):
            """Unified approach to dealing with 'author' blocks in data.
            """
            attrs = {
                    'name': author_block.get('name', None),
                    'email': author_block.get('email', None),
                    'github_user': author_block.get('login', None) or (author_block.get('user') or {}).get('login', None),
            }
            return entity_lookup(ET.person, attrs)

        for pr_group_i, pr_group in enumerate(tqdm.tqdm(data, desc='importing prs')):
            if 'repository' in pr_group:
                # Weird that this is needed
                assert 'pullRequests' not in pr_group, pr_group.keys()
                pr_group = pr_group['repository']

            if not pr_group['pullRequests']:
                continue

            for pr_i, pr in enumerate(pr_group['pullRequests'].get('edges') or []):
                try:
                    pr = pr['node']
                    e_pr = entity_lookup(ET.github_pullrequest, {
                            'github_pr_number': pr['number'],
                            'message': pr['bodyText'],
                            'created': date_field_resolve(pr['publishedAt']).timestamp(),
                    })

                    if pr['author'] is not None:
                        e_pr_auth = entity_author_lookup(pr['author'])
                        e_pr_auth.obs_as_src.append(sch.Observation(dst=e_pr, batch=batch,
                                type=OT.created, time=date_field_resolve(pr['publishedAt'])))

                    # New format missing this
                    if 'mergeCommit' in pr and pr['mergeCommit']:
                        e_pr_c = entity_lookup(ET.git_commit, {
                                'commit_sha': pr['mergeCommit']['oid'],
                        })
                        e_pr_c.obs_as_dst.append(sch.Observation(src=e_pr, batch=batch,
                                type=OT.merged_as, time=date_field_resolve(pr['closedAt'])))

                    # Commits
                    for pr_c in pr['commits']['nodes']:
                        pr_c = pr_c['commit']
                        sha = pr_c['oid']
                        e_pr_c = entity_lookup(ET.git_commit, {
                                'commit_sha': sha,
                        })
                        e_pr_c.obs_as_src.append(sch.Observation(dst=e_pr, batch=batch,
                                type=OT.attached_to,
                                # Unsure why the date is attached to the author
                                time=date_field_resolve(pr_c['author']['date'])))

                        # We want to save author information, because this is a
                        # source of GitHub usernames
                        if pr_c['author']:
                            entity_author_lookup(pr_c['author'])
                        if pr_c['committer']:
                            entity_author_lookup(pr_c['committer'])

                    # Comments
                    for pr_c in pr.get('comments', {}).get('edges', []):
                        pr_c = pr_c['node']
                        pr_c_date = date_field_resolve(pr_c['publishedAt'])
                        e_pr_c = entity_lookup(ET.message, {
                                'message': pr_c['bodyText'],
                                'name': f'for PR {pr["number"]} on '
                                    f'{arrow.get(pr_c_date).format("YYYY-MM-DD")}',
                        })
                        e_pr_c.obs_as_src.append(sch.Observation(dst=e_pr, batch=batch,
                                type=OT.message_to, time=date_field_resolve(pr_c['publishedAt'])))

                        if pr_c['author'] is not None:
                            e_pr_c_auth = entity_author_lookup(pr_c['author'])
                            e_pr_c_auth.obs_as_src.append(sch.Observation(dst=e_pr_c, batch=batch,
                                    type=OT.message_from, time=date_field_resolve(pr_c['publishedAt'])))

                    # Reviews
                    for pr_c in pr.get('reviews', {}).get('nodes', []):
                        rev_name = f'for PR {pr["number"]} review {pr_c["id"]}'
                        e_pr_c = entity_lookup(ET.github_review, {
                                'name': rev_name,
                                'id': pr_c['id'],
                                'message': pr_c['bodyText'],
                        })
                        pr_c_date = date_field_resolve(pr_c['publishedAt'])
                        e_pr_c.obs_as_src.append(sch.Observation(dst=e_pr, batch=batch,
                                type=OT.attached_to, time=pr_c_date))
                        if pr_c['author']:
                            e_pr_a = entity_author_lookup(pr_c['author'])
                            e_pr_a.obs_as_src.append(sch.Observation(dst=e_pr_c, batch=batch,
                                    type=OT.created, time=pr_c_date))

                        for pr_cc in pr_c['comments']['nodes']:
                            e_pr_cc = entity_lookup(ET.message, {
                                    'message': pr_cc['bodyText'],
                                    'name': f'{rev_name} comment {pr_cc["id"]}',
                            })
                            e_pr_cc.obs_as_src.append(sch.Observation(dst=e_pr_c, batch=batch,
                                    type=OT.message_to, time=date_field_resolve(pr_cc['publishedAt'])))
                            if pr_cc['author']:
                                e_pr_cc_a = entity_author_lookup(pr_cc['author'])
                                e_pr_cc_a.obs_as_src.append(sch.Observation(dst=e_pr_cc, batch=batch,
                                        type=OT.message_from, time=date_field_resolve(pr_cc['publishedAt'])))
                except:
                    raise ValueError(f'Handling [{pr_group_i}][{pr_i}]')

