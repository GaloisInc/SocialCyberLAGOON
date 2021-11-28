"""Import data from Python Enhancement Proposals (PEPs).
Data graph (entities are nodes, observations are labeled edges):

.. mermaid::

    flowchart LR
    pep[pep<br/><div style='text-align:left'>+number<br/> \
        +url<br/> \
        +type<br/> \
        +status<br/> \
        +title<br/> \
        +created<br/> \
        +replaces<br/> \
        +superseded_by<br/> \
        +requires<br/></div>]
    person[person<br/><div style='text-align:left'>+name<br/>+email</div>]
    person -- created --> pep
    pep -- superseded_by --> pep
    pep -- requires --> pep
    message -- message_ref --> pep
    git_commit -- message_ref --> pep
"""

from lagoon.db.connection import get_session
import lagoon.db.schema as sch
from lagoon.ingest.python_peps.peps import get_peps
from lagoon.ingest.util import clean_for_ingest

import arrow
from collections import defaultdict
import re
import sqlalchemy as sa
from tqdm import tqdm


def load_peps():
    """
    Scrape the web and get PEPs
    """
    entities = defaultdict(lambda: {})

    ## Function to get author
    def db_get_author(author):
        """ author is a str of the format 'John Doe <jon.doe@gmail.com>' """
        r = entities['author'].get(author)
        if r is None:
            author_match = re.match(r"^(?P<name>.*) <(?P<email>.*)>$", author)
            r = entities['author'][author] = sch.Entity(
                name = author,
                type = sch.EntityTypeEnum.person,
                attrs = {
                    'name': author_match.group("name"),
                    'email': author_match.group("email")
                }
            )
        return r
    
    ## Function to get PEP
    def db_get_pep(number):
        """ number is the PEP number """
        r = entities['pep'].get(number)
        if r is None:
            r = entities['pep'][number] = sch.Entity(
                name = f'PEP {number}',
                type = sch.EntityTypeEnum.pep,
                attrs = {
                    'number': number
                }
            )
        return r

    ## Function to prevent duplicate superseded_by observations
    def superseded_by_ob_exists(src_pep, dst_pep):
        """
        Superseded_by observations have the potential to be duplicated
            since both 'replaces' and 'superseded_by' attributes are present
            (see comments in peps.py for why this is the case)
        This method is to check if one already exists, so that a duplicate is not created
        
        Note that such a method is not required for the 'requires' attribute
            since those relationships are one-way, i.e. there is no 'required_by' attribute
        """
        for ob in src_pep.obs_as_src:
            if ob.type == sch.ObservationTypeEnum.superseded_by and ob.dst == dst_pep:
                return True
        return False
    
    
    ## Before investing significant time processing, ensure server is up
    with get_session() as sess:
        _b = sess.execute(sa.select(sch.Batch).limit(1))

    ## Create entities and observations
    print('Scraping web to get info for all PEPs ...')
    peps = get_peps()
    
    for number,row in peps.items():
        pep = db_get_pep(number)
        pep.attrs.update({
            'url': row['url'],
            'type': row['type'],
            'status': row['status'],
            'title': row['title'],
            'created': row['created'],
            'replaces': row['replaces'].split(' ') if row['replaces'] else [],
            'superseded_by': row['superseded_by'].split(' ') if row['superseded_by'] else [],
            'requires': row['requires'].split(' ') if row['requires'] else [],
        })

        for linked_pep_number in pep.attrs['replaces']:
            linked_pep = db_get_pep(int(linked_pep_number))
            if not superseded_by_ob_exists(linked_pep,pep):
                linked_pep.obs_as_src.append(sch.Observation(
                    dst = pep,
                    type = sch.ObservationTypeEnum.superseded_by,
                    time = arrow.get(row['created'], 'DD-MMM-YYYY').datetime
                ))

        for linked_pep_number in pep.attrs['superseded_by']:
            linked_pep = db_get_pep(int(linked_pep_number))
            if not superseded_by_ob_exists(pep,linked_pep):
                pep.obs_as_src.append(sch.Observation(
                    dst = linked_pep,
                    type = sch.ObservationTypeEnum.superseded_by,
                    time = arrow.get(peps[int(linked_pep_number)]['created'], 'DD-MMM-YYYY').datetime
                ))

        for linked_pep_number in pep.attrs['requires']:
            linked_pep = db_get_pep(int(linked_pep_number))
            pep.obs_as_src.append(sch.Observation(
                dst = linked_pep,
                type = sch.ObservationTypeEnum.requires,
                time = arrow.get(row['created'], 'DD-MMM-YYYY').datetime
            ))
        
        row_authors = row['authors'].split('; ')
        for row_author in row_authors:
            author = db_get_author(row_author)
            author.obs_as_src.append(sch.Observation(
                dst = pep,
                type = sch.ObservationTypeEnum.created,
                time = arrow.get(row['created'], 'DD-MMM-YYYY').datetime
            ))

    ## Write to database
    print('Writing to database...')

    with get_session() as sess:
        clean_for_ingest(sess)

        resource = 'ingest-python-peps'
        sch.Batch.cls_reset_resource(resource, session=sess)

        batch = sch.Batch(resource=resource)
        for _, edict in entities.items():
            for _, e in edict.items():
                batch.entities.append(e)
                for o in e.obs_as_src:
                    batch.observations.append(o)
                for o in e.obs_as_dst:
                    batch.observations.append(o)
        sess.add(batch)

        sess.flush()
        print(f'Finished with batch {batch.id}')


def link_peps():
    """
    Link PEP entities via observations to related existing entities like messages and commits
    """
    entities = []
    observations = []
    
    with get_session() as sess:
        pep_entities = sess.query(sch.Entity).where(sch.Entity.type==sch.EntityTypeEnum.pep)
        for pep_entity in tqdm(pep_entities, desc="Getting PEPs and their linked entities", total=pep_entities.count()):
            entities.append({
                'pep_id': pep_entity.id,
                'linked_messages': [] # list of tuples, each tuple = (id,time)
            })
            
            search_string = rf"pep {pep_entity.attrs['number']}\D" #must use lowercase since comparison is made using lowercase
            search_areas = [(sch.EntityTypeEnum.message, 'subject'), (sch.EntityTypeEnum.message, 'body_text'), (sch.EntityTypeEnum.git_commit, 'message')]
            for entity_type, attrs_field in search_areas:
                linked_messages = (sess.query(sch.Entity)
                    .where(sch.Entity.type == entity_type)
                    .where(sa.func.lower(sa.func.jsonb_extract_path_text(sch.Entity.attrs,attrs_field)).regexp_match(search_string))
                )
                # Easier debugging
                for elem in linked_messages.all():
                    try:
                        entities[-1]['linked_messages'].append((elem.id, elem.attrs['time']))
                    except KeyError:
                        raise KeyError(f'{elem} -- {elem.attrs}')

    for entity in tqdm(entities, desc='Creating observations'):
        for id_,time in entity['linked_messages']:
            observations.append(sch.Observation(
                src_id = id_,
                dst_id = entity['pep_id'],
                type = sch.ObservationTypeEnum.message_ref,
                time = arrow.get(time).datetime
            ))

    ## Write to database
    print('Writing to database...')
    
    with get_session() as sess:
        resource = 'link-python-peps'
        sch.Batch.cls_reset_resource(resource, session=sess)

        batch = sch.Batch(resource=resource)
        for o in observations:
            batch.observations.append(o)
        sess.add(batch)

        sess.flush()
        print(f'Finished with batch {batch.id}')
