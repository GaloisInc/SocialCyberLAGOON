"""Import data from Have I Been Pwned (https://haveibeenpwned.com/).

.. mermaid::

    flowchart LR
    person[person<br/><div style='text-align:left'>+email</br>+breaches</br>+pastes</div>]
"""

import sqlalchemy as sa
from lagoon.db.connection import get_session
import lagoon.db.schema as sch
from lagoon.ingest.util import clean_for_ingest

from tqdm import tqdm
import time
import os
import requests


def do_request(email: str, stat: str, hibp_api_key: str) -> int:
    """
    email: The email ID to search HIBP for
    stat: As in load_hibp()
    hibp_api_key: A vald key for accessing the HIBP APIs
    """
    while True:
        r = requests.get(
            url = f"https://haveibeenpwned.com/api/v3/{'breached' if stat=='breaches' else 'paste'}account/{email}",
            headers = {'hibp-api-key': hibp_api_key}
        )
        if r.status_code == 429: # rate limiting
            time.sleep(2) # according to the website, requests can be made every 1500 milliseconds, we round it up to 2 seconds for extra buffer

            ## Alternate method - Wait for exactly prescribed number of seconds
            ## Commented out since (limited) testing suggests that the runtime for this is not different from a constant time.sleep(n)
            # match = re.search(r"ry again in (?P<wait>\d+) second", r.json()['message'])
            # time.sleep(int(match.group('wait')))
        
        elif r.status_code == 200: # successful
            try:
                return len(r.json())
            except: # typically simplejson.errors.JSONDecodeError, but we want to catch other exceptions as well, hence don't specify
                return 0

        else: # usually 404 if the email is not found in the database, sometimes 400 if the email is invalid (e.g. without '@')
            return 0


def get_hibp_api_keys(num_keys=2):
    """
    Get the number of breaches or pastes for each email address in the DB, and create entities
    """
    hibp_api_keys = []
    for i in range(1,num_keys+1):
        hibp_api_keys.append(os.getenv(f'HIBP_API_KEY_{i}'))
    assert None not in hibp_api_keys, f'{num_keys} keys are expected to access the APIs for https://haveibeenpwned.com/. Please set the environment variables `HIBP_API_KEY_1`, `HIBP_API_KEY_2`, ... to these keys. See https://haveibeenpwned.com/API/v3 on how to purchase keys.'
    return hibp_api_keys


def load_hibp(stat='breaches', num_keys=2):
    """
    stat: Either 'breaches' or 'pastes'
    num_keys: The number of keys available to access the Have I Been Pwned APIs.
        See https://haveibeenpwned.com/API/v3 on how to purchase keys.
        Then, set environment variables to the key values:
            export HIBP_API_KEY_1=<key_value_1>
            export HIBP_API_KEY_2=<key_value_2>
            ...
    """
    stat_choices = ['breaches','pastes']
    assert stat in stat_choices, f"'stat' must be one out of {stat_choices}"
    
    hibp_api_keys = get_hibp_api_keys(num_keys=num_keys)
    key_selector = 0
    
    with get_session() as sess:
        # Before investing significant time processing, ensure server is up
        _b = sess.execute(sa.select(sch.Batch).limit(1))

        clean_for_ingest(session=sess)
        resource = f'hibp-{stat}'
        sch.Batch.cls_reset_resource(resource, session=sess)
        batch = sch.Batch(resource=resource)
        sess.add(batch)
        
        email_tuples = (
            sess.query(sch.Entity.attrs['email'].astext)
            .where(sch.Entity.type == sch.EntityTypeEnum.person)
            .distinct()
        )
        count = email_tuples.count()
        
        for idx,email_tuple in enumerate(tqdm(email_tuples, total=count)):
            email = email_tuple[0]
            
            # Explicitly check if the email is valid
            # If it's not, although the pastes API will return 400 'Invalid email address',
            # the breaches API will not, which leads to incorrect results
            # E.g. querying None in the breaches API returns 8 valid results
            if type(email) != str or '@' not in email:
                continue
            
            result = do_request(
                email = email,
                stat = stat,
                hibp_api_key = hibp_api_keys[key_selector]
            )
            key_selector = (key_selector+1)%len(hibp_api_keys)
            sess.add(
                sch.Entity(
                    name = f'HIBP {stat} <{email}>',
                    type = sch.EntityTypeEnum.person,
                    attrs = {
                        'email': email,
                        stat: result
                    },
                    batch = batch
                )
            )

            if idx%1000 == 0:
                sess.flush()

        sess.flush()
        print(f'Finished with batch {batch.id}')
