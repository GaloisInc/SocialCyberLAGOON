"""
Generate CSVs for entities in the DB which have toxicity_nlp attributes
Required by Sam and Brad at UVM
"""

import os
import sys
import csv
from lagoon.db.connection import get_session
from lagoon.db import schema as sch
from lagoon.ingest.toxicity_nlp import process_message, process_commit
from tqdm import tqdm


# Change this as desired
OUTPUT_FOLDER = '/Volumes/GoogleDrive/Shared drives/SocialCyber LAGOON/20220217_db_toxicity_nlp_records/'

# Initially create the CSVs in the current directory for speedy writing
os.chdir(sys.path[0])

with get_session() as sess:
    
    ## Messages
    objs = sess.query(sch.FusedEntity).where(sch.FusedEntity.type == sch.EntityTypeEnum.message)
    with open('toxicity_nlp_messages.csv','w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow([
            'db_id_20220217',
            'name',
            'subject',
            'body_text',
            'nlp_input_string',
            'toxicity_nlp_classification',
            'toxicity_nlp_regression'
        ])
        objs_count = objs.count()
        for obj in tqdm(objs, total=objs_count):
            csvwriter.writerow([
                obj.id,
                obj.name,
                obj.attrs.get('subject'),
                obj.attrs.get('body_text'),
                process_message(obj),
                obj.attrs.get('toxicity_nlp_classification'),
                obj.attrs.get('toxicity_nlp_regression')
            ])

    ## Commits
    objs = sess.query(sch.FusedEntity).where(sch.FusedEntity.type == sch.EntityTypeEnum.git_commit)
    with open('toxicity_nlp_commits.csv','w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow([
            'db_id_20220217',
            'name',
            'message',
            'nlp_input_string',
            'toxicity_nlp_classification',
            'toxicity_nlp_regression'
        ])
        objs_count = objs.count()
        for obj in tqdm(objs, total=objs_count):
            csvwriter.writerow([
                obj.id,
                obj.name,
                obj.attrs.get('message'),
                process_commit(obj),
                obj.attrs.get('toxicity_nlp_classification'),
                obj.attrs.get('toxicity_nlp_regression')
            ])

# Now, move the CSVs to the desired location
os.system(f"mv toxicity_nlp_messages.csv '{OUTPUT_FOLDER}'")
os.system(f"mv toxicity_nlp_commits.csv '{OUTPUT_FOLDER}'")
