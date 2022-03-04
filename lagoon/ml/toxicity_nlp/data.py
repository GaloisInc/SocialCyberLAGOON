from lagoon.db.connection import get_session
from lagoon.db import schema as sch
from lagoon.ml.config import *

from collections import defaultdict
from tqdm import tqdm
import pickle
import os


def badword_count_bins(count):
    if count <= 5:
        return str(count)
    elif 6 <= count <= 7:
        return '6-7'
    elif 8 <= count <= 10:
        return '8-10'
    elif 11 <= count <= 15:
        return '11-15'
    elif 16 <= count <= 20:
        return '16-20'
    elif 21 <= count <= 30:
        return '21-30'
    elif 31 <= count <= 40:
        return '31-40'
    elif 41 <= count <= 50:
        return '41-50'
    else:
        return '50+'


def save_scores_badwords():
    result_clf = defaultdict(list)
    result_reg = defaultdict(list)

    with get_session() as sess:
        objs = sess.query(sch.FusedEntity).where(sch.FusedEntity.type.in_([sch.EntityTypeEnum.message, sch.EntityTypeEnum.git_commit]))
        objs_count = objs.count()
        for obj in tqdm(objs, total=objs_count):
            num_badwords = sum(obj.attrs.get(key) for key in obj.attrs.keys() if key.startswith('computed_badwords'))
            result_clf[badword_count_bins(num_badwords)].append(obj.attrs['toxicity_nlp_classification'][1])
            result_reg[badword_count_bins(num_badwords)].append(obj.attrs['toxicity_nlp_regression'])
    
    result = {
        'clf': result_clf,
        'reg': result_reg
    }
    with open(os.path.join(DATA_FOLDER, 'toxicity_nlp/scores_badwords.pkl'), 'wb') as f:
        pickle.dump(result, f)
