
import os
import pandas as pd

from lagoon.fusion import _fuse_entity_valid_email

def test_valid_email_basic():
    """Basic tests"""
    assert _fuse_entity_valid_email('hithere@gmail.com')


def test_valid_email_fake_email_stats():
    """Check email stats per Sourya's collection
    """
    df = pd.read_csv(os.path.join(os.path.dirname(__file__),
            'fake_email_stats.csv'))
    for email in df['email'].values:
        assert not _fuse_entity_valid_email(email)

