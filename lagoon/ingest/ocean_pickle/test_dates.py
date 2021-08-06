from . import _date_field_resolve

import pytest

def test_bad_row():
    d = 'hey there'
    with pytest.raises(ValueError) as exc:
        _date_field_resolve(d)
    assert 'hey there' in str(exc)

def test_normal_row():
    d = '2021-08-01 03:45:56'
    _date_field_resolve(d)

def test_weird_row():
    d = 'Thu, 23 May 2002 23.19.24 +0200'
    _date_field_resolve(d)

