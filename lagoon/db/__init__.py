"""Database interaction code and utilities.

SQLALchemy tricks (as with `lagoon_cli.py shell`):

```python
>>> s = get_session().__enter__()
>>> o = s.execute(sa.select(sch.Entity).limit(1)).all()[0][0]
# Now, print number of objects loaded
>>> len(s.identity_map.keys())
1
# Load all observations for this entity
>>> o.obs_as_dst.all()
>>> len(s.identity_map.keys())
56
# Load only observations within a certain time range, e.g. first 2 days
>>> o.obs_as_dst.filter(sch.Observation.time < arrow.get('2021-02-01')).all()
...
```
"""

