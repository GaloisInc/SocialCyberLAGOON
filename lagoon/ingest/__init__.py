"""Modules for ingesting information from the wide world into a format amenable
to LAGOON.

TODO NOTE -- Each ingest module should generate a unique ID for each operation;
incremental updates / clearing out previously loaded data is the burden of each
module on its own!

Entity/Observation model
------------------------

Spatiotemporal graphs within LAGOON are represented via an entity/observation
model. This model implies that there are certain, time-invariant entities (nouns)
which interact with one another over some number of observations. Observations
are directional by default, and occur between two entities. Both entities and
observations may have additional attributes attached to them, the significance
of which is specific to the type of entity or observation.

.. mermaid::

    flowchart RL
    entity["entity, first type<br/><div style='text-align:left'>+attribute1<br/>+attribute2</div>"]
    entity -- "observation, type<br/>+attributes" --> entity
    entity2["entity, second type"]
    entity2 -- "observation, other type" --> entity
"""

