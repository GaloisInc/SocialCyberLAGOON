"""LAGOON root module. Contains all importable modules.

Submodules
==========

.. autosummary::
    :toctree: _autosummary

    cli
    db
    ingest

Architecture
============

.. mermaid::

    graph LR;
    ingest --> db
    db --> ml
    ml --> db
"""

