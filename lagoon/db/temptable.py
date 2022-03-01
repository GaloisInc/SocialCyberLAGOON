"""Sourced from https://stackoverflow.com/a/66156968; modified.

Notably, `multi_session` is new.

Example usage with specific analytic query; see ./temptable_test.py
"""

import contextlib
import time

import sqlalchemy as sa
from sqlalchemy.schema import CreateTable

_multi_session_tables = {}

def _copy_table_args(model, **kwargs):
    """Try to copy existing __table_args__, override params with kwargs"""
    table_args = getattr(model, '__table_args__', ())

    if isinstance(table_args, tuple):
        new_args = []
        saw_dict = False
        for arg in table_args:
            if isinstance(arg, dict):
                saw_dict = True
                table_args_dict = arg.copy()
                table_args_dict.update(**kwargs)
                new_args.append(table_args_dict)
            elif isinstance(arg, sa.Index):
                index = sa.Index(
                    arg.name,
                    *[col for col in arg.columns.keys()],
                    unique=arg.unique,
                    **arg.kwargs,
                )
                new_args.append(index)
            else:
                # TODO: need to handle Constraints
                raise Exception(f"Unhandled table arg: {arg}")
        if not saw_dict:
            new_args.append(kwargs)
        table_args = tuple(new_args)
    elif isinstance(table_args, dict):
        table_args = {
            k: (v.copy() if hasattr(v, "copy") else v) for k, v in table_args.items()
        }
        table_args.update(**kwargs)
    else:
        raise Exception(f"Unexpected __table_args__ type: {table_args}")

    return table_args


def _copy_table_from_model(model, multi_session):
    model_name = model.__name__ + "Tmp"
    if not multi_session:
        table_name = model.__table__.name + "_" + str(time.time()).replace(".", "_")
    else:
        table_name = _multi_session_table_name(model)
    kwargs = {}
    if not multi_session:
        kwargs = {'prefixes': ['TEMPORARY']}
    table_args = _copy_table_args(model, extend_existing=True, _extend_on=True,
            **kwargs)

    args = {c.name: c.copy() for c in model.__table__.c}
    args["__tablename__"] = table_name
    args["__table_args__"] = table_args

    copy_model = type(model_name, model.__bases__, args)
    if multi_session:
        assert table_name not in _multi_session_tables, \
                'Should only call temp_table_from_model with multi_session once'
        _multi_session_tables[table_name] = copy_model
    return copy_model


@contextlib.contextmanager
def multi_session_dropper(sess_factory, table, *, keep=False):
    """Given some `table` from :meth:`temp_table_from_model` with
    ``multi_session=True``, produces a context manager that, on exit, uses
    `sess_factory` to destroy the table.
    """
    table_name = table.__table__.name
    assert table_name.endswith('_inst'), table_name
    try:
        yield
    finally:
        if not keep:
            with sess_factory() as sess:
                sess.execute(sa.text(f'''DROP TABLE IF EXISTS "{table_name}"'''))
        _multi_session_tables.pop(table_name, None)


def multi_session_remap(model):
    """Get a remapped version of `model` which is the object with multi_session
    support. This is important for e.g. multiprocessing.
    """
    # Early exit in case this was already called; squelches warning in local
    # code.
    table_name = _multi_session_table_name(model)
    table_cls = _multi_session_tables.get(table_name)
    if table_cls is not None:
        return table_cls
    return _copy_table_from_model(model, multi_session=True)


def temp_table_from_model(sess, model, *, multi_session=False):
    """Create a temporary table (dies with the connection) for a given
    SQLAlchemy definition. Supports indices.

    Args:
        sess: Active session, only used for table creation.
        model: Model to be created; should derive from `sa.orm.declarative_base`
        multi_session: If ``True``, do NOT use a temporary table, but DO delete
                any existing versin of this table and create a new one.

                If ``'streamlit'``, then only create the table if it does not
                exist. Otherwise, assume the user has a `multi_session_dropper`
                block which can clean up extra rows as part of the page that
                makes the temporary table.
    """
    assert multi_session in [False, True, 'streamlit'], multi_session

    if multi_session:
        table_name = _multi_session_table_name(model)
        if multi_session == 'streamlit':
            copy_model = _multi_session_tables.get(table_name)
            if copy_model:
                return copy_model
        else:
            sess.execute(sa.text(f'''DROP TABLE IF EXISTS "{table_name}"'''))
    copy_model = _copy_table_from_model(model, multi_session)

    # Actually create the table
    try:
        #print(str(CreateTable(copy_model.__table__)))
        copy_model.__table__.create(sess.connection())
    except sa.exc.ProgrammingError as e:
        if ('psycopg2.errors.DuplicateTable' not in str(e)
                or multi_session != 'streamlit'):
            raise

    return copy_model


def _multi_session_table_name(model):
    return model.__table__.name + '_inst'

