"""Pure statistics modules: standard library only, no config, no database.

Nothing here may import from ``services`` (whose ``__init__`` loads SQLAlchemy
and the Composio client), so code outside the platform can import these modules
without the platform's dependencies.
"""
