"""
database/db.py
---------------
Single shared SQLAlchemy instance. Kept in its own file (separate from
models.py) to avoid circular imports between app.py and models.py.
"""

from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()
