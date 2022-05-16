import sqlite3
import numpy as np
import io
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String


class UpdateTable:
    """Add feature array to db."""

    def __init__(self):
        engine = create_engine("sqlite:///imageMeta.db", echo=True)
        self.conn = engine.connect()
        meta = MetaData()
        self.images = Table(
            "images",
            meta,
            Column("id", Integer, primary_key=True),
            Column("filename", String),
            Column("feature", String),
        )

    def add(self, row, feature_arr):
        """Convert array to binery and add to table."""
        sqlite3.register_adapter(np.ndarray, self.adapt_array)
        arr = (
            self.images.update()
            .where(self.images.c.id == row)
            .values(feature=feature_arr)
        )
        self.conn.execute(arr)

    def get(self, row):
        """Convert binery text back to array."""
        s = self.images.select().where(self.images.c.id == row)
        res = self.conn.execute(s).fetchone()
        return self.convert_array(res[-1])
        # print(res[-1])
        # print(self.convert_array(res[-1]))

    def adapt_array(self, arr):
        """
        http://stackoverflow.com/a/31312102/190597 (SoulNibbler)
        """
        out = io.BytesIO()
        np.save(out, arr)
        out.seek(0)
        return sqlite3.Binary(out.read())

    def convert_array(self, text):
        """
        https://stackoverflow.com/questions/18621513/python-insert-numpy-array-into-sqlite3-database
        """
        out = io.BytesIO(text)
        out.seek(0)
        return np.load(out)
