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
            Column("ORB_keypoint", String),
            Column("ORB_descriptor", String),
            Column("KAZE_keypoint", String),
            Column("KAZE_descriptor", String),
        )

    def add_by_id(self, id, keypoint, descriptor):
        """Convert array to binery and add to table."""
        sqlite3.register_adapter(np.ndarray, self.adapt_array)
        arr = (
            self.images.update().where(self.images.c.id == id).values(keypoint=keypoint)
        )
        self.conn.execute(arr)
        arr = (
            self.images.update()
            .where(self.images.c.id == id)
            .values(descriptor=descriptor)
        )
        self.conn.execute(arr)

    def add_by_filename(self, filename, keypoint, descriptor):
        """Convert array to binery and add to table."""
        sqlite3.register_adapter(np.ndarray, self.adapt_array)
        arr = (
            self.images.update()
            .where(self.images.c.filename == filename)
            .values(keypoint=keypoint)
        )
        self.conn.execute(arr)
        arr = (
            self.images.update()
            .where(self.images.c.filename == filename)
            .values(descriptor=descriptor)
        )
        self.conn.execute(arr)

    def get_by_id(self, id):
        """Convert binery text back to array."""
        s = self.images.select().where(self.images.c.id == id)
        res = self.conn.execute(s).fetchone()
        return self.convert_array(res[-2]), self.convert_array(res[-1])

    def get_by_filename(self, filename):
        """Convert binery text back to array."""
        s = self.images.select().where(self.images.c.filename == filename)
        res = self.conn.execute(s).fetchone()
        return self.convert_array(res[-2]), self.convert_array(res[-1])

    def get_id(self, filename):
        """Get id by using filename."""
        s = self.images.select().where(self.images.c.filename == filename)
        res = self.conn.execute(s).fetchone()
        return res[0]

    def get_filename(self, id):
        """Get filename by using id."""
        s = self.images.select().where(self.images.c.id == id)
        res = self.conn.execute(s).fetchone()
        return res[1]

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
