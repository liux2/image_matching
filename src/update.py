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
            Column("caption", String),
            Column("ORB_keypoint", String),
            Column("ORB_descriptor", String),
            Column("KAZE_keypoint", String),
            Column("KAZE_descriptor", String),
        )

    def add_by_id(self, id, orb_kps, orb_des, kaze_kps, kaze_des):
        """Convert array to binery and add to table."""
        sqlite3.register_adapter(np.ndarray, self.adapt_array)
        # Keypoints
        orb = (
            self.images.update()
            .where(self.images.c.id == id)
            .values(ORB_keypoint=orb_kps)
        )
        self.conn.execute(orb)
        kaze = (
            self.images.update()
            .where(self.images.c.id == id)
            .values(KAZE_keypoint=kaze_kps)
        )
        self.conn.execute(kaze)
        # Descriptors
        orb = (
            self.images.update()
            .where(self.images.c.id == id)
            .values(ORB_descriptor=orb_des)
        )
        self.conn.execute(orb)
        kaze = (
            self.images.update()
            .where(self.images.c.id == id)
            .values(KAZE_descriptor=kaze_des)
        )
        self.conn.execute(kaze)

    def add_by_filename(self, filename, orb_kps, orb_des, kaze_kps, kaze_des):
        """Convert array to binery and add to table."""
        sqlite3.register_adapter(np.ndarray, self.adapt_array)
        # Keypoints
        orb = (
            self.images.update()
            .where(self.images.c.filename == filename)
            .values(ORB_keypoint=orb_kps)
        )
        self.conn.execute(orb)
        kaze = (
            self.images.update()
            .where(self.images.c.filename == filename)
            .values(KAZE_keypoint=kaze_kps)
        )
        self.conn.execute(kaze)
        # Descriptors
        orb = (
            self.images.update()
            .where(self.images.c.filename == filename)
            .values(ORB_descriptor=orb_des)
        )
        self.conn.execute(orb)
        kaze = (
            self.images.update()
            .where(self.images.c.filename == filename)
            .values(KAZE_descriptor=kaze_des)
        )
        self.conn.execute(kaze)

    def get_by_id(self, id, method="ORB"):
        """Convert binery text back to array."""
        s = self.images.select().where(self.images.c.id == id)
        res = self.conn.execute(s).fetchone()
        if method == "ORB":
            return self.convert_array(res[-4]), self.convert_array(res[-3])
        elif method == "KAZE":
            return self.convert_array(res[-2]), self.convert_array(res[-1])

    def get_by_filename(self, filename, method="ORB"):
        """Convert binery text back to array."""
        s = self.images.select().where(self.images.c.filename == filename)
        res = self.conn.execute(s).fetchone()
        if method == "ORB":
            return self.convert_array(res[-4]), self.convert_array(res[-3])
        elif method == "KAZE":
            return self.convert_array(res[-2]), self.convert_array(res[-1])

    def get_caption(self, filename=None, id=None):
        """Get caption by using filename or id."""
        if filename:
            s = self.images.select().where(self.images.c.filename == filename)
            res = self.conn.execute(s).fetchone()
            return res[2]
        elif id:
            s = self.images.select().where(self.images.c.id == id)
            res = self.conn.execute(s).fetchone()
            return res[2]

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
        return np.load(out, allow_pickle=True)
