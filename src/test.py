from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String
import sqlite3
import numpy as np
import io

engine = create_engine("sqlite:///imageMeta.db", echo=True)
conn = engine.connect()
meta = MetaData()


def adapt_array(arr):
    """
    http://stackoverflow.com/a/31312102/190597 (SoulNibbler)
    """
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())


def convert_array(text):
    """
    https://stackoverflow.com/questions/18621513/python-insert-numpy-array-into-sqlite3-database
    """
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)


images = Table(
    "images",
    meta,
    Column("id", Integer, primary_key=True),
    Column("filename", String),
    Column("feature", String),
)

# Converts np.array to TEXT when inserting
# sqlite3.register_adapter(np.ndarray, adapt_array)
# Converts TEXT to np.array when selecting
# sqlite3.register_converter("array", convert_array)

x = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]])
stmt = images.update().where(images.c.id == 1).values(feature=convert_array(x))
conn.execute(stmt)
s = images.select()
res = conn.execute(s).fetchone()

print(res[-1])
print(convert_array(res[-1]))
# i = 0
# for row in res:
#     if i != 1:
#         print(row[-1])
#         i += 1
#     else:
#         break
