import io
import re
import os

import torch
import pandas as pd

from experiment import NULL_VALUE, IMDB_DIR, CARDEST_DIR, CACHE_DIR, Table

os.makedirs(CACHE_DIR, exist_ok=True)

# STATS

print("Caching STATS tables...")

os.makedirs(os.path.join(CACHE_DIR, "stats"), exist_ok=False)

schema_path = os.path.join(CARDEST_DIR, "datasets", "stats_simplified", "stats.sql")
with open(schema_path) as f:
    sql = f.read()

table_specs = sql.split("CREATE TABLE ")[1:]
table_specs = [t.split(");")[0] for t in table_specs]
table_specs = [t.split("(", 1) for t in table_specs]

for name, attributes in table_specs:
    name = name.strip()
    print(name)

    attributes = attributes.split(",")
    # Creates list of (attribute_name, type)
    attributes = [tuple(a.strip().split(" ", 1)) for a in attributes]

    data_path = os.path.join(CARDEST_DIR, "datasets", "stats_simplified", name + ".csv")

    attribute_names = [a[0] for a in attributes]
    string_attributes = [
        a[0] for a in attributes if a[1].upper().startswith("CHARACTER")
    ]
    datetime_attributes = [a[0] for a in attributes if a[1] == "TIMESTAMP"]

    dtype_mapping = {
        "CHARACTER": str,
        "TIMESTAMP": str,
        "INTEGER": float,
        "SERIAL": float,
        "SMALLINT": float,
    }

    dtypes = {a[0]: dtype_mapping[a[1].upper().split(" ")[0]] for a in attributes}

    df = pd.read_csv(
        data_path,
        header=0,
        parse_dates=datetime_attributes,
        encoding="utf-8",
        sep=",",
        names=attribute_names,
        dtype=dtypes,
    )

    null_occurences = df.isin([NULL_VALUE]).values.sum()
    if null_occurences > 0:
        raise RuntimeError(
            f"Found the NULL value in the table {name}, consider using a different NULL value."
        )

    # arbitrary value used to denote NULL
    df.fillna(NULL_VALUE, inplace=True)

    table = Table(df, name, attribute_names, string_attributes, datetime_attributes)

    path = os.path.join(CACHE_DIR, "stats", name + ".pkl")
    torch.save(table, path)

# IMDB

print("Caching IMDB tables...")

os.makedirs(os.path.join(CACHE_DIR, "imdb"), exist_ok=False)

with open("imdb/schematext.sql") as f:
    sql = f.read()

table_specs = sql.split("CREATE TABLE ")[1:]
table_specs = [t.split(");")[0] for t in table_specs]
table_specs = [t.split("(", 1) for t in table_specs]

for name, attributes in table_specs:
    name = name.strip()
    print(name)

    attributes = attributes.split(",")
    # Creates list of (attribute_name, type)
    attributes = [tuple(a.strip().split(" ", 1)) for a in attributes]

    data_path = os.path.join(IMDB_DIR, name + ".csv")

    attribute_names = [a[0] for a in attributes]
    string_attributes = [
        a[0] for a in attributes if a[1].upper().startswith("CHARACTER")
    ]
    datetime_attributes = [a[0] for a in attributes if a[1] == "TIMESTAMP"]

    dtype_mapping = {
        "CHARACTER": str,
        "TIMESTAMP": str,
        "INTEGER": float,
        "SERIAL": float,
        "SMALLINT": float,
    }

    dtypes = {a[0]: dtype_mapping[a[1].upper().split(" ")[0]] for a in attributes}

    with open(data_path, "r") as f:
        # TODO: memory usage could be improved by replacing unescaped variable
        # in an iterator fashion instead of all at ones.
        data = f.read()
        # only replace/escape the backslashes that are not proceded by a quote
        # because those can escape the comma which causes incorrect parsing
        data = re.sub(r"\\\"", "\"\"", data)
        if name == "movie_info":
            # These two lines cause trouble becuase they end with a backslash before the final quote
            data = data.replace(
                "'Harry Bellaver' (qv).\\\"\",",
                "'Harry Bellaver' (qv).\\\\\","
            )
            data = data.replace(
                "who must go back and find his bloodlust one last time. \\\"\",",
                "who must go back and find his bloodlust one last time. \\\\\","
            )

        elif name == "person_info":
            data = data.replace("\\\"\",", "\\\\\",")

    df = pd.read_csv(
        io.StringIO(data),
        header=None,
        parse_dates=datetime_attributes,
        encoding='utf-8', 
        sep=",",
        names=attribute_names,
        dtype=dtypes,
    )

    null_occurences = df.isin([NULL_VALUE]).values.sum()
    if null_occurences > 0:
        raise RuntimeError(
            f"Found the NULL value in the table {name}, consider using a different NULL value."
        )

    # arbitrary value used to denote NULL
    df.fillna(NULL_VALUE, inplace=True)

    table = Table(df, name, attribute_names, string_attributes, datetime_attributes)

    path = os.path.join(CACHE_DIR, "imdb", name + ".pkl")
    torch.save(table, path)
