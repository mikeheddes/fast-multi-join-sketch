#
# Software License
# Commercial reservation
#
# This License governs use of the accompanying Software, and your use of the Software constitutes acceptance of this license.
#
# You may use this Software for any non-commercial purpose, subject to the restrictions in this license. Some purposes which can be non-commercial are teaching, academic research, and personal experimentation. 
#
# You may not use or distribute this Software or any derivative works in any form for any commercial purpose. Examples of commercial purposes would be running business operations, licensing, leasing, or selling the Software, or distributing the Software for use with commercial products. 
#
# You may modify this Software and distribute the modified Software for non-commercial purposes; however, you may not grant rights to the Software or derivative works that are broader than those provided by this License. For example, you may not distribute modifications of the Software under terms that would permit commercial use, or under terms that purport to require the Software or derivative works to be sublicensed to others.
#
# You agree:
#
# 1. Not remove any copyright or other notices from the Software.
#
# 2. That if you distribute the Software in source or object form, you will include a verbatim copy of this license.
#
# 3. That if you distribute derivative works of the Software in source code form you do so only under a license that includes all of the provisions of this License, and if you distribute derivative works of the Software solely in object form you do so only under a license that complies with this License.
#
# 4. That if you have modified the Software or created derivative works, and distribute such modifications or derivative works, you will cause the modified files to carry prominent notices so that recipients know that they are not receiving the original Software. Such notices must state: (i) that you have changed the Software; and (ii) the date of any changes.
#
# 5. THAT THIS PRODUCT IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS PRODUCT, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. YOU MUST PASS THIS LIMITATION OF LIABILITY ON WHENEVER YOU DISTRIBUTE THE SOFTWARE OR DERIVATIVE WORKS.
#
# 6. That if you sue anyone over patents that you think may apply to the Software or anyone's use of the Software, your license to the Software ends automatically.
#
# 7. That your rights under the License end automatically if you breach it in any way.
#
# 8. UC Irvine and The Regents of the University of California reserves all rights not expressly granted to you in this license.
#
# To obtain a commercial license to this software, please contact:
# UCI Beall Applied Innovation
# Attn: Director, Research Translation Group
# 5270 California Ave, Suite 100
# Irvine, CA 92697
# Website: innovation.uci.edu
# Phone: 949-824-COVE (2683) 
# Email: cove@uci.edu
#
# Standard BSD License
#
# <OWNER> = The Regents of the University of California
# <ORGANIZATION> = University of California, Irvine
# <YEAR> = 2020
#
# Copyright (c) <2020>, The Regents of the University of California
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of The Regents of the University of California or the University of California, Irvine, nor the names of its contributors, may be used to endorse or promote products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
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
