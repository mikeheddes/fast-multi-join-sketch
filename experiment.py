from typing import Literal, Optional, Generator, Tuple, Dict, List, Set, Any, Union
from collections import defaultdict, deque
import io
import re
import os
import csv
import math
import time
import copy
import string
import random
from datetime import datetime

import torch
from torch.fft import fft, ifft
import numpy
import pandas as pd
from tap import Tap
from tqdm import tqdm

from kwisehash import KWiseHash

camel_to_snake_re = re.compile(r"(?<!^)(?=[A-Z])")
selection_ops_re = re.compile(r"(\>\=?|\<\=?|\<\>|\=|BETWEEN|IN|LIKE|NOT LIKE)")
attribute_re = re.compile(r"(_|[a-zA-Z])(_|\d|[a-zA-Z])*.(_|[a-zA-Z])+")
escaped_backslash_re = re.compile(r"\\\"")

NULL_VALUE = -123456
CARDEST_DIR = "End-to-End-CardEst-Benchmark-master"
IMDB_DIR = "imdb"
CACHE_DIR = ".cache"


# http://en.wikipedia.org/wiki/Mersenne_prime
MERSENNE_PRIME = (1 << 61) - 1


def text_between(input: str, start: str, end: str):
    # getting index of substrings
    idx_start = input.index(start)
    idx_end = input.index(end)

    # length of substring 1 is added to
    # get string from next character
    return input[idx_start + len(start) + 1 : idx_end]


def is_number(n):
    try:
        # Type-casting the string to `float`.
        # If string is not a valid `float`,
        # it'll raise `ValueError` exception
        float(n)
    except ValueError:
        return False
    return True


def random_string(len: int = 7) -> str:
    chars = string.ascii_letters + string.digits
    rand_chars = random.choices(chars, k=len)
    rand_str = "".join(rand_chars)
    return rand_str


class Timer(object):
    def __init__(self):
        self.start = time.perf_counter()

    def stop(self):
        return time.perf_counter() - self.start


class SignHash(object):
    fn: KWiseHash

    def __init__(self, *size, k=2) -> None:
        self.fn = KWiseHash(*size, k=k)

    def __call__(self, items: torch.Tensor) -> torch.Tensor:
        return self.fn.sign(items)


class ComposedSigns(object):
    hashes: List[SignHash]

    def __init__(self, *hashes: SignHash) -> None:
        self.hashes = hashes

    def add(self, hash: SignHash) -> None:
        self.hashes.append(hash)

    def __call__(self, items: torch.Tensor) -> torch.Tensor:
        result = 1

        for hash in self.hashes:
            result *= hash(items)

        return result


class BinHash(object):
    fn: KWiseHash

    def __init__(self, *size, bins, k=2) -> None:
        self.num_bins = bins
        self.fn = KWiseHash(*size, k=k)

    def __call__(self, items: torch.Tensor) -> torch.Tensor:
        return self.fn.bin(items, self.num_bins)


MethodName = Literal["exact", "ams", "compass-merge", "compass-partition", "count-conv"]


class Arguments(Tap):
    method: MethodName # Use count-conv for our proposed method
    query: str # For the list of available queries see the README
    seed: Optional[int] = None
    bins: int = 1
    means: int = 1
    medians: int = 1
    estimates: int = 1
    batch_size: int = 2**16
    result_dir: str = "results"
    data_dir: str = ""

    def process_args(self):
        # Validate arguments
        if self.method == "ams" and self.bins != 1:
            raise ValueError("Bins must be 1 for AMS")

        if self.method == "ams":
            self.batch_size = max(self.batch_size // self.means, 1)

        if self.method == "ams" and self.bins != 1:
            raise ValueError("bins must be 1 for the ams method")

        if self.method != "ams" and self.means != 1:
            raise ValueError("means can only be used with the ams methods")
        
        if self.method == "exact":
            if self.bins != 1 or self.means != 1 or self.medians != 1 or self.estimates != 1:
                raise ValueError("bins, means, medians, and estimates must be 1 for the exact method")

        if self.bins < 1:
            raise ValueError("Number of bins cannot be negative")

        if self.means < 1:
            raise ValueError("Number of means cannot be negative")

        if self.medians < 1:
            raise ValueError("Number of medians cannot be negative")

        if self.estimates < 1:
            raise ValueError("Number of estimates cannot be negative")


def seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    numpy.random.seed(seed)


def read_sql_query(root: str, benchmark: str, name: str) -> str:

    if benchmark == "job_light":
        path = os.path.join(
            root, CARDEST_DIR, "workloads", "job-light", "job_light_queries.sql"
        )
    elif benchmark == "job_light_sub":
        path = os.path.join(
            root,
            CARDEST_DIR,
            "workloads",
            "job-light",
            "sub_plan_queries",
            "job_light_sub_query.sql",
        )
    elif benchmark == "stats":
        path = os.path.join(
            root, CARDEST_DIR, "workloads", "stats_CEB", "stats_CEB.sql"
        )
    elif benchmark == "stats_sub":
        path = os.path.join(
            root,
            CARDEST_DIR,
            "workloads",
            "stats_CEB",
            "sub_plan_queries",
            "stats_CEB_sub_queries.sql",
        )
    else:
        raise ValueError(f"Query '{benchmark}-{name}' does not exist.")

    with open(path, "r") as f:
        if benchmark == "job":
            sql = f.read()

        else:
            sqls = f.readlines()
            idx = int(name) - 1

            if idx < 0 or idx >= len(sqls):
                raise ValueError(f"Query '{benchmark}-{name}' does not exist.")

            sql = sqls[idx]

            # stats has the true cardinality prepended to the query
            if benchmark == "stats":
                sql = sql.split("||", 1)[1]
            # stats_sub has the corresponding full query appended to the query
            elif benchmark == "stats_sub":
                sql = sql.split("||", 1)[0]

    return sql.strip()


class Tokenizer(object):
    token2idx: dict[str, int]
    idx2token: list[str]

    def __init__(self):
        self.token2idx = {}
        self.idx2token = []

    def add(self, token: str) -> int:
        """Adds a token to the dictionary and returns its index

        For any input that is not a string, returns NaN.

        Args:
            token (str): the token to add

        Returns:
            int: the index of the token
        """
        if type(token) != str:
            return float("nan")

        if token not in self.token2idx:
            self.idx2token.append(token)
            self.token2idx[token] = len(self.idx2token) - 1

        return self.token2idx[token]

    def __getitem__(self, index_or_token: Union[int, str]) -> Union[str, int]:
        """Returns the token at the given index or the index of the given token"""
        if type(index_or_token) == int:
            return self.idx2token[index_or_token]
        else:
            return self.token2idx[index_or_token]

    def __len__(self) -> int:
        return len(self.idx2token)


class Table(object):
    name: str
    attributes: List[str]
    attribute2idx: Dict[str, int]
    data: torch.Tensor
    tokenizers: Dict[str, Tokenizer]

    def __init__(
        self,
        df: pd.DataFrame,
        name: str,
        attributes: List[str],
        string_attributes: List[str],
        datetime_attributes: List[str],
    ) -> None:

        self.name = name
        self.attributes = attributes
        self.attribute2idx = {name: i for i, name in enumerate(self.attributes)}
        self.tokenizers = {}

        for attr in datetime_attributes:
            self._datetime_attr(df, attr)

        for attr in string_attributes:
            self._tokenize_attr(df, attr)

        self.data = torch.as_tensor(df.values, dtype=torch.long)

    def __len__(self) -> int:
        return self.num_records

    def __repr__(self) -> str:
        attributes = ", ".join(self.attributes)
        return f"{self.name}({attributes})"

    @property
    def num_records(self) -> int:
        return self.data.size(0)

    @property
    def num_attributes(self) -> int:
        return self.data.size(1)

    @staticmethod
    def datetime2int(date: str, format: Optional[str] = None) -> int:
        return int(pd.to_datetime(date, format=format).timestamp())

    def _datetime_attr(self, df: pd.DataFrame, attribute: str) -> None:
        df[attribute] = df[attribute].astype("int64") // 10**9

    def _tokenize_attr(self, df: pd.DataFrame, attribute: str) -> None:
        # create dictionary mapping unique values to integers
        # map over rows and replace values with integers
        dictionary = Tokenizer()
        df[attribute] = df[attribute].apply(lambda x: dictionary.add(x))
        self.tokenizers[attribute] = dictionary


class Query(object):
    sql: str
    joins: List[Tuple[str, str, str]]
    selects: List[Tuple[str, str, str]]
    node2component: Dict[str, int]
    num_components: int
    id2joined_attrs: Dict[str, Set[str]]

    def __init__(self, sql: str):
        self.sql = sql

        self.joins = []
        self.selects = []

        for left, op, right, is_select in self.condition_iter():
            if is_select:
                self.selects.append((left, op, right))
            else:
                self.joins.append((left, op, right))

        self.node2component, self.num_components = self.component_labeling(self.joins)

        self.id2joined_attrs: Dict[str, Set[str]] = defaultdict(lambda: set())

        for join in self.joins:
            left, _, right = join

            id, attr = left.split(".")
            self.id2joined_attrs[id].add(attr)

            id, attr = right.split(".")
            self.id2joined_attrs[id].add(attr)

    def __repr__(self) -> str:
        return self.sql

    def table_mapping_iter(self) -> Generator[Tuple[str, str], None, None]:

        table_list = text_between(self.sql, "FROM", "WHERE")
        table_list = table_list.split(",")

        for table in table_list:
            table = table.strip()
            
            # First try splitting on AS otherwise split on space
            splits = re.split(" AS ", table, flags=re.IGNORECASE, maxsplit=1)
            if len(splits) == 1:
                splits = table.split(" ", maxsplit=1)
            
            name, id = splits

            name = name.strip()
            id = id.strip()

            yield id, name

    def condition_iter(self) -> Generator[Tuple[str, str, str, bool], None, None]:

        # remove closing semicolon if present
        if self.sql.endswith(";"):
            sql_query = self.sql[:-1]
        else:
            sql_query = self.sql

        selections = re.split("\sWHERE\s", sql_query)[1]

        if " OR " in selections:
            raise NotImplementedError("OR selections are not supported yet.")

        selections = re.split("\sAND\s", selections)
        # print(selections)

        # TODO support more complicated LIKE and OR statements
        # TODO support for parentheses

        for i, selection in enumerate(selections):
            left, op, right = selection_ops_re.split(selection)
            left = left.strip()
            right = right.strip()

            # With BETWEEN the next AND is part of BETWEEN
            if op == "BETWEEN":
                right += " AND " + selections[i + 1].strip()
                selections.pop(i + 1)

            is_selection = attribute_re.match(right) == None

            if attribute_re.match(left) == None:
                raise NotImplementedError(
                    "Selection values on the left are not supported"
                )

            if not is_selection and op != "=":
                raise ValueError(f"Must be equi-join but got: {op}")

            yield left, op, right, is_selection

    def component_labeling(self, joins: List[Tuple[str, str, str]]) -> Dict[str, int]:
        to_visit: Set[str] = set()
        node2component: Dict[str, int] = {}
        num_components = 0

        for join in joins:
            left, op, right = join

            to_visit.add(left)
            to_visit.add(right)

        def depth_first_search(node: str, component: int):
            node2component[node] = component

            for join in joins:
                left, op, right = join

                # get the other node if this join involves the current node
                # if not then continue to the next join
                if left == node:
                    other = right
                elif right == node:
                    other = left
                else:
                    continue

                # if the other node has already been visited then continue
                if other not in to_visit:
                    continue

                to_visit.remove(other)
                depth_first_search(other, component)

        while len(to_visit) > 0:
            node = to_visit.pop()
            depth_first_search(node, num_components)
            num_components += 1

        return node2component, num_components

    def joins_of(self, table_id: str) -> List[Tuple[str, str, str]]:
        # ensures that left always has the table id attribute
        joins = []

        for join in self.joins:
            left, op, right = join

            id, _ = left.split(".")
            if id == table_id:
                joins.append(join)

            id, _ = right.split(".")
            if id == table_id:
                joins.append((right, op, left))

        return joins

    def joined_nodes(self, table_id: str) -> Set[str]:
        nodes: Set[str] = set()

        for join in self.joins:
            left, _, right = join

            id, _ = left.split(".")
            if id == table_id:
                nodes.add(left)

            id, _ = right.split(".")
            if id == table_id:
                nodes.add(right)

        return nodes

    def joined_with(self, node: str) -> Set[str]:
        nodes: Set[str] = set()

        for join in self.joins:
            left, _, right = join

            if left == node:
                nodes.add(right)

            if right == node:
                nodes.add(left)

        return nodes

    def random_node(self) -> str:
        nodes = list(self.node2component.keys())
        idx = random.randint(0, len(nodes) - 1)
        return nodes[idx]


def load_tables(root: str, benchmark: str, query: Query) -> Dict[str, Table]:

    id2table: Dict[str, Table] = {}

    # Read the SQL definitions of the tables

    if benchmark.startswith("job"):
        schema_path = os.path.join(root, IMDB_DIR, "schematext.sql")
    elif benchmark.startswith("stats"):
        schema_path = os.path.join(
            root, CARDEST_DIR, "datasets", "stats_simplified", "stats.sql"
        )
    else:
        raise ValueError(f"Benchmark '{benchmark}' does not exist.")

    with open(schema_path, "r") as f:
        sql = f.read()

    # Load only each table in the SQL query once

    for id, name in query.table_mapping_iter():

        # For each table check if it was already loaded
        # because we can reference the same data table multiple times

        table = None
        for t in id2table.values():
            if t.name == name:
                table = t
                break

        if table != None:
            id2table[id] = table
            continue

        # If the table was not loaded already load it,
        # try loading it from a pickle cache (faster)

        if benchmark.startswith("job"):
            pickle_path = os.path.join(CACHE_DIR, "imdb", name + ".pkl")
        elif benchmark.startswith("stats"):
            pickle_path = os.path.join(CACHE_DIR, "stats", name + ".pkl")
        else:
            raise ValueError(f"Benchmark '{benchmark}' does not exist.")
        
        if os.path.isfile(pickle_path):
            print("Using cached table:", pickle_path)
            table = torch.load(pickle_path)
            id2table[id] = table
            continue

        # Otherwise, load it from the csv files (slower)

        # Read the SQL definition of the table
        idx = sql.index(f"CREATE TABLE {name}")
        idx_start = sql.index("(", idx)
        idx_end = sql.index(");", idx)
        attributes = sql[idx_start + 1 : idx_end]
        attributes = attributes.split(",")
        # Creates list of (attribute_name, type)
        attributes = [tuple(a.strip().split(" ", 1)) for a in attributes]

        if benchmark.startswith("job"):
            data_path = os.path.join(root, IMDB_DIR, name + ".csv")
        elif benchmark.startswith("stats"):
            data_path = os.path.join(
                root, CARDEST_DIR, "datasets", "stats_simplified", name + ".csv"
            )
        else:
            raise ValueError(f"Benchmark '{benchmark}' does not exist.")

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

        if benchmark.startswith("stats"):
            df = pd.read_csv(
                data_path,
                header=0,
                parse_dates=datetime_attributes,
                encoding='utf-8', 
                sep=",",
                names=attribute_names,
                dtype=dtypes,
            )

        elif benchmark.startswith("job"):
            with open(data_path, "r") as f:
                # TODO: memory usage could be improved by replacing unescaped variable
                # in an iterator fashion instead of all at ones.
                data = f.read()
                
            # Replace the escaped quotes by double quotes to fix parsing errors
            data = escaped_backslash_re.sub("\"\"", data)

            # These lines cause trouble because they end with a backslash before the final quote
            if name == "movie_info":
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
        id2table[id] = table

    return id2table


def make_selection_filters(
    id2table: Dict[str, Table], query: Query
) -> Dict[str, torch.Tensor]:
    id2mask: Dict[str, torch.Tensor] = {}

    for select in query.selects:
        left, op, right = select

        id, attr = left.split(".")
        table = id2table[id]
        attr_idx = table.attribute2idx[attr]

        if right.endswith("::timestamp"):
            timestamp = right[1 : -len("'::timestamp")]
            value = table.datetime2int(timestamp)

        elif is_number(right):
            value = float(right) if "." in right else int(right)

        elif right.startswith(("'", '"')) and right.endswith(("'", '"')):
            value = table.tokenizers[attr][right[1:-1]]

        else:
            raise ValueError(f"Not sure how to handle right value: {right}")

        if op == "=":
            mask = table.data[:, attr_idx] == value
        elif op == "<>":
            mask = table.data[:, attr_idx] != value
        elif op == ">":
            mask = table.data[:, attr_idx] > value
        elif op == "<":
            mask = table.data[:, attr_idx] < value
        elif op == "<=":
            mask = table.data[:, attr_idx] <= value
        elif op == ">=":
            mask = table.data[:, attr_idx] >= value

        # Ensure that the NULL values are removed from the column
        # because any condition with NULL is false
        mask &= table.data[:, attr_idx] != NULL_VALUE

        if id in id2mask:
            # Assumes all the selections are AND together
            id2mask[id] &= mask
        else:
            id2mask[id] = mask

    # Ensure that the NULL values are removed from the joined columns
    for join in query.joins:
        left, op, right = join

        id, attr = left.split(".")
        table = id2table[id]
        attr_idx = table.attribute2idx[attr]

        mask = table.data[:, attr_idx] != NULL_VALUE

        if id in id2mask:
            # Assumes all the selections are AND together
            id2mask[id] &= mask
        else:
            id2mask[id] = mask

        id, attr = right.split(".")
        table = id2table[id]
        attr_idx = table.attribute2idx[attr]

        mask = table.data[:, attr_idx] != NULL_VALUE

        if id in id2mask:
            # Assumes all the selections are AND together
            id2mask[id] &= mask
        else:
            id2mask[id] = mask

    return id2mask


def prepare_batches(
    id2table: Dict[str, Table],
    id2mask: Dict[str, torch.Tensor],
    batch_size: int,
    query: Query,
) -> Dict[str, List[torch.Tensor]]:

    node2batches: Dict[str, List[torch.Tensor]] = {}

    # Capture the set of all unique nodes
    nodes: Set[str] = set()
    for join in query.joins:
        left, _, right = join
        nodes.add(left)
        nodes.add(right)

    # For each node load its data
    for node in nodes:

        id, attr = node.split(".")
        table = id2table[id]

        attr_idx = table.attribute2idx[attr]
        attr_data = table.data[:, attr_idx]

        mask = id2mask.get(id, None)
        if mask != None:
            attr_data = attr_data[mask]

        attr_batches = attr_data.split(batch_size)
        node2batches[node] = attr_batches

    return node2batches


def combine_sketches(
    node: str, visited: Set[str], query: Query, id2sketch: Dict[str, torch.Tensor]
) -> torch.Tensor:
    id, _ = node.split(".")
    sketch = id2sketch[id]
    visited.add(node)

    for other_node in query.joined_nodes(id):
        # skip the current node
        if other_node == node:
            continue

        visited.add(other_node)

        tmp = 1
        for joined_node in query.joined_with(other_node):
            tmp = tmp * combine_sketches(joined_node, visited, query, id2sketch)

        # efficient circular cross-correlation
        sketch = ifft(fft(tmp).conj() * fft(sketch)).real

    for joined_node in query.joined_with(node).difference(visited):
        sketch = sketch * combine_sketches(joined_node, visited, query, id2sketch)

    return sketch


def ams_estimate(id2sketch: Dict[str, torch.Tensor], means: int) -> torch.Tensor:
    estimates = 1
    for sketch in id2sketch.values():
        estimates = estimates * sketch

    if means == 1:
        return estimates.float()

    estimates = estimates.view(-1, means)
    return torch.mean(estimates, dim=1, dtype=torch.float)


def merge_sketches(id2sketch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    id2merged_sketch = {}

    for id, sep_sketches in id2sketch.items():

        num_estimates, num_joins, num_bins = sep_sketches.shape

        if num_joins == 1:
            id2merged_sketch[id] = sep_sketches.squeeze(1)
            continue

        reshape_size = [num_estimates] + [1] * num_joins
        expand_size = [num_estimates] + [num_bins] * num_joins

        sketches = torch.empty(num_joins, *expand_size, dtype=torch.long)
        for i in range(num_joins):
            size = copy.deepcopy(reshape_size)
            size[i + 1] = num_bins

            sk = sep_sketches[:, i].reshape(size)
            sk = sk.expand(expand_size)
            sketches[i] = sk

        # The Numpy argmin is about 10x faster than PyTorch
        # index = sketches_abs.argmin(dim=0)
        index = numpy.argmin(sketches.abs().numpy(), axis=0)
        index = torch.from_numpy(index)
        index.unsqueeze_(0)

        merged_sketch = torch.gather(sketches, 0, index)
        merged_sketch.squeeze_(0)

        id2merged_sketch[id] = merged_sketch

    return id2merged_sketch


def compass_estimate(id2sketch: Dict[str, torch.Tensor], query: Query) -> torch.Tensor:
    # all einsum indices start with ellipses for the batch dimension
    # that contians the medians and estimates
    id2einsum_indices = defaultdict(lambda: [...])

    for idx, join in enumerate(query.joins):
        left, _, right = join

        id = left.split(".")[0]
        id2einsum_indices[id].append(idx)

        id = right.split(".")[0]
        id2einsum_indices[id].append(idx)

    einsum_args = []
    for id in id2sketch.keys():
        einsum_args.append(id2sketch[id])
        einsum_args.append(id2einsum_indices[id])

    # Add ellipses to keep the batch dimension intact
    einsum_args.append([...])

    return torch.einsum(*einsum_args)


def exact_estimate(
    id2table: Dict[str, Table], id2mask: Dict[str, torch.Tensor], query: Query
) -> torch.Tensor:

    visited: Set[str] = set()
    to_visit: deque[str] = deque()

    # Select a random table to start with
    id = random.choice(list(id2table.keys()))
    table = id2table[id]
    to_visit.append(id)

    # Mapping from nodes to their column in tmp_data
    node2idx: Dict[str, int] = {}
    attr_idxs = []
    for attr in query.id2joined_attrs[id]:
        attr_idxs.append(table.attribute2idx[attr])
        node2idx[f"{id}.{attr}"] = len(node2idx)

    tmp_data = table.data[:, attr_idxs]

    mask = id2mask.get(id, None)
    if mask != None:
        tmp_data = tmp_data[mask]

    tmp_data = tmp_data.tolist()

    # Traverse the join graph in a breath-first fashion
    while len(to_visit) > 0:
        id = to_visit.popleft()
        visited.add(id)

        for join in query.joins_of(id):
            left, _, right = join

            id, joined_attr = right.split(".")
            if id in visited:
                continue

            table = id2table[id]
            to_visit.append(id)

            attr_idxs = []
            for attr in query.id2joined_attrs[id]:
                attr_idxs.append(table.attribute2idx[attr])
                node2idx[f"{id}.{attr}"] = len(node2idx)

                # Keep track of the column of table_data to join
                if attr == joined_attr:
                    joined_idx = len(attr_idxs) - 1

            table_data = table.data[:, attr_idxs]

            mask = id2mask.get(id, None)
            if mask != None:
                table_data = table_data[mask]

            table_data = table_data.tolist()

            tmp_data = hash_join(tmp_data, node2idx[left], table_data, joined_idx)

    count = 0
    for _ in tqdm(tmp_data):
        count += 1

    return torch.tensor([count], dtype=torch.long)


def median_trick(estimates: torch.Tensor, medians: int) -> torch.Tensor:
    """Takes a tensor of iid estimates and returns the median among groups"""
    if medians == 1:
        return estimates

    estimates = estimates.view(-1, medians)
    return torch.median(estimates, dim=1).values


def hash_join(larger_table, larger_idx, smaller_table, smaller_idx) -> List[List[Any]]:

    inner_row_by_key = defaultdict(lambda: [])

    for inner_row in smaller_table:
        inner_row_by_key[inner_row[smaller_idx]].append(inner_row)

    for outer_row in larger_table:
        for inner_row in inner_row_by_key[outer_row[larger_idx]]:
            yield outer_row + inner_row


def experiment():

    args = Arguments(underscores_to_dashes=True).parse_args()

    # Get random ID before setting the seed
    exp_id = random_string()
    date_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    os.makedirs(args.result_dir, exist_ok=True)
    result_path = os.path.join(args.result_dir, f"{date_str}-multijoin-{exp_id}.csv")
    print("Saving results in:", result_path)

    if args.seed is not None:
        seed(args.seed)
        print("Seed", args.seed)

    print("Reading SQL query...")
    benchmark, query_name = args.query.split("-")
    sql_query = read_sql_query(args.data_dir, benchmark, query_name)
    query = Query(sql_query)
    print(query)
    print("Filters:", query.selects)
    print("Joins:", query.joins)
    print(f"Components ({query.num_components}):", query.node2component)

    print("Loading tables...")
    loading_timer = Timer()
    id2table = load_tables(args.data_dir, benchmark, query)
    loading_time = loading_timer.stop()
    print(id2table)

    # Keep track of how much memory is required by
    # the hash functions and the sketches
    memory_usage = 0

    init_timer = Timer()

    print("Initialize hash functions...")
    if args.method == "ams":
        num_estimates = args.estimates * args.medians * args.means

        sign_hashes = []
        for _ in query.joins:
            sign_hash = SignHash(num_estimates, k=4)
            memory_usage += sign_hash.fn.seeds.numel() * 8
            sign_hashes.append(sign_hash)

    elif args.method == "count-conv":
        num_estimates = args.estimates * args.medians

        sign_hashes = []
        for _ in query.joins:
            sign_hash = SignHash(num_estimates, k=4)
            memory_usage += sign_hash.fn.seeds.numel() * 8
            sign_hashes.append(sign_hash)

        bin_hashes = []
        for _ in range(query.num_components):
            bin_hash = BinHash(num_estimates, bins=args.bins, k=2)
            memory_usage += bin_hash.fn.seeds.numel() * 8
            bin_hashes.append(bin_hash)

    elif "compass-" in args.method:
        num_estimates = args.estimates * args.medians

        sign_hashes = []
        bin_hashes = []
        for _ in query.joins:
            sign_hash = SignHash(num_estimates, k=4)
            memory_usage += sign_hash.fn.seeds.numel() * 8
            sign_hashes.append(sign_hash)

            bin_hash = BinHash(num_estimates, bins=args.bins, k=2)
            memory_usage += bin_hash.fn.seeds.numel() * 8
            bin_hashes.append(bin_hash)

    print("Initialize sketches...")
    id2sketch: Dict[str, torch.Tensor] = {}
    for id in id2table.keys():

        if args.method == "ams":
            id2sketch[id] = torch.zeros(num_estimates, dtype=torch.long)

        elif args.method == "count-conv":
            id2sketch[id] = torch.zeros(num_estimates, args.bins, dtype=torch.long)

        elif args.method == "compass-partition":
            # creates a seperate sketch dimension for ever join with a table
            num_joins_with_table = len(query.joins_of(id))
            size = (args.bins,) * num_joins_with_table
            id2sketch[id] = torch.zeros(num_estimates, *size, dtype=torch.long)

        elif args.method == "compass-merge":
            # creates a seperate sketch for ever join with a table
            num_joins_with_table = len(query.joins_of(id))
            size = (num_joins_with_table, args.bins)
            id2sketch[id] = torch.zeros(num_estimates, *size, dtype=torch.long)

        if args.method != "exact":
            memory_usage += id2sketch[id].numel() * 8

    init_time = init_timer.stop()

    stream_timer = Timer()

    print("Applying filters...")
    id2mask = make_selection_filters(id2table, query)
    node2batches = prepare_batches(id2table, id2mask, args.batch_size, query)

    print("Filling sketches...")

    if args.method == "ams":

        for id in id2table:

            # Align data with hash indices

            batches = []
            join_indices = []

            for join_idx, join in enumerate(query.joins):
                left, _, right = join

                if left.split(".")[0] == id:
                    batches.append(node2batches[left])
                    join_indices.append(join_idx)

                elif right.split(".")[0] == id:
                    batches.append(node2batches[right])
                    join_indices.append(join_idx)

            # Add data to sketch

            for batch in tqdm(zip(*batches), total=len(batches[0])):

                signs = 1
                for attr_values, join_idx in zip(batch, join_indices):
                    sign_hash = sign_hashes[join_idx]
                    signs = signs * sign_hash(attr_values)

                id2sketch[id] += torch.sum(signs, dim=1)

    elif args.method == "count-conv":

        for id in id2table:

            # Align data with hash indices

            batches = []
            all_join_indices = []
            components = []

            for attr in query.id2joined_attrs[id]:
                node = f"{id}.{attr}"
                batches.append(node2batches[node])
                components.append(query.node2component[node])

                join_indices = []

                for join_idx, join in enumerate(query.joins):
                    left, _, right = join

                    if left == node or right == node:
                        join_indices.append(join_idx)

                all_join_indices.append(join_indices)

            # Add data to sketch

            for batch in zip(*batches):

                signs = 1
                bins = 0

                # For each joined attribute of table id
                for attr_values, join_indices, component in zip(
                    batch, all_join_indices, components
                ):

                    bin_hash = bin_hashes[component]
                    bins = bins + bin_hash(attr_values)

                    # For each join with attribute
                    for join_idx in join_indices:

                        sign_hash = sign_hashes[join_idx]
                        signs = signs * sign_hash(attr_values)

                bins = bins % args.bins
                id2sketch[id].scatter_add_(1, bins, signs)

    elif args.method == "compass-partition":

        for id in id2table:

            # Align data with hash indices

            batches = []
            join_indices = []

            for join_idx, join in enumerate(query.joins):
                left, _, right = join

                if left.split(".")[0] == id:
                    batches.append(node2batches[left])
                    join_indices.append(join_idx)

                elif right.split(".")[0] == id:
                    batches.append(node2batches[right])
                    join_indices.append(join_idx)

            # Add data to sketches

            sketch = id2sketch[id].view(num_estimates, -1)

            for batch in zip(*batches):

                bins = 0
                signs = 1

                # The order is arbitrarily determined by that of the joins in the SQL
                dimension_idx = len(join_indices) - 1
                for attr_values, join_idx in zip(batch, join_indices):

                    bin_scale = args.bins**dimension_idx
                    dimension_idx -= 1

                    bin_hash = bin_hashes[join_idx]
                    bins = bins + bin_hash(attr_values) * bin_scale

                    sign_hash = sign_hashes[join_idx]
                    signs = signs * sign_hash(attr_values)

                sketch.scatter_add_(1, bins, signs)

    elif args.method == "compass-merge":

        for id in id2table:

            sketch_idx = 0
            for join_idx, join in enumerate(query.joins):
                left, _, right = join

                if left.split(".")[0] == id:
                    node = left
                elif right.split(".")[0] == id:
                    node = right
                else:
                    continue

                bin_hash = bin_hashes[join_idx]
                sign_hash = sign_hashes[join_idx]
                sketch = id2sketch[id][:, sketch_idx]
                sketch_idx += 1

                for attr_values in node2batches[node]:
                    bins = bin_hash(attr_values)
                    signs = sign_hash(attr_values)
                    sketch.scatter_add_(1, bins, signs)

    stream_time = stream_timer.stop()

    inference_timer = Timer()

    print("Estimating cardinality...")
    if args.method == "ams":
        estimates = ams_estimate(id2sketch, args.means)

    elif args.method == "count-conv":
        start_node = query.random_node()
        visited: Set[str] = set()

        estimates = combine_sketches(start_node, visited, query, id2sketch)
        estimates = torch.sum(estimates, dim=1)

    elif args.method == "compass-partition":
        estimates = compass_estimate(id2sketch, query)

    elif args.method == "compass-merge":
        id2merged_sketch = merge_sketches(id2sketch)
        estimates = compass_estimate(id2merged_sketch, query)

    elif args.method == "exact":
        estimates = exact_estimate(id2table, id2mask, query)

    if args.method != "exact":
        # Refine estimate using the median trick
        estimates = median_trick(estimates, args.medians)

    inference_time = inference_timer.stop()

    print("Data loading time:", loading_time)
    print("Initialization time:", init_time)
    print("Sketching time:", stream_time)
    print("Inference time:", inference_time)
    print("Total time:", init_time + stream_time + inference_time)

    estimates = estimates.tolist()
    print(estimates)

    fieldnames = [
        "method",
        "query",
        "batch_size",
        "seed",
        "bins",
        "means",
        "medians",
        "estimate",
        "memory_usage",
        "init_time",
        "stream_time",
        "inference_time",
    ]

    with open(result_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for estimate in estimates:
            writer.writerow(
                {
                    "method": args.method,
                    "query": args.query,
                    "batch_size": args.batch_size,
                    "seed": args.seed,
                    "bins": args.bins,
                    "means": args.means,
                    "medians": args.medians,
                    "estimate": estimate,
                    "memory_usage": memory_usage,
                    "init_time": init_time,
                    "stream_time": stream_time,
                    "inference_time": inference_time,
                }
            )


if __name__ == "__main__":
    experiment()
