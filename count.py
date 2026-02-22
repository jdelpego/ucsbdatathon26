"""Quick count tool for the main dataset with arbitrary filters.

Usage:
    python3 count.py                                  # total row count
    python3 count.py court_type=FD                    # equality filter
    python3 count.py court_type=FD "court_jurisdiction=USA, Federal"
    python3 count.py court_type                       # value_counts for a column
    python3 count.py "attorneys!=NULL"                 # NOT NULL check
    python3 count.py "date_filed>=2001-01-01"         # date/value comparison
    python3 count.py --has-opinion                    # at least one opinion with text
    python3 count.py court_type=ST "attorneys!=NULL" "judges!=NULL" \\
        "date_filed>=2001-01-01" "court_short_name!=NULL" \\
        "court_full_name!=NULL" --has-opinion          # full combo
"""

import pyarrow.parquet as pq
import pyarrow.compute as pc
import pyarrow as pa
import glob
import sys
import time
import datetime

parquet_files = sorted(glob.glob("./hf_data/*.parquet"))
print(f"Found {len(parquet_files)} parquet files\n")

args = sys.argv[1:]

# Parse arguments into filter types and group-by columns
eq_filters = {}       # col=val  (equality)
notnull_filters = []  # col!=NULL
gte_filters = {}      # col>=val
group_cols = []
has_opinion = False

for arg in args:
    if arg == "--has-opinion":
        has_opinion = True
    elif "!=" in arg:
        k, v = arg.split("!=", 1)
        if v.upper() == "NULL":
            notnull_filters.append(k)
    elif ">=" in arg:
        k, v = arg.split(">=", 1)
        gte_filters[k] = v
    elif "=" in arg:
        k, v = arg.split("=", 1)
        eq_filters[k] = v
    else:
        group_cols.append(arg)

# Figure out which columns to read (None = all, needed for opinions)
if has_opinion:
    needed_cols = None  # opinions is nested, just read everything
else:
    cols = set(eq_filters.keys()) | set(notnull_filters) | set(gte_filters.keys()) | set(group_cols)
    needed_cols = list(cols) if cols else None

t0 = time.time()
total_rows = 0
matched_rows = 0
value_counts = {}  # col -> {value: count}

for fpath in parquet_files:
    pf = pq.ParquetFile(fpath)

    for batch in pf.iter_batches(batch_size=100_000, columns=needed_cols):
        table = pa.Table.from_batches([batch])
        total_rows += len(table)

        # Build combined arrow mask
        mask = None

        # Equality filters
        for col, val in eq_filters.items():
            cond = pc.equal(table.column(col), val)
            mask = cond if mask is None else pc.and_(mask, cond)

        # NOT NULL filters
        for col in notnull_filters:
            cond = pc.is_valid(table.column(col))
            mask = cond if mask is None else pc.and_(mask, cond)

        # >= filters (auto-detect date columns)
        for col, val in gte_filters.items():
            try:
                dt = datetime.date.fromisoformat(val)
                cond = pc.and_(pc.is_valid(table.column(col)),
                               pc.greater_equal(table.column(col), dt))
            except ValueError:
                cond = pc.greater_equal(table.column(col), val)
            mask = cond if mask is None else pc.and_(mask, cond)

        if mask is not None:
            table = table.filter(mask)

        # Opinion check (Python loop on remaining rows only)
        if has_opinion and len(table) > 0:
            opinions_col = table.column("opinions")
            keep = []
            for i in range(len(table)):
                ops = opinions_col[i].as_py()
                if ops and any(op.get("opinion_text") is not None for op in ops):
                    keep.append(i)
            table = table.take(keep) if keep else table.slice(0, 0)

        matched_rows += len(table)

        # Accumulate value counts for group-by columns
        for col in group_cols:
            if col not in value_counts:
                value_counts[col] = {}
            for val in table.column(col).to_pylist():
                key = str(val)
                value_counts[col][key] = value_counts[col].get(key, 0) + 1

    elapsed = time.time() - t0
    print(f"  progress: {total_rows:,} rows scanned | {elapsed:.1f}s", end="\r")

elapsed = time.time() - t0
print(f"\nScanned {total_rows:,} total rows in {elapsed:.1f}s")

desc = []
if eq_filters:
    desc += [f"{k}={v}" for k, v in eq_filters.items()]
if notnull_filters:
    desc += [f"{k} IS NOT NULL" for k in notnull_filters]
if gte_filters:
    desc += [f"{k}>={v}" for k, v in gte_filters.items()]
if has_opinion:
    desc.append("has opinion_text")
if desc:
    print(f"Filters: {', '.join(desc)}")
print(f"Matched: {matched_rows:,} rows")

for col in group_cols:
    counts = value_counts[col]
    sorted_counts = sorted(counts.items(), key=lambda x: -x[1])
    print(f"\n--- {col} value counts ({len(sorted_counts)} unique) ---")
    for val, cnt in sorted_counts:
        print(f"  {val}: {cnt:,}")
