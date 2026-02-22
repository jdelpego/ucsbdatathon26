"""Fast filtered count: FD courts, last 10 years, with attorneys & judges."""

import pyarrow.dataset as ds
import pyarrow.compute as pc
import datetime
import time
from functools import reduce

COURT_TYPE = "FD"
YEARS_BACK = 10
NOT_NULL_FIELDS = ["attorneys", "judges"]
FILTER_ONLY_NOT_NULL = ["opinions"]  # filter but don't load (heavy column)

cutoff = datetime.date.today() - datetime.timedelta(days=YEARS_BACK * 365)

filters = reduce(
    lambda a, b: a & b,
    [
        pc.field("court_type") == COURT_TYPE,
        pc.field("date_filed") >= cutoff,
        *(pc.field(f).is_valid() for f in NOT_NULL_FIELDS),
        *(pc.field(f).is_valid() for f in FILTER_ONLY_NOT_NULL),
    ],
)

columns = ["court_type", "date_filed"] + NOT_NULL_FIELDS

t0 = time.time()
dataset = ds.dataset("hf_data", format="parquet", exclude_invalid_files=True)
table = dataset.to_table(columns=columns, filter=filters)
elapsed = time.time() - t0

not_null_str = " & ".join(NOT_NULL_FIELDS + FILTER_ONLY_NOT_NULL)
print(
    f"Scanned in {elapsed:.1f}s\n"
    f"Filters: court_type={COURT_TYPE}, {not_null_str} NOT NULL, "
    f"date_filed >= {cutoff}\n"
    f"Matched: {len(table):,} rows"
)
