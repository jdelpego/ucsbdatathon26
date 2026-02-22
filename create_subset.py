"""Create a parquet subset: FD courts, last 10 years, with attorneys, judges & opinions."""

import pyarrow.dataset as ds
import pyarrow.compute as pc
import pyarrow.parquet as pq
import datetime
import time
from functools import reduce

COURT_TYPE = "FD"
YEARS_BACK = 10
NOT_NULL_FIELDS = ["attorneys", "judges", "opinions"]
OUTPUT_FILE = "subset.parquet"

cutoff = datetime.date.today() - datetime.timedelta(days=YEARS_BACK * 365)

filters = reduce(
    lambda a, b: a & b,
    [
        pc.field("court_type") == COURT_TYPE,
        pc.field("date_filed") >= cutoff,
        *(pc.field(f).is_valid() for f in NOT_NULL_FIELDS),
    ],
)

t0 = time.time()
dataset = ds.dataset("hf_data", format="parquet", exclude_invalid_files=True)
table = dataset.to_table(filter=filters)
elapsed_read = time.time() - t0

t1 = time.time()
pq.write_table(table, OUTPUT_FILE)
elapsed_write = time.time() - t1

not_null_str = " & ".join(NOT_NULL_FIELDS)
print(
    f"Read in {elapsed_read:.1f}s, wrote in {elapsed_write:.1f}s\n"
    f"Filters: court_type={COURT_TYPE}, {not_null_str} NOT NULL, "
    f"date_filed >= {cutoff}\n"
    f"Rows: {len(table):,}\n"
    f"Output: {OUTPUT_FILE}"
)
