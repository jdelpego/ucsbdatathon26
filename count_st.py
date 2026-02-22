import pyarrow.parquet as pq
import pyarrow as pa
import pyarrow.compute as pc
import glob, time, datetime

parquet_files = sorted(glob.glob("./hf_data/*.parquet"))
t0 = time.time()
total = 0
kept = 0

for fpath in parquet_files:
    pf = pq.ParquetFile(fpath)
    for batch in pf.iter_batches(batch_size=100_000):
        table = pa.Table.from_batches([batch])
        total += len(table)

        conditions = [
            pc.equal(table.column("court_type"), "ST"),
            pc.is_valid(table.column("attorneys")),
            pc.is_valid(table.column("judges")),
            pc.is_valid(table.column("date_filed")),
            pc.greater_equal(table.column("date_filed"), datetime.date(2001, 1, 1)),
            pc.is_valid(table.column("court_short_name")),
            pc.is_valid(table.column("court_full_name")),
        ]
        mask = conditions[0]
        for c in conditions[1:]:
            mask = pc.and_(mask, c)
        filtered = table.filter(mask)

        if len(filtered) == 0:
            continue

        opinions_col = filtered.column("opinions")
        for i in range(len(filtered)):
            ops = opinions_col[i].as_py()
            if ops and any(op.get("opinion_text") is not None for op in ops):
                kept += 1

        print(f"  scanned {total:,} | kept {kept:,} | {time.time()-t0:.1f}s", end="\r")

print(f"\nDone. Scanned {total:,} rows, matched {kept:,} in {time.time()-t0:.1f}s")
