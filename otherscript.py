import pyarrow.parquet as pq
import pyarrow as pa
import pyarrow.compute as pc
import glob
import sys
import time
import datetime

TEST_MODE = "--test" in sys.argv

parquet_files = sorted(glob.glob("./hf_data/*.parquet"))
print(f"Found {len(parquet_files)} parquet files")

out_path = "./filtered_federal_subset.parquet"
writer = None
total_kept = 0
total_scanned = 0
t0 = time.time()

for fpath in parquet_files:
    pf = pq.ParquetFile(fpath)
    for batch in pf.iter_batches(batch_size=50_000):
        table = pa.Table.from_batches([batch])
        total_scanned += len(table)

        # Fast arrow-level filters (no Python row loop)
        conditions = [
            pc.equal(table.column("court_type"), "FD"),
            pc.equal(table.column("court_jurisdiction"), "USA, Federal"),
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

        # Opinion check — must iterate the nested list column
        opinions_col = filtered.column("opinions")
        keep_indices = []
        for i in range(len(filtered)):
            ops = opinions_col[i].as_py()
            if ops and any(op.get("opinion_text") is not None for op in ops):
                keep_indices.append(i)

        if not keep_indices:
            continue

        subset = filtered.take(keep_indices)
        total_kept += len(subset)

        if writer is None:
            writer = pq.ParquetWriter(out_path, subset.schema, compression="zstd")
        writer.write_table(subset)

        elapsed = time.time() - t0
        print(f"  scanned {total_scanned:,} | kept {total_kept:,} | {elapsed:.1f}s", end="\r")

        if TEST_MODE and total_scanned >= 100_000:
            print(f"\n[TEST] Stopping after {total_scanned:,} rows scanned, {total_kept:,} kept.")
            break

    if TEST_MODE and total_scanned >= 100_000:
        break

if writer:
    writer.close()

elapsed = time.time() - t0
print(f"\nDone. Scanned {total_scanned:,} rows, kept {total_kept:,} in {elapsed:.1f}s")
print(f"Output: {out_path}")