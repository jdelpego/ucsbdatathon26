import pyarrow.parquet as pq

table = pq.read_table("./subset.parquet")
print(f"Total rows: {len(table)}\n")

for i in range(min(10, len(table))):
    row = {col: table.column(col)[i].as_py() for col in table.column_names}
    print(f"--- Row {i+1} ---")
    for k, v in row.items():
        if k == "opinions":
            print(f"  {k}: [{len(v) if v else 0} opinion(s)]")
        else:
            val = str(v)[:120] if v is not None else None
            print(f"  {k}: {val}")
    print()
