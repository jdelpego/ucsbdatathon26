import os
import duckdb
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.environ["HF_TOKEN"]

PARQUET_URL = "hf://datasets/harvard-lil/cold-cases/**/*.parquet"
OUTPUT_FILE = "cold_cases_fd.parquet"

con = duckdb.connect()
con.execute(f"CREATE SECRET (TYPE HUGGINGFACE, TOKEN '{HF_TOKEN}')")

print("Downloading 100 Federal District cases (court_type='FD', court_jurisdiction='USA, Federal')...")
print("This queries remote parquet with predicate pushdown — only matching rows are transferred.\n")

con.execute(f"""
    COPY (
        SELECT *
        FROM '{PARQUET_URL}'
        WHERE court_type = 'FD'
          AND court_jurisdiction = 'USA, Federal'
        LIMIT 100
    ) TO '{OUTPUT_FILE}' (FORMAT PARQUET, COMPRESSION ZSTD)
""")

# Verify what we saved
result = con.execute(f"""
    SELECT count(*) AS total_rows,
           count(DISTINCT court_short_name) AS distinct_courts,
           min(date_filed) AS earliest,
           max(date_filed) AS latest
    FROM '{OUTPUT_FILE}'
""").fetchdf()

print(f"Saved to {OUTPUT_FILE}")
print(result.to_string(index=False))

con.close()