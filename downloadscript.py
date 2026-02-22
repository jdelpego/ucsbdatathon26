import duckdb
import os
import multiprocessing
from dotenv import load_dotenv

load_dotenv()

# -----------------------------
# CONFIG
# -----------------------------
HF_TOKEN = os.environ["HF_TOKEN"]
PARQUET_URL = "hf://datasets/harvard-lil/cold-cases/*.parquet"
DB_PATH = "cold_cases.duckdb"

# -----------------------------
# CONNECT TO PERSISTENT DB
# -----------------------------
print("Connecting to DuckDB...")
con = duckdb.connect(DB_PATH)

# Use all available cores
num_threads = multiprocessing.cpu_count()
con.execute(f"PRAGMA threads={num_threads};")
con.execute("PRAGMA enable_progress_bar;")

# -----------------------------
# LOAD REQUIRED EXTENSION
# -----------------------------
print("Installing/loading httpfs extension...")
con.execute("INSTALL httpfs;")
con.execute("LOAD httpfs;")

# -----------------------------
# SET HUGGING FACE SECRET
# -----------------------------
print("Setting Hugging Face token...")
con.execute(f"""
CREATE OR REPLACE SECRET (
    TYPE HUGGINGFACE,
    TOKEN '{HF_TOKEN}'
)
""")

# -----------------------------
# QUICK CONNECTIVITY TEST
# -----------------------------
print("Testing HF connection...")
test = con.execute(f"""
SELECT count(*) 
FROM '{PARQUET_URL}'
""").fetchall()

print(f"Dataset row count: {test[0][0]}")
print("HF connection confirmed.\n")

# -----------------------------
# STEP 1: CREATE BASE TABLE
# -----------------------------
print("Creating base table (one-time operation)...")

con.execute(f"""
CREATE TABLE IF NOT EXISTS base AS
SELECT *
FROM '{PARQUET_URL}'
WHERE court_type = 'FD'
  AND court_jurisdiction = 'USA, Federal'
  AND date_filed >= '2001-01-01'
  AND attorneys IS NOT NULL
  AND judges IS NOT NULL
  AND court_short_name IS NOT NULL
  AND court_full_name IS NOT NULL;
""")

print("Base table ready.\n")

# -----------------------------
# STEP 2: FILTER OPINIONS (LOCAL)
# -----------------------------
print("Creating opinion-filtered table (one-time operation)...")

con.execute("""
CREATE TABLE IF NOT EXISTS base_with_opinion AS
SELECT *
FROM base
WHERE list_any(opinions, x -> x.opinion_text IS NOT NULL);
""")

print("Opinion-filtered table ready.\n")

# -----------------------------
# STEP 3: FINAL AGGREGATION
# -----------------------------
print("Running final aggregation...")

result = con.execute("""
SELECT count(*) AS total_rows,
       count(DISTINCT court_short_name) AS distinct_courts,
       min(date_filed) AS earliest,
       max(date_filed) AS latest
FROM base_with_opinion;
""").fetchdf()

print("\nFinal Result:")
print(result.to_string(index=False))

con.close()
print("\nDone.")