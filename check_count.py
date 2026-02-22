import duckdb, os
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.environ["HF_TOKEN"]

con = duckdb.connect()
con.execute(f"CREATE SECRET (TYPE HUGGINGFACE, TOKEN '{HF_TOKEN}')")

PARQUET_URL = "hf://datasets/harvard-lil/cold-cases/**/*.parquet"

print("Counting rows matching all filters...")
r = con.execute(f"""
    SELECT count(*) AS total_rows,
           count(DISTINCT court_short_name) AS distinct_courts,
           min(date_filed) AS earliest,
           max(date_filed) AS latest
    FROM '{PARQUET_URL}'
    WHERE court_type = 'FD'
      AND court_jurisdiction = 'USA, Federal'
      AND attorneys IS NOT NULL
      AND judges IS NOT NULL
      AND date_filed >= '2001-01-01'
      AND court_short_name IS NOT NULL
      AND court_full_name IS NOT NULL
      AND len(list_filter(opinions, x -> x.opinion_text IS NOT NULL)) > 0
""").fetchdf()
print(r.to_string(index=False))
con.close()
