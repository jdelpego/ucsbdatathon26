"""Test: check opinion_text not null per row using vectorized approach."""
import pyarrow.parquet as pq
import pyarrow.compute as pc
import pyarrow as pa
import numpy as np
import glob


def has_valid_opinion_text(opinions_chunked):
    """Return boolean array: True if row has at least one non-null opinion_text."""
    arr = opinions_chunked.chunk(0)
    n = len(arr)

    offsets_np = np.frombuffer(
        arr.offsets.buffers()[1], dtype=np.int32
    )[: n + 1].copy()

    flat_structs = arr.values
    opinion_texts = flat_structs.field("opinion_text")
    text_valid = pc.is_valid(opinion_texts).to_numpy(
        zero_copy_only=False
    ).astype(np.int64)

    cumsum = np.zeros(len(text_valid) + 1, dtype=np.int64)
    cumsum[1:] = np.cumsum(text_valid)
    row_sums = cumsum[offsets_np[1:]] - cumsum[offsets_np[:-1]]

    return pc.and_(pc.is_valid(opinions_chunked), pa.array(row_sums > 0))


f = sorted(glob.glob("./hf_data/*.parquet"))[0]
pf = pq.ParquetFile(f)

# Read one row group (avoids nested chunked array issue with iter_batches)
rg = pf.read_row_group(0, columns=["opinions"])
opinions = rg.column("opinions")
print("type:", type(opinions))
print("chunks:", opinions.num_chunks)
print("null count:", opinions.chunk(0).null_count)
print("total rows:", len(opinions))

result = has_valid_opinion_text(opinions)
print("has_valid_text count true:", pc.sum(result).as_py())
print("has_valid_text count false:", len(opinions) - pc.sum(result).as_py())
