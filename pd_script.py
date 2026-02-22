from datasets import load_dataset

dataset = load_dataset(
    "parquet",
    data_files="./hf_data/*.parquet",
    split="train"
)

print(dataset[:100])