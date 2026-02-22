from datasets import load_dataset
from collections import defaultdict, Counter
import random

# -----------------------------------
# 1. Load dataset
# -----------------------------------

ds = load_dataset("parquet", data_files="subset.parquet")["train"]

# -----------------------------------
# 2. Build nested hashmap (counts)
# jurisdiction -> judge -> count
# -----------------------------------

counts = defaultdict(Counter)

for jurisdiction, judge in zip(ds["court_jurisdiction"], ds["judges"]):
    counts[jurisdiction][judge] += 1

# -----------------------------------
# 3. Convert counts -> probabilities
# -----------------------------------

probabilities = {}

for jurisdiction, judge_counts in counts.items():
    total = sum(judge_counts.values())

    probabilities[jurisdiction] = {
        judge: count / total
        for judge, count in judge_counts.items()
    }

# -----------------------------------
# 4. Function: Get probability distribution
# -----------------------------------

def get_probability_distribution(jurisdiction, sort=True):
    """
    Returns P(judge | jurisdiction)
    """
    if jurisdiction not in probabilities:
        raise ValueError(f"Jurisdiction '{jurisdiction}' not found.")

    distribution = probabilities[jurisdiction]

    if sort:
        return dict(
            sorted(distribution.items(), key=lambda x: x[1], reverse=True)
        )

    return distribution

# -----------------------------------
# 6. Example usage
# -----------------------------------

print("Available jurisdictions:")
print(list(probabilities.keys()))

example_jurisdiction = "Iowa, IA"

print(f"\nProbability distribution for: {example_jurisdiction}")
distribution = get_probability_distribution(example_jurisdiction)

for judge, prob in distribution.items():
    print(f"{judge}: {prob:.4f}")