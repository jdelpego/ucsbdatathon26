def get_judge_pmf(judges: list[str]) -> dict[str, float]:
    """
    Compute a uniform probability mass function over a list of judges.

    Each judge is equally likely to be assigned, so every judge receives
    a probability of 1 / n where n is the number of judges.

    Args:
        judges: List of judge names.

    Returns:
        A dict mapping each judge name to its probability.
        Returns an empty dict if the list is empty.
    """
    if not judges:
        return {}
    prob = 1.0 / len(judges)
    return {name: prob for name in judges}


if __name__ == "__main__":
    sample = ["Judge A", "Judge B", "Judge C"]
    pmf = get_judge_pmf(sample)
    for name, p in pmf.items():
        print(f"  {name}: {p:.4f}")
