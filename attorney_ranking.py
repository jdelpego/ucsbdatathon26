import random
from collections import defaultdict


# Pool of 20 possible attorneys
ATTORNEYS = [f"attorney{i}" for i in range(1, 21)]


def get_judge_attorney_rankings(
    judges: list[str],
) -> dict[str, dict[str, int]]:
    """
    Mock function: for each judge, produce a top-10 attorney ranking.

    Uses each judge's name as a random seed so results are deterministic
    per judge but vary between judges.

    Args:
        judges: List of judge names.

    Returns:
        A dict mapping each judge name to a dict of
        {attorney_name: score (0–1000)} with 10 attorneys each.
    """
    rankings: dict[str, dict[str, int]] = {}
    for judge_name in judges:
        rng = random.Random(judge_name)
        selected = rng.sample(ATTORNEYS, 10)
        rankings[judge_name] = {
            attorney: rng.randint(0, 1000) for attorney in selected
        }
    return rankings


def get_weighted_attorney_rankings(
    judge_pmf: dict[str, float],
    judge_attorney_rankings: dict[str, dict[str, int]],
) -> dict[str, float]:
    """
    Compute an overall attorney ranking by weighting each judge's attorney
    scores by that judge's PMF probability and summing across all judges.

    For each judge:
        weighted_score(attorney) += score * P(judge)

    Then sum across judges to get the final ranking.

    Args:
        judge_pmf: A dict mapping judge names to their probability (from
                   get_judge_pmf, should sum to 1.0).
        judge_attorney_rankings: A dict mapping each judge name to a dict
                                 of {attorney_name: score}.

    Returns:
        A dict mapping attorney names to their weighted total score,
        sorted descending by score.
    """
    totals: dict[str, float] = defaultdict(float)

    for judge_name, prob in judge_pmf.items():
        ranking = judge_attorney_rankings.get(judge_name, {})
        for attorney, score in ranking.items():
            totals[attorney] += score * prob

    # Sort descending by weighted score
    sorted_totals = dict(
        sorted(totals.items(), key=lambda item: item[1], reverse=True)
    )
    return sorted_totals


if __name__ == "__main__":
    # Quick demo with fake judges
    judges = ["Judge A", "Judge B", "Judge C"]
    sample_pmf = {"Judge A": 0.5, "Judge B": 0.3, "Judge C": 0.2}
    rankings = get_judge_attorney_rankings(judges)

    print("Per-judge rankings:")
    for judge, ranking in rankings.items():
        print(f"\n  {judge}:")
        for att, score in sorted(ranking.items(), key=lambda x: x[1], reverse=True):
            print(f"    {att}: {score}")

    print("\nWeighted overall ranking:")
    overall = get_weighted_attorney_rankings(sample_pmf, rankings)
    for att, score in overall.items():
        print(f"  {att}: {score:.1f}")
