import json
from pathlib import Path


def get_top_candidate_ids(limit=10):

    path = Path("outputs/scored_candidates.json")

    with open(path, encoding="utf-8") as file:
        candidates = json.load(file)

    candidates.sort(
        key=lambda x: x["total_score"],
        reverse=True
    )

    return [
        candidate["candidate_id"]
        for candidate in candidates[:limit]
    ]


if __name__ == "__main__":

    candidate_ids = get_top_candidate_ids(10)

    print("Top 10 candidates for LLM evaluation:\n")

    for rank, candidate_id in enumerate(candidate_ids, 1):
        print(rank, candidate_id)