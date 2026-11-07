
import json
from pathlib import Path


OUTPUT_DIR = Path("outputs")

WEIGHTS = {
    "rule": 0.20,
    "hook": 0.25,
    "cliffhanger": 0.25,
    "segment": 0.30
}


def load_json(filename):
    path = OUTPUT_DIR / filename

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def index_by_candidate(records):
    return {
        record["candidate_id"]: record
        for record in records
    }


def category_to_score(category):
    scores = {
        "very_weak": 10,
        "weak": 30,
        "moderate": 50,
        "strong": 70,
        "exceptional": 90
    }

    if not category:
        return None

    return scores.get(category.lower())


def calculate_ranking():
    rule_records = load_json("scored_candidates.json")
    hook_records = load_json("hook_evaluations.json")
    cliff_records = load_json("cliffhanger_evaluations.json")
    segment_records = load_json("segment_evaluations.json")

    hooks = index_by_candidate(hook_records)
    cliffs = index_by_candidate(cliff_records)
    segments = index_by_candidate(segment_records)

    rankings = []

    for candidate in rule_records:

        candidate_id = candidate["candidate_id"]

        rule_score = candidate["total_score"]

        hook = hooks.get(candidate_id)
        cliff = cliffs.get(candidate_id)
        segment = segments.get(candidate_id)

        hook_score = None
        cliff_score = None
        segment_score = None

        if hook:
            hook_score = category_to_score(
                hook["hook_judgment"].get("hook_level")
            )

        if cliff:
            cliff_score = category_to_score(
                cliff["cliffhanger_judgment"].get(
                    "cliffhanger_level"
                )
            )

        if segment:
            segment_score = segment["overall_score"]

        available_scores = {
            "rule": rule_score,
            "hook": hook_score,
            "cliffhanger": cliff_score,
            "segment": segment_score
        }

        # Use only available components.
        # Renormalize weights so missing AI scores are not treated as zero.
        weighted_sum = 0
        available_weight = 0

        for component, score in available_scores.items():

            if score is not None:
                weight = WEIGHTS[component]
                weighted_sum += score * weight
                available_weight += weight

        final_score = (
            weighted_sum / available_weight
            if available_weight > 0
            else 0
        )

        rankings.append({
            "candidate_id": candidate_id,
            "episode_number": candidate["episode_number"],
            "scene_number": candidate["scene_number"],
            "start_time": candidate["start_time"],
            "end_time": candidate["end_time"],
            "duration": candidate["duration"],

            "rule_score": rule_score,
            "hook_score": hook_score,
            "cliffhanger_score": cliff_score,
            "segment_score": segment_score,

            "final_score": round(final_score, 2),

            "evaluation_coverage": round(
                available_weight / sum(WEIGHTS.values()) * 100,
                1
            ),

            "has_complete_ai_evaluation": all(
                score is not None
                for score in [
                    hook_score,
                    cliff_score,
                    segment_score
                ]
            )
        })

    rankings.sort(
        key=lambda item: item["final_score"],
        reverse=True
    )

    for rank, item in enumerate(rankings, start=1):
        item["rank"] = rank

    return rankings


def main():

    rankings = calculate_ranking()

    output_path = OUTPUT_DIR / "final_rankings.json"

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(
            rankings,
            file,
            indent=2,
            ensure_ascii=False
        )

    print("\n========== FINAL PROMO RANKINGS ==========")

    print("Total candidates:", len(rankings))
    print("Saved to:", output_path)

    print("\nTOP 10 CANDIDATES")

    for item in rankings[:10]:

        print("\nRank:", item["rank"])
        print("Candidate:", item["candidate_id"])
        print("Episode:", item["episode_number"])
        print(
            "Duration:",
            item["duration"],
            "seconds"
        )
        print("Rule score:", item["rule_score"])
        print("Hook score:", item["hook_score"])
        print("Cliffhanger score:", item["cliffhanger_score"])
        print("Segment score:", item["segment_score"])
        print("Final score:", item["final_score"])
        print(
            "Evaluation coverage:",
            item["evaluation_coverage"],
            "%"
        )


if __name__ == "__main__":
    main()
