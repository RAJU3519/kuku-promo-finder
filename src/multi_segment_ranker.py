
import json
from pathlib import Path


def main():

    root = Path(__file__).resolve().parents[1]
    output_dir = root / "outputs"

    input_path = (
        output_dir / "scored_multi_segment_candidates.json"
    )

    output_path = (
        output_dir / "final_multi_segment_rankings.json"
    )

    with input_path.open("r", encoding="utf-8") as file:
        candidates = json.load(file)

    candidates.sort(
        key=lambda item: item["total_score"],
        reverse=True
    )

    for rank, candidate in enumerate(candidates, 1):
        candidate["rank"] = rank

    output_path.write_text(
        json.dumps(
            candidates,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print("\n========== MULTI-SEGMENT FINAL RANKING ==========")
    print("Total candidates:", len(candidates))

    for candidate in candidates[:10]:

        print(f"\nRank: {candidate['rank']}")
        print("Candidate:", candidate["candidate_id"])
        print("Duration:", candidate["duration"])
        print("Final score:", candidate["total_score"])

        for segment in candidate["segments"]:
            print(
                f"  Segment {segment['segment_order']}: "
                f"Episode {segment['episode_number']}, "
                f"Scene {segment['scene_number']}, "
                f"{segment['start_time']}–{segment['end_time']}"
            )

    print("\nSaved to:", output_path)


if __name__ == "__main__":
    main()
