
import json
from pathlib import Path
from itertools import combinations

# Configuration
DURATION_LIMIT = 60
TOP_PER_DURATION = 8
ALLOWED_DURATIONS = [20, 30, 40]


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def select_segment_pool(candidates, scored_candidates):
    """
    Select promising short candidates separately by duration.
    """

    scores = {
        item["candidate_id"]: item["total_score"]
        for item in scored_candidates
    }

    selected = []

    for duration in ALLOWED_DURATIONS:

        matching = [
            candidate
            for candidate in candidates
            if candidate["duration"] == duration
        ]

        matching.sort(
            key=lambda item: scores.get(
                item["candidate_id"], 0
            ),
            reverse=True
        )

        selected.extend(matching[:TOP_PER_DURATION])

    return selected


def same_scene(segment_a, segment_b):
    return (
        segment_a["episode_number"]
        == segment_b["episode_number"]
        and segment_a["scene_number"]
        == segment_b["scene_number"]
    )


def build_multi_candidate(segments):
    """
    Preserve the original source and timestamps
    for every segment.
    """

    total_duration = sum(
        segment["duration"] for segment in segments
    )

    candidate_id = "MULTI_" + "_".join(
        segment["candidate_id"]
        for segment in segments
    )

    return {
        "candidate_id": candidate_id,
        "candidate_type": "multi_segment",
        "duration": total_duration,
        "segment_count": len(segments),
        "segments": [
            {
                "segment_order": index + 1,
                "candidate_id": segment["candidate_id"],
                "episode_number": segment["episode_number"],
                "scene_number": segment["scene_number"],
                "scene_heading": segment["scene_heading"],
                "start_time": segment["start_time"],
                "end_time": segment["end_time"],
                "duration": segment["duration"],
                "events": segment["events"]
            }
            for index, segment in enumerate(segments)
        ]
    }


def generate_multi_segment_candidates(pool):
    """
    Generate ordered combinations of 2 or 3 segments.
    Total promo duration must be 20–60 seconds.
    """

    generated = []
    seen = set()

    for segment_count in [2, 3]:

        for selected_segments in combinations(pool, segment_count):

            # Keep the segment order consistent with screenplay chronology:
            # earlier episodes first, then earlier scenes and timestamps.
            combination = tuple(sorted(
                selected_segments,
                key=lambda segment: (
                    segment["episode_number"],
                    segment["scene_number"],
                    segment["start_time"],
                    segment["end_time"],
                )
            ))

            total_duration = sum(
                segment["duration"]
                for segment in combination
            )

            if not 20 <= total_duration <= DURATION_LIMIT:
                continue

            # Avoid combining overlapping candidates
            # from the same original scene.
            if any(
                same_scene(combination[i], combination[j])
                for i in range(segment_count)
                for j in range(i + 1, segment_count)
            ):
                continue

            candidate = build_multi_candidate(combination)

            if candidate["candidate_id"] in seen:
                continue

            seen.add(candidate["candidate_id"])
            generated.append(candidate)

    return generated


def main():

    root = Path(__file__).resolve().parents[1]
    output_dir = root / "outputs"

    candidates = load_json(
        output_dir / "promo_candidates.json"
    )

    scored_candidates = load_json(
        output_dir / "scored_candidates.json"
    )

    pool = select_segment_pool(
        candidates,
        scored_candidates
    )

    multi_candidates = generate_multi_segment_candidates(
        pool
    )

    output_path = (
        output_dir / "multi_segment_candidates.json"
    )

    output_path.write_text(
        json.dumps(
            multi_candidates,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print("\n========== MULTI-SEGMENT GENERATION ==========")
    print("Original candidates:", len(candidates))
    print("Selected segment pool:", len(pool))
    print("Multi-segment candidates:", len(multi_candidates))

    if multi_candidates:
        print("\nSample multi-segment candidates:")

        for candidate in multi_candidates[:3]:
            print("\nCandidate:", candidate["candidate_id"])
            print("Total duration:", candidate["duration"])

            for segment in candidate["segments"]:
                print(
                    f"  Segment {segment['segment_order']}: "
                    f"Episode {segment['episode_number']}, "
                    f"Scene {segment['scene_number']}, "
                    f"{segment['start_time']}–"
                    f"{segment['end_time']} sec"
                )

    print("\nSaved to:", output_path)


if __name__ == "__main__":
    main()
