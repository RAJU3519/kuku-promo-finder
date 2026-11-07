
import json
from pathlib import Path

from src.multi_segment_scorer import score_multi_candidate


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"

# Each candidate contains two segments:
# Episode 42: selected ending
# Episode 43: continuation of the confrontation

CANDIDATE_VARIATIONS = [
    {
        "name": "A",
        "segments": [
            (42, 1, 64.0, 77.5),
            (43, 1, 0.0, 30.0),
        ],
    },
    {
        "name": "B",
        "segments": [
            (42, 1, 68.0, 77.5),
            (43, 1, 0.0, 30.0),
        ],
    },
    {
        "name": "C",
        "segments": [
            (42, 1, 72.5, 77.5),
            (43, 1, 0.0, 30.0),
        ],
    },
]


def load_episodes():
    path = OUTPUT_DIR / "parsed_episodes.json"

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def extract_segment(episodes, episode_number, scene_number,
                    start_time, end_time, segment_order):

    episode = next(
        item for item in episodes
        if item["episode_number"] == episode_number
    )

    scene = next(
        item for item in episode["scenes"]
        if item["scene_number"] == scene_number
    )

    # Keep events that overlap the selected time window.
    events = [
        event.copy()
        for event in scene["events"]
        if event["start_time"] < end_time
        and event["end_time"] > start_time
    ]

    events.sort(key=lambda event: event["start_time"])

    duration = round(end_time - start_time, 2)

    return {
        "segment_order": segment_order,
        "candidate_id": (
            f"EP{episode_number}_SC{scene_number}_"
            f"{start_time}_{end_time}"
        ),
        "episode_number": episode_number,
        "scene_number": scene_number,
        "scene_heading": scene.get("scene_heading", ""),
        "start_time": start_time,
        "end_time": end_time,
        "duration": duration,
        "events": events,
    }


def build_candidates(episodes):
    candidates = []

    for variation in CANDIDATE_VARIATIONS:
        segments = []

        for index, details in enumerate(
            variation["segments"], start=1
        ):
            episode, scene, start, end = details

            segment = extract_segment(
                episodes,
                episode,
                scene,
                start,
                end,
                index,
            )

            segments.append(segment)

        total_duration = round(
            sum(segment["duration"] for segment in segments),
            2,
        )

        candidate = {
            "candidate_id": (
                f"CUSTOM_{variation['name']}_"
                + "_".join(
                    segment["candidate_id"]
                    for segment in segments
                )
            ),
            "candidate_type": "multi_segment",
            "duration": total_duration,
            "segment_count": len(segments),
            "segments": segments,
        }

        candidates.append(candidate)

    return candidates


def main():
    episodes = load_episodes()

    candidates = build_candidates(episodes)

    candidates_path = (
        OUTPUT_DIR / "custom_multi_segment_candidates.json"
    )

    candidates_path.write_text(
        json.dumps(candidates, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    scored = [
        score_multi_candidate(candidate)
        for candidate in candidates
    ]

    scored.sort(
        key=lambda item: item["total_score"],
        reverse=True,
    )

    for rank, item in enumerate(scored, start=1):
        item["rank"] = rank

    scored_path = (
        OUTPUT_DIR / "custom_scored_multi_segment_candidates.json"
    )

    scored_path.write_text(
        json.dumps(scored, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n========== CUSTOM MULTI-SEGMENT EVALUATION ==========")
    print("Candidates generated:", len(candidates))

    for item in scored:
        print(f"\nRank {item['rank']}: {item['candidate_id']}")
        print("Duration:", item["duration"])
        print("Hook:", item["scores"]["hook"])
        print("Cliffhanger:", item["scores"]["cliffhanger"])
        print("Emotion:", item["scores"]["emotional_intensity"])
        print("Context:", item["scores"]["context"])
        print("Suitability:", item["scores"]["promo_suitability"])
        print("Final score:", item["total_score"])

    print("\nCandidate file:", candidates_path)
    print("Scored file:", scored_path)


if __name__ == "__main__":
    main()
