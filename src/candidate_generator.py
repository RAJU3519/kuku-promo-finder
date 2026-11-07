
from pathlib import Path
import json


# Candidate generation configuration
PROMO_DURATIONS = [20, 30, 40, 50, 60]
WINDOW_STRIDE = 5


def get_overlapping_events(events, window_start, window_end):
    """
    Return all events that overlap a candidate window.
    """

    overlapping = []

    for event in events:

        event_start = event["start_time"]
        event_end = event["end_time"]

        # Two time intervals overlap if:
        # event_start < window_end AND event_end > window_start

        if event_start < window_end and event_end > window_start:
            overlapping.append(event)

    return overlapping


def generate_scene_candidates(episode, scene):
    """
    Generate 20-60 second promo candidates within one scene.
    """

    candidates = []

    scene_start = scene["start_time"]
    scene_end = scene["end_time"]

    events = scene["events"]

    for duration in PROMO_DURATIONS:

        window_start = scene_start

        while window_start + duration <= scene_end:

            window_end = window_start + duration

            overlapping_events = get_overlapping_events(
                events,
                window_start,
                window_end
            )

            # Ignore windows without any screenplay events.
            if overlapping_events:

                candidate = {
                    "candidate_id": (
                        f"EP{episode['episode_number']}_"
                        f"SC{scene['scene_number']}_"
                        f"{int(window_start)}_"
                        f"{int(window_end)}"
                    ),

                    "episode_number": episode["episode_number"],

                    "scene_number": scene["scene_number"],

                    "scene_heading": scene["scene_heading"],

                    "start_time": round(window_start, 2),

                    "end_time": round(window_end, 2),

                    "duration": duration,

                    "events": overlapping_events,

                    "event_count": len(overlapping_events)
                }

                candidates.append(candidate)

            window_start += WINDOW_STRIDE

    return candidates


def generate_all_candidates(episodes):
    """
    Generate candidates across all episodes and scenes.
    """

    all_candidates = []

    for episode in episodes:

        for scene in episode["scenes"]:

            candidates = generate_scene_candidates(
                episode,
                scene
            )

            all_candidates.extend(candidates)

    return all_candidates


def main():

    project_root = Path(__file__).resolve().parents[1]

    input_path = (
        project_root / "outputs" / "parsed_episodes.json"
    )

    output_path = (
        project_root / "outputs" / "promo_candidates.json"
    )

    with input_path.open("r", encoding="utf-8") as file:
        episodes = json.load(file)

    candidates = generate_all_candidates(episodes)

    output_path.write_text(
        json.dumps(
            candidates,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print("\n========== PROMO CANDIDATE REPORT ==========")

    print(f"Episodes processed : {len(episodes)}")
    print(f"Candidates generated: {len(candidates)}")

    if candidates:

        print("\nSample candidates:")

        for candidate in candidates[:5]:

            print(
                f"\n{candidate['candidate_id']}"
                f"\nEpisode: {candidate['episode_number']}"
                f"\nScene: {candidate['scene_number']}"
                f"\nDuration: {candidate['duration']} seconds"
                f"\nTime: {candidate['start_time']} - "
                f"{candidate['end_time']}"
                f"\nEvents: {candidate['event_count']}"
            )

    print(f"\nJSON saved to: {output_path}")


if __name__ == "__main__":
    main()
