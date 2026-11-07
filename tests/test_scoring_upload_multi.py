from src.multi_segment_generator import generate_multi_segment_candidates


def make_segment(candidate_id, episode, scene, start, end):
    return {
        "candidate_id": candidate_id,
        "episode_number": episode,
        "scene_number": scene,
        "scene_heading": f"Scene {scene}",
        "start_time": float(start),
        "end_time": float(end),
        "duration": float(end - start),
        "events": [],
    }


def test_multi_segment_candidates_are_chronologically_ordered():
    # Deliberately provide the segments out of chronological order.
    pool = [
        make_segment("EP2_SC1_0_20", 2, 1, 0, 20),
        make_segment("EP1_SC2_0_20", 1, 2, 0, 20),
        make_segment("EP1_SC1_0_20", 1, 1, 0, 20),
    ]

    generated = generate_multi_segment_candidates(pool)

    assert generated
    for candidate in generated:
        ordering = [
            (segment["episode_number"], segment["scene_number"], segment["start_time"])
            for segment in candidate["segments"]
        ]
        assert ordering == sorted(ordering)


def test_multi_segment_candidates_do_not_combine_same_scene():
    pool = [
        make_segment("EP1_SC1_0_20", 1, 1, 0, 20),
        make_segment("EP1_SC1_20_40", 1, 1, 20, 40),
    ]

    assert generate_multi_segment_candidates(pool) == []


def test_multi_segment_total_duration_stays_within_limit():
    pool = [
        make_segment("EP1_SC1_0_20", 1, 1, 0, 20),
        make_segment("EP2_SC1_0_30", 2, 1, 0, 30),
        make_segment("EP3_SC1_0_40", 3, 1, 0, 40),
    ]

    generated = generate_multi_segment_candidates(pool)

    assert generated
    assert all(20 <= item["duration"] <= 60 for item in generated)
