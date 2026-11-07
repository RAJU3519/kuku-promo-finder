from src.multi_segment_scorer import score_segment_continuity, WEIGHTS


def make_segment(episode, scene, start, heading, speakers, text):
    events = [
        {
            "event_type": "DIALOGUE",
            "speaker": speaker,
            "content": text,
            "start_time": start,
            "end_time": start + 1,
        }
        for speaker in speakers
    ]
    return {
        "episode_number": episode,
        "scene_number": scene,
        "start_time": start,
        "end_time": start + 20,
        "scene_heading": heading,
        "events": events,
    }


def test_continuity_rewards_shared_story_world():
    first = make_segment(
        42, 1, 10, "INT. CATHEDRAL - CONTINUOUS",
        ["ALEXANDER", "BLOND NOBLEMAN"],
        "Alexander confronts Rupert in the cathedral.",
    )
    second = make_segment(
        43, 1, 55, "INT. RUINED CATHEDRAL - CONTINUOUS",
        ["ALEXANDER", "BLOND NOBLEMAN", "CROWN OFFICIAL"],
        "Rupert, the Crown Official reads the names. Alexander accuses Rupert.",
    )
    coherent, reasons = score_segment_continuity([first, second])

    unrelated = make_segment(
        43, 4, 55, "EXT. FOREST - NIGHT",
        ["FOREST GUARD", "QUEEN"],
        "The Queen escapes through the forest.",
    )
    incoherent, _ = score_segment_continuity([first, unrelated])

    assert coherent > incoherent
    assert any("Recurring speakers" in reason for reason in reasons)
    assert any("Shared setting cues" in reason for reason in reasons)


def test_continuity_weights_sum_to_100():
    assert sum(WEIGHTS.values()) == 100


def test_single_segment_has_no_continuity_score():
    segment = make_segment(1, 1, 0, "INT. ROOM - DAY", ["ALICE"], "Alice enters.")
    score, reasons = score_segment_continuity([segment])
    assert score == 0
    assert reasons


def test_near_duplicate_candidates_are_detected():
    from src.multi_segment_scorer import are_near_duplicate_multi_candidates

    first = {
        "segments": [
            {"episode_number": 42, "scene_number": 1, "start_time": 10, "end_time": 30},
            {"episode_number": 43, "scene_number": 1, "start_time": 55, "end_time": 75},
        ]
    }
    slight_boundary_variant = {
        "segments": [
            {"episode_number": 42, "scene_number": 1, "start_time": 10, "end_time": 40},
            {"episode_number": 43, "scene_number": 1, "start_time": 55, "end_time": 75},
        ]
    }
    genuinely_different = {
        "segments": [
            {"episode_number": 42, "scene_number": 1, "start_time": 35, "end_time": 55},
            {"episode_number": 43, "scene_number": 1, "start_time": 55, "end_time": 75},
        ]
    }

    assert are_near_duplicate_multi_candidates(first, slight_boundary_variant)
    assert not are_near_duplicate_multi_candidates(first, genuinely_different)


def test_diversifier_keeps_highest_scoring_representative():
    from src.multi_segment_scorer import diversify_multi_rankings

    records = [
        {"candidate_id": "lower", "total_score": 60, "segments": [
            {"episode_number": 1, "scene_number": 1, "start_time": 10, "end_time": 30},
            {"episode_number": 2, "scene_number": 1, "start_time": 20, "end_time": 40},
        ]},
        {"candidate_id": "higher", "total_score": 70, "segments": [
            {"episode_number": 1, "scene_number": 1, "start_time": 10, "end_time": 40},
            {"episode_number": 2, "scene_number": 1, "start_time": 20, "end_time": 40},
        ]},
    ]
    result = diversify_multi_rankings(records)
    assert [item["candidate_id"] for item in result] == ["higher"]


def test_weak_edit_junction_prevents_perfect_continuity_score():
    """Shared cast/setting must not hide a cut-off line and dangling reaction."""
    first = make_segment(
        42, 1, 10, "INT. CATHEDRAL - CONTINUOUS",
        ["ALEXANDER", "BLOND NOBLEMAN", "CATHERINE"],
        "A confrontation continues in the cathedral.",
    )
    first["events"][-1].update({
        "speaker": "CATHERINE",
        "content": "Rupert...",
        "start_time": 29.0,
        "end_time": 30.3,
    })
    second = make_segment(
        43, 1, 55, "INT. RUINED CATHEDRAL - CONTINUOUS",
        ["ALEXANDER", "BLOND NOBLEMAN"],
        "No...",
    )
    second["events"][0].update({
        "speaker": "BLOND NOBLEMAN",
        "content": "No...",
        "start_time": 55.0,
        "end_time": 56.5,
    })

    score, reasons = score_segment_continuity([first, second])

    assert score < 60
    assert any("cuts off" in reason for reason in reasons)
    assert any("context-dependent reaction" in reason for reason in reasons)
    assert any("switches speakers without a direct" in reason for reason in reasons)


def test_coherent_dialogue_handoff_scores_above_dangling_reaction():
    first = make_segment(
        1, 1, 0, "INT. CATHEDRAL - DAY", ["ALICE", "BOB"],
        "Alice asks Bob whether he has the evidence?",
    )
    first["events"][-1].update({
        "speaker": "ALICE", "content": "Do you have the evidence?",
        "start_time": 18.0, "end_time": 19.0,
    })
    second = make_segment(
        2, 1, 0, "INT. CATHEDRAL - CONTINUOUS", ["ALICE", "BOB"],
        "Bob confirms that he has the evidence and shows it to her.",
    )
    second["events"][0].update({
        "speaker": "BOB", "content": "Yes, I have it right here.",
        "start_time": 0.0, "end_time": 2.0,
    })

    coherent, _ = score_segment_continuity([first, second])
    weak_first = make_segment(
        1, 1, 0, "INT. CATHEDRAL - DAY", ["ALICE", "BOB"], "Alice says Rupert..."
    )
    weak_first["events"][-1].update({
        "speaker": "ALICE", "content": "Rupert...",
        "start_time": 19.0, "end_time": 20.5,
    })
    weak_second = make_segment(
        2, 1, 0, "INT. CATHEDRAL - CONTINUOUS", ["ALICE", "BOB"], "No..."
    )
    weak_second["events"][0].update({
        "speaker": "BOB", "content": "No...", "start_time": 0.0, "end_time": 1.0,
    })
    weak, _ = score_segment_continuity([weak_first, weak_second])

    assert coherent > weak


def test_unlinked_speaker_change_is_flagged_at_edit_junction():
    first = make_segment(
        42, 1, 10, "INT. CATHEDRAL - CONTINUOUS", ["CATHERINE", "ALEXANDER"],
        "Catherine speaks to Alexander.",
    )
    first["events"][-1].update({
        "speaker": "CATHERINE", "content": "Rupert...",
        "start_time": 29.0, "end_time": 30.0,
    })
    second = make_segment(
        43, 1, 35, "INT. RUINED CATHEDRAL - CONTINUOUS", ["CROWN OFFICIAL", "ALEXANDER"],
        "The official reads a military intelligence report.",
    )
    second["events"][0].update({
        "speaker": "CROWN OFFICIAL",
        "content": "Military Intelligence Report, Iberian front, three years ago.",
        "start_time": 37.8, "end_time": 42.5,
    })

    score, reasons = score_segment_continuity([first, second])
    assert any("switches speakers without a direct" in reason for reason in reasons)
    assert score < 70


def test_repeated_character_name_alone_is_not_a_dialogue_handoff():
    first = make_segment(
        42, 1, 25, "INT. CATHEDRAL - CONTINUOUS", ["ALEXANDER'S MOTHER"],
        "Alexander...",
    )
    first["events"][0].update({
        "speaker": "ALEXANDER'S MOTHER", "content": "Alexander...",
        "start_time": 40.2, "end_time": 42.5,
    })
    second = make_segment(
        43, 1, 45, "INT. RUINED CATHEDRAL - CONTINUOUS", ["CROWN OFFICIAL"],
        "Duke Alexander Ashburn's unit was ambushed after his location was compromised.",
    )
    second["events"][0].update({
        "speaker": "CROWN OFFICIAL",
        "content": "Duke Alexander Ashburn's unit was ambushed after his location was compromised.",
        "start_time": 44.0, "end_time": 47.4,
    })

    _, reasons = score_segment_continuity([first, second])
    assert not any("shares a specific term" in reason for reason in reasons)
    assert any("switches speakers without a direct" in reason for reason in reasons)
