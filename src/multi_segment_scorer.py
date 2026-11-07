
import json
import re
from pathlib import Path

from src.scorer import (
    score_hook,
    score_cliffhanger,
    score_emotional_intensity,
    score_context,
    score_promo_suitability,
)


WEIGHTS = {
    "hook": 25,
    "cliffhanger": 25,
    "emotional_intensity": 10,
    "context": 10,
    "promo_suitability": 5,
    "segment_continuity": 25,
}


_LOCATION_STOPWORDS = {
    "int", "ext", "interior", "exterior", "day", "night",
    "continuous", "same", "scene", "the", "and", "of", "in",
}
def _speaker_names(segment):
    return {
        re.sub(r"\s+", " ", str(event.get("speaker", "")).strip()).casefold()
        for event in segment.get("events", [])
        if event.get("event_type") == "DIALOGUE" and event.get("speaker")
    }


def _location_tokens(segment):
    heading = str(segment.get("scene_heading", ""))
    tokens = re.findall(r"[A-Za-z]{3,}", heading.casefold())
    return {token for token in tokens if token not in _LOCATION_STOPWORDS}


def _dialogue_events(segment):
    """Dialogue events overlapping a segment, in source-time order."""
    start = float(segment.get("start_time", 0))
    end = float(segment.get("end_time", 0))
    return sorted(
        [
            event for event in segment.get("events", [])
            if event.get("event_type") == "DIALOGUE"
            and float(event.get("start_time", 0)) < end
            and float(event.get("end_time", 0)) > start
        ],
        key=lambda event: (
            float(event.get("start_time", 0)),
            float(event.get("end_time", 0)),
        ),
    )


def _words(text):
    return re.findall(r"[a-z0-9']+", str(text).casefold())


def _is_context_dependent_opener(text):
    tokens = _words(text)
    response_starters = {
        "no", "yes", "but", "because", "then", "and", "what",
        "why", "how", "who", "he", "she", "they", "it", "that",
        "this", "never", "impossible",
    }
    return len(tokens) <= 2 or bool(tokens and tokens[0] in response_starters)


def opening_terms_answer(text):
    tokens = _words(text)
    return bool(tokens and tokens[0] in {"yes", "no", "because", "i", "we", "he", "she", "they"})


def _score_edit_junction(left, right):
    """Score whether the dialogue at a cut has a traceable handoff.

    This deliberately checks only evidence at the two edit boundaries. It
    cannot understand the full narrative, but it can flag clipped lines and
    context-dependent reaction openings that would otherwise be hidden by
    broad speaker/location overlap.
    """
    score = 60
    reasons = []
    left_end = float(left.get("end_time", 0))
    right_start = float(right.get("start_time", 0))

    left_dialogue = _dialogue_events(left)
    right_dialogue = _dialogue_events(right)
    closing = left_dialogue[-1] if left_dialogue else None
    opening = right_dialogue[0] if right_dialogue else None

    if closing:
        closing_text = str(closing.get("content", "")).strip()
        closing_speaker = str(closing.get("speaker") or "Unknown")
        clipped_at_end = float(closing.get("end_time", left_end)) > left_end + 0.01
        closing_tokens = _words(closing_text)

        if clipped_at_end:
            score -= 25
            reasons.append(
                f"Edit risk: previous segment cuts off {closing_speaker}'s "
                f"dialogue ({closing_text!r})"
            )
        elif len(closing_tokens) >= 4:
            score += 15
            reasons.append("Previous segment ends on a substantive dialogue beat")
        elif closing_text.endswith(("...", "…", "—", "-")):
            score -= 10
            reasons.append(
                f"Edit risk: previous segment ends on an unfinished line "
                f"({closing_speaker}: {closing_text!r})"
            )

    if opening:
        opening_text = str(opening.get("content", "")).strip()
        opening_speaker = str(opening.get("speaker") or "Unknown")
        opening_clipped = float(opening.get("start_time", right_start)) < right_start - 0.01
        opening_tokens = _words(opening_text)

        if opening_clipped:
            score -= 20
            reasons.append(
                f"Edit risk: next segment begins inside {opening_speaker}'s "
                f"dialogue ({opening_text!r})"
            )

        if _is_context_dependent_opener(opening_text):
            score -= 25
            reasons.append(
                f"Edit risk: next segment opens with a context-dependent "
                f"reaction ({opening_speaker}: {opening_text!r})"
            )
        elif len(opening_tokens) >= 4:
            score += 10
            reasons.append("Next segment opens with a substantive dialogue beat")

    if closing and opening:
        left_speaker = str(closing.get("speaker") or "").casefold()
        right_speaker = str(opening.get("speaker") or "").casefold()
        left_terms = set(_words(closing.get("content", "")))
        right_terms = set(_words(opening.get("content", "")))
        speaker_tokens = set(re.findall(r"[a-z0-9]+", left_speaker)) | set(re.findall(r"[a-z0-9]+", right_speaker))
        shared_terms = {
            term for term in left_terms & right_terms
            if len(term) >= 4
            and term not in {"that", "this", "what", "when", "your", "have"}
            and term not in speaker_tokens
        }

        if closing.get("content", "").strip().endswith("?") and opening_terms_answer(opening.get("content", "")):
            score += 15
            reasons.append("Dialogue handoff: the next line plausibly answers the previous question")
        elif left_speaker and left_speaker == right_speaker:
            score += 10
            reasons.append(f"Dialogue handoff: same speaker continues ({left_speaker})")
        elif shared_terms:
            score += 10
            reasons.append("Dialogue handoff shares a specific term: " + ", ".join(sorted(shared_terms)))
        elif left_speaker and right_speaker and left_speaker != right_speaker:
            score -= 15
            reasons.append(
                "Edit risk: adjacent dialogue switches speakers without a direct "
                "question/answer or shared-term handoff"
            )
    else:
        score -= 15
        reasons.append("Edit junction has insufficient dialogue evidence to assess the handoff")

    return max(0, min(100, score)), reasons


def score_segment_continuity(segments):
    """Combine structural source links with dialogue-level edit-junction quality.

    Structural order, recurring speakers and shared setting are not enough to
    prove story coherence. The final score therefore also evaluates how the
    first segment ends and how the next segment begins. This remains a
    screenplay-based heuristic, not a first-time-viewer comprehension test.
    """
    if len(segments) < 2:
        return 0, ["Continuity cannot be assessed with fewer than two segments"]

    structural_scores = []
    junction_scores = []
    all_reasons = []

    for left, right in zip(segments, segments[1:]):
        structural = 0
        left_order = (
            left.get("episode_number", 0),
            left.get("scene_number", 0),
            left.get("start_time", 0),
        )
        right_order = (
            right.get("episode_number", 0),
            right.get("scene_number", 0),
            right.get("start_time", 0),
        )
        if left_order <= right_order:
            structural += 25
            all_reasons.append("Structural link: segments follow screenplay chronology")

        shared_speakers = _speaker_names(left) & _speaker_names(right)
        if shared_speakers:
            structural += min(len(shared_speakers), 2) * 20
            all_reasons.append(
                "Structural link: Recurring speakers: " + ", ".join(sorted(shared_speakers))
            )

        shared_locations = _location_tokens(left) & _location_tokens(right)
        if shared_locations:
            structural += 35
            all_reasons.append(
                "Structural link: Shared setting cues: " + ", ".join(sorted(shared_locations))
            )

        junction, junction_reasons = _score_edit_junction(left, right)
        structural_scores.append(min(structural, 100))
        junction_scores.append(junction)
        all_reasons.extend(junction_reasons)

    structural_score = sum(structural_scores) / len(structural_scores)
    junction_score = sum(junction_scores) / len(junction_scores)
    final_score = round(0.40 * structural_score + 0.60 * junction_score)

    all_reasons.append(
        f"Continuity combines structural links ({structural_score:.0f}/100) "
        f"and edit-junction quality ({junction_score:.0f}/100); "
        "it does not prove first-time-viewer comprehension."
    )
    return final_score, all_reasons


def _interval_containment_overlap(segment_a, segment_b):
    """Return overlap divided by the shorter interval, for same source scene."""
    if (
        segment_a.get("episode_number") != segment_b.get("episode_number")
        or segment_a.get("scene_number") != segment_b.get("scene_number")
    ):
        return 0.0
    overlap = max(
        0.0,
        min(segment_a.get("end_time", 0), segment_b.get("end_time", 0))
        - max(segment_a.get("start_time", 0), segment_b.get("start_time", 0)),
    )
    shorter = min(
        max(0.0, segment_a.get("end_time", 0) - segment_a.get("start_time", 0)),
        max(0.0, segment_b.get("end_time", 0) - segment_b.get("start_time", 0)),
    )
    return overlap / shorter if shorter else 0.0


def are_near_duplicate_multi_candidates(candidate_a, candidate_b, threshold=0.70):
    """Detect multi promos made from substantially the same source intervals."""
    segments_a = candidate_a.get("segments", [])
    segments_b = candidate_b.get("segments", [])
    if not segments_a or len(segments_a) != len(segments_b):
        return False
    return all(
        _interval_containment_overlap(left, right) >= threshold
        for left, right in zip(segments_a, segments_b)
    )


def diversify_multi_rankings(records, max_items=100, threshold=0.70):
    """Keep the strongest representative and suppress near-duplicate edits."""
    ranked = sorted(records, key=lambda item: item.get("total_score", 0), reverse=True)
    selected = []
    for record in ranked:
        if any(
            are_near_duplicate_multi_candidates(record, prior, threshold)
            for prior in selected
        ):
            continue
        selected.append(record)
        if len(selected) >= max_items:
            break
    return selected


def combine_events(segments):
    """Combine events for overall promo analysis."""

    events = []

    for segment in segments:
        for event in segment["events"]:
            copied = event.copy()
            copied["source_episode"] = segment["episode_number"]
            copied["source_scene"] = segment["scene_number"]
            events.append(copied)

    return events


def make_segment_view(segment):
    """
    Create an independent scoring view for one segment.
    Its timestamps are never mixed with another episode.
    """

    return {
        "candidate_id": segment["candidate_id"],
        "episode_number": segment["episode_number"],
        "scene_number": segment["scene_number"],
        "start_time": segment["start_time"],
        "end_time": segment["end_time"],
        "duration": segment["duration"],
        "events": segment["events"],
    }


def score_multi_candidate(candidate):

    segments = candidate["segments"]

    first_segment = segments[0]
    last_segment = segments[-1]

    # Hook is evaluated only in the first segment.
    hook, hook_reasons = score_hook(
        make_segment_view(first_segment)
    )

    # Cliffhanger is evaluated only in the final segment.
    cliff, cliff_reasons = score_cliffhanger(
        make_segment_view(last_segment)
    )

    # These dimensions use events across all segments.
    combined = {
        "candidate_id": candidate["candidate_id"],
        "episode_number": first_segment["episode_number"],
        "scene_number": first_segment["scene_number"],
        "start_time": first_segment["start_time"],
        "end_time": last_segment["end_time"],
        "duration": candidate["duration"],
        "events": combine_events(segments),
    }

    emotion, emotion_reasons = (
        score_emotional_intensity(combined)
    )

    context, context_reasons = score_context(combined)

    suitability, suitability_reasons = (
        score_promo_suitability(combined)
    )

    continuity, continuity_reasons = score_segment_continuity(segments)

    scores = {
        "hook": hook,
        "cliffhanger": cliff,
        "emotional_intensity": emotion,
        "context": context,
        "promo_suitability": suitability,
        "segment_continuity": continuity,
    }

    total = sum(
        scores[key] * WEIGHTS[key] / 100
        for key in WEIGHTS
    )

    return {
        "candidate_id": candidate["candidate_id"],
        "candidate_type": "multi_segment",
        "duration": candidate["duration"],
        "segment_count": candidate["segment_count"],

        "segments": [
            {
                "segment_order": segment["segment_order"],
                "episode_number": segment["episode_number"],
                "scene_number": segment["scene_number"],
                "start_time": segment["start_time"],
                "end_time": segment["end_time"],
                "duration": segment["duration"],
            }
            for segment in segments
        ],

        "scores": scores,
        "total_score": round(total, 2),

        "reasons": {
            "hook": hook_reasons,
            "cliffhanger": cliff_reasons,
            "emotional_intensity": emotion_reasons,
            "context": context_reasons,
            "promo_suitability": suitability_reasons,
            "segment_continuity": continuity_reasons,
        },
    }


def main():

    root = Path(__file__).resolve().parents[1]
    output_dir = root / "outputs"

    input_path = output_dir / "multi_segment_candidates.json"
    output_path = (
        output_dir / "scored_multi_segment_candidates.json"
    )

    with input_path.open("r", encoding="utf-8") as file:
        candidates = json.load(file)

    scored = [
        score_multi_candidate(candidate)
        for candidate in candidates
    ]

    scored.sort(
        key=lambda item: item["total_score"],
        reverse=True
    )

    output_path.write_text(
        json.dumps(
            scored,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print("\n========== MULTI-SEGMENT SCORING V2 ==========")
    print("Candidates scored:", len(scored))

    print("\nTOP 10 MULTI-SEGMENT PROMOS")

    for rank, item in enumerate(scored[:10], 1):
        print(f"\n{rank}. {item['candidate_id']}")
        print("Duration:", item["duration"])
        print("Hook:", item["scores"]["hook"])
        print("Cliffhanger:", item["scores"]["cliffhanger"])
        print("Final score:", item["total_score"])

        for segment in item["segments"]:
            print(
                f"  EP{segment['episode_number']} "
                f"SC{segment['scene_number']}: "
                f"{segment['start_time']}–"
                f"{segment['end_time']}"
            )

    print("\nSaved to:", output_path)


if __name__ == "__main__":
    main()
