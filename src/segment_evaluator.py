import re
import json
import urllib.request
from pathlib import Path
from src.candidate_shortlist import get_top_candidate_ids


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

PARSED_FILE = BASE_DIR / "outputs" / "parsed_episodes.json"
SCORED_FILE = BASE_DIR / "outputs" / "scored_candidates.json"

OUTPUT_FILE = BASE_DIR / "outputs" / "segment_evaluations.json"

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "qwen2.5:3b"
TEST_CANDIDATE_IDS = [
    "EP43_SC1_10_60",
    "EP16_SC1_5_55",
    "EP3_SC3_74_134",
    "EP43_SC1_10_70",
    "EP42_SC1_10_30",
    "EP3_SC3_74_104",
    "EP42_SC1_25_45",
    "EP43_SC1_15_75",
    "EP42_SC1_25_75",
    "EP16_SC1_25_55",
    "EP43_SC1_0_60"
]

WEIGHTS = {
    "narrative_coherence": 20,
    "engagement": 25,
    "emotional_intensity": 20,
    "first_time_clarity": 15,
    "promo_suitability": 20
}


# ============================================================
# JSON UTILITIES
# ============================================================

def load_json(file_path):

    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json(file_path, data):

    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)


# ============================================================
# EXTRACT THE COMPLETE CANDIDATE SEGMENT
# ============================================================

def get_segment_events(candidate, episodes):

    episode = next(
        (
            item for item in episodes
            if item["episode_number"] == candidate["episode_number"]
        ),
        None
    )

    if episode is None:
        raise ValueError("Episode not found")

    scene = next(
        (
            item for item in episode["scenes"]
            if item["scene_number"] == candidate["scene_number"]
        ),
        None
    )

    if scene is None:
        raise ValueError("Scene not found")

    start = candidate["start_time"]
    end = candidate["end_time"]

    if end <= start:
        raise ValueError(
            "Candidate end_time must be greater than start_time"
        )

    segment_events = []

    for event in scene["events"]:

        event_start = event["start_time"]
        event_end = event["end_time"]

        overlap_start = max(event_start, start)
        overlap_end = min(event_end, end)

        # Ignore events that do not overlap the candidate.
        if overlap_start >= overlap_end:
            continue

        starts_before = event_start < start
        ends_after = event_end > end

        fully_contained = not starts_before and not ends_after

        if starts_before and ends_after:
            boundary_status = "crosses_both_boundaries"

        elif starts_before:
            boundary_status = "starts_before_candidate"

        elif ends_after:
            boundary_status = "ends_after_candidate"

        else:
            boundary_status = "fully_inside_candidate"

        if fully_contained:
            content = event["content"]

        else:
            content = (
                f"[PARTIAL {event['event_type']} EVENT. "
                f"The event {boundary_status.replace('_', ' ')}. "
                "Full event text withheld because its exact "
                "in-candidate portion is unavailable.]"
            )

        segment_events.append({
            "event_type": event["event_type"],
            "speaker": event.get("speaker"),
            "delivery": event.get("delivery"),

            # Original screenplay timestamps
            "original_start_time": event_start,
            "original_end_time": event_end,

            # Actual overlap with the candidate
            "start_time": overlap_start,
            "end_time": overlap_end,

            "is_fully_contained": fully_contained,
            "boundary_status": boundary_status,

            "content": content
        })

    if not segment_events:
        raise ValueError("No events found inside candidate segment")

    return segment_events


# ============================================================
# BUILD FULL-SEGMENT EVALUATION PROMPT
# ============================================================
def build_segment_prompt(candidate, segment_events):

    evidence = json.dumps(
        segment_events,
        indent=2,
        ensure_ascii=False
    )

    return f"""
You are a strict, independent editor selecting short-form microdrama promos.

Evaluate the COMPLETE candidate segment using only the screenplay events
provided below. Your job is to distinguish average, good and excellent
material. Do not reward a segment merely because it contains many events,
dramatic words or action descriptions.

CANDIDATE:
ID: {candidate['candidate_id']}
Episode: {candidate['episode_number']}
Scene: {candidate['scene_number']}
Time range: {candidate['start_time']} to {candidate['end_time']} seconds.
Duration: {candidate['end_time'] - candidate['start_time']} seconds.

BOUNDARY AND TIMESTAMP RULES:

1. The candidate time range is the exact promo boundary.

2. Each event's start_time and end_time represent only its
   overlap with the candidate window.

3. original_start_time and original_end_time represent the
   event's original screenplay timestamps.

4. An event marked fully_inside_candidate contains its
   complete available screenplay description inside the promo.

5. An event marked starts_before_candidate, ends_after_candidate,
   or crosses_both_boundaries is only partially inside the promo.

6. For partial events, the content field deliberately withholds
   the full dialogue or action description because exact
   in-window content is unavailable.


7. NEVER infer the missing portion of a partial event.
   NEVER assume that a complete dialogue, revelation, action,
   or resolution happens inside the candidate.

8. Evaluate only what is established within the candidate window.
   An incomplete final event may create anticipation, but do not
   assume its eventual outcome.

9. STRICT EVIDENCE RULES:

- Use ONLY events marked fully_inside_candidate as evidence
  for dialogue, actions, character behavior, and story claims.

- NEVER quote, paraphrase, or invent the missing content
  of a partial event.

- NEVER include a partial event in the evidence array.

- Partial events may ONLY be used to identify a boundary cut
  and classify the ending.

- Every evidence item must be traceable to one specific
  fully contained screenplay event.

- Prefer exact dialogue or concise descriptions copied
  from fully contained camera/action events.

- If there is not enough complete evidence to support a claim,
  explicitly say that the evidence is insufficient.

10. Do not penalize a candidate merely because an event crosses
    a boundary. Judge the actual evidence available inside the
    candidate window.

11. MANDATORY ENDING ASSESSMENT:

Inspect the final event or events in the candidate.

Classify the ending into exactly one of these categories:

- COMPLETE_BEAT: A meaningful action or dialogue finishes.
- UNRESOLVED_CLIFFHANGER: A question, threat, revelation or conflict
  remains unresolved at the end.
- CUTS_MID_EVENT: An event continues beyond the candidate's end time.
- UNCLEAR: The supplied evidence is insufficient to determine the ending.

12. BOUNDARY-AWARE ENDING:

If an event has boundary_status = ends_after_candidate
or crosses_both_boundaries, explicitly consider this in ending_effect.

Do not describe an incomplete event as a completed revelation.

Do not invent missing dialogue or action.

A cut during an event is not automatically bad. It may create
a compelling cliffhanger, but explain what is actually established
and what remains unknown.

13. IDENTIFY MATERIAL WEAKNESSES:

Do not automatically return None for weakest_portion or main_weakness.

Identify an actual limitation when supported by the screenplay.
A cut-off event may be a limitation or an effective cliffhanger.
Explain which one applies and why.

SCREENPLAY EVENTS OVERLAPPING THE CANDIDATE WINDOW:

{evidence}

SCORING DIMENSIONS AND ANCHORS

For every dimension, select one category using these anchors.
A strong rating requires clear evidence.

1. narrative_coherence

- very_weak: Events are confusing or disconnected.
- weak: Major context or transitions are missing.
- moderate: Basic situation can be followed, but important links are unclear.
- strong: Events form a clear progression with understandable cause and effect.
- exceptional: A particularly clean and satisfying mini-story with strong progression.

2. engagement

- very_weak: Little happens to hold attention.
- weak: Long stretches are repetitive, static or low-interest.
- moderate: Some interesting moments, but momentum drops or is uneven.
- strong: Interest is maintained through meaningful changes, reveals or conflict.
- exceptional: Nearly every beat advances curiosity; momentum builds noticeably.

3. emotional_intensity

- very_weak: Little discernible emotional or dramatic tension.
- weak: Emotion or conflict is faint and unsupported.
- moderate: Some tension or emotion, but limited escalation or stakes.
- strong: Clear conflict, threat, revelation or emotional reaction with meaningful stakes.
- exceptional: Multiple powerful emotional beats or major escalation with high stakes.

4. first_time_clarity

- very_weak: A new viewer cannot understand the basic situation.
- weak: Essential roles or situation depend too much on unknown backstory.
- moderate: Immediate situation is partly understandable, but key context is missing.
- strong: A new viewer can follow the immediate conflict and understand what matters.
- exceptional: Situation and stakes are immediately clear without prior knowledge.

5. promo_suitability

- very_weak: Does not work as a promotional clip.
- weak: Lacks a compelling reason for a new viewer to continue.
- moderate: Has useful moments but an incomplete or uneven promotional arc.
- strong: Has a compelling entry, clear progression and an effective reason to watch more.
- exceptional: A highly effective standalone teaser with a powerful hook,
  escalating progression and memorable unresolved payoff.

IMPORTANT EVALUATION RULES

1. SCORE EACH DIMENSION INDEPENDENTLY.

Missing backstory should primarily affect first_time_clarity.
Do not automatically reduce narrative_coherence, engagement,
emotional_intensity or promo_suitability because a viewer does
not know the complete series history.

2. DISTINGUISH LOCAL CLARITY FROM BACKSTORY.

A viewer does not need to know the entire history of a character.
They only need enough information to understand the immediate
situation and become curious.

3. RECOGNIZE EMOTIONAL IMPACT.

A threat, betrayal, physical confrontation, shocking revelation,
romantic gesture or emotional reaction can be powerful even
without complete character background.

4. EVALUATE ENGAGEMENT THROUGHOUT THE SEGMENT.

Identify actual changes in the situation.
Do not call a segment repetitive without identifying the
specific events that repeat or fail to advance the story.

5. EVALUATE PROMO SUITABILITY SEPARATELY.

Ask whether the segment makes a new viewer curious.
A promotional clip does not need to resolve the story.
An unresolved question can be a strength.

6. IDENTIFY A REAL WEAKNESS.

Inspect the supplied events and identify the weakest or least
engaging portion, with its approximate timestamp.
If no significant weakness exists, explain why rather than
automatically writing None.

7. AVOID SCORE COPYING.

Do not assign moderate to all dimensions by default.
Do not assign the same rating to multiple dimensions merely
because they are related.
Use the complete category range when justified by evidence.

8. BE EVIDENCE-BASED.

Do not invent dialogue, actions, motives or relationships.
Do not claim actual acting, editing, music or visual quality.
Only screenplay text is available.

REQUIRED REASONING BEFORE SCORING

First identify:

1. The main situation or conflict.
2. The most important change, reveal or escalation.
3. The weakest or least engaging portion, if evident.
4. What a first-time viewer may not understand.
5. Whether the ending creates anticipation or completes the moment.

Then score all five dimensions.

RETURN ONLY VALID JSON:


{{
    "segment_summary": "One or two sentences describing the segment",
    "main_conflict": "The central conflict",
    "progression": "The most important escalation or revelation",
    "weakest_portion": "Specific weakness with timestamp, or justified None",
"first_time_viewer_gap": "Missing context, or None",
"ending_type": "CUTS_MID_EVENT",
"ending_effect": "Explain the ending using boundary timestamps and available evidence.",

    "narrative_coherence": {{
        "score": 7,
        "explanation": "Specific evidence-based explanation"
    }},

    "engagement": {{
        "score": 8,
        "explanation": "Specific evidence-based explanation"
    }},

    "emotional_intensity": {{
        "score": 6,
        "explanation": "Specific evidence-based explanation"
    }},

    "first_time_clarity": {{
        "score": 7,
        "explanation": "Specific evidence-based explanation"
    }},

    "promo_suitability": {{
        "score": 8,
        "explanation": "Specific evidence-based explanation"
    }},

    "overall_explanation": "Balanced assessment",
    "evidence": [
        "Exact dialogue or screenplay event"
    ],
    "main_strength": "Most important strength",
    "main_weakness": "Most important weakness"
}}

SCORING SCALE:

0-2: Very poor. Almost no supporting evidence.
3-4: Weak. Significant problems limit effectiveness.
5-6: Moderate. Some strengths, but meaningful limitations.
7-8: Strong. Clear evidence of effectiveness.
9-10: Exceptional. Outstanding evidence and execution.

IMPORTANT:

- Give an integer score from 0 to 10 for every dimension.
- Score every dimension independently.
- Do not default to 5.
- Do not copy the same score across dimensions without independent justification.
- Missing backstory should primarily affect first_time_clarity.
- Do not reward action or dramatic language without considering its actual impact.
- Explain why the score is not one point higher.
- Use only the supplied screenplay events.
- Return valid JSON only.
"""


# ============================================================
# OLLAMA REQUEST
# ============================================================

def evaluate_with_ollama(prompt):

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a precise and critical screenplay evaluator. "
                    "Evaluate only the supplied complete segment. "
                    "Do not invent evidence. Return valid JSON only."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0
        }
    }

    request = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    with urllib.request.urlopen(request, timeout=300) as response:

        result = json.loads(
            response.read().decode("utf-8")
        )

    return json.loads(result["message"]["content"])


# ============================================================
# VALIDATE MODEL RESPONSE AND CALCULATE SCORE
# ============================================================

def validate_segment_judgment(judgment):

    dimensions = list(WEIGHTS.keys())

    if not isinstance(judgment, dict):
        raise ValueError("Model response must be a JSON object")

    dimension_scores = {}
    weighted_total = 0

    for dimension in dimensions:

        result = judgment.get(dimension)

        if not isinstance(result, dict):
            raise ValueError(
                f"Missing or invalid dimension: {dimension}"
            )

        score = result.get("score")

        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ValueError(
                f"Invalid numerical score for {dimension}"
            )

        if not 0 <= score <= 10:
            raise ValueError(
                f"Score outside 0-10 range for {dimension}"
            )

        if not isinstance(result.get("explanation"), str):
            raise ValueError(
                f"Missing explanation for {dimension}"
            )

        score = round(float(score), 2)
        result["score"] = score

        weight = WEIGHTS[dimension]

        weighted_score = score * 10 * weight / 100

        dimension_scores[dimension] = {
            "score": score,
            "score_out_of_100": score * 10,
            "weight": weight,
            "weighted_score": round(weighted_score, 2)
        }

        weighted_total += weighted_score

    for field in [
        "segment_summary",
        "main_conflict",
        "progression",
        "weakest_portion",
        "first_time_viewer_gap",
        "ending_effect",
        "overall_explanation",
        "main_strength",
        "main_weakness"
    ]:

        if not isinstance(judgment.get(field), str):
            raise ValueError(
                f"Missing or invalid field: {field}"
            )


    allowed_ending_types = {
        "COMPLETE_BEAT",
        "UNRESOLVED_CLIFFHANGER",
        "CUTS_MID_EVENT",
        "UNCLEAR"
    }

    if judgment.get("ending_type") not in allowed_ending_types:
        raise ValueError(
            f"Invalid ending_type: {judgment.get('ending_type')}"
        )
    allowed_ending_types = {
        "COMPLETE_BEAT",
        "UNRESOLVED_CLIFFHANGER",
        "CUTS_MID_EVENT",
        "UNCLEAR"
    }

    if judgment.get("ending_type") not in allowed_ending_types:
        raise ValueError(
            f"Invalid ending_type: {judgment.get('ending_type')}"
        )

    if not isinstance(judgment.get("evidence"), list):
        raise ValueError("Evidence must be a list")

    return (
        judgment,
        dimension_scores,
        round(weighted_total, 2)
    )


# ============================================================
# DETERMINISTIC EVIDENCE VERIFICATION
# ============================================================

def normalize_text(text):

    return re.sub(
        r"[^a-z0-9]+",
        " ",
        str(text).lower()
    ).strip()


def verify_evidence(evidence_items, segment_events):

    verification_results = []

    for item in evidence_items:

        if not isinstance(item, str) or not item.strip():

            verification_results.append({
                "evidence": str(item),
                "verified": False,
                "reason": "Evidence is empty or invalid."
            })

            continue

        normalized_evidence = normalize_text(item)
        matched_event = None

        for event in segment_events:

            normalized_content = normalize_text(event["content"])

            if (
                normalized_evidence in normalized_content
                or normalized_content in normalized_evidence
            ):

                matched_event = event
                break

        if matched_event:

            verification_results.append({
                "evidence": item,
                "verified": True,
                "matched_event_type": matched_event["event_type"],
                "matched_start_time": matched_event["start_time"],
                "matched_end_time": matched_event["end_time"],
                "matched_content": matched_event["content"]
            })

        else:

            verification_results.append({
                "evidence": item,
                "verified": False,
                "reason": (
                    "No matching screenplay event found "
                    "inside the candidate segment."
                )
            })

    return verification_results


# ============================================================
# EVALUATE ONE CANDIDATE
# ============================================================

def evaluate_segment(candidate, episodes):

    segment_events = get_segment_events(candidate, episodes)

    prompt = build_segment_prompt(candidate, segment_events)

    raw_judgment = evaluate_with_ollama(prompt)

    judgment, dimension_scores, overall_score = (
        validate_segment_judgment(raw_judgment)
    )

    evidence_verification = verify_evidence(
        judgment["evidence"],
        segment_events
    )

    verified_count = sum(
        1 for item in evidence_verification
        if item["verified"]
    )

    total_evidence = len(evidence_verification)

    return {
        "candidate_id": candidate["candidate_id"],
        "episode_number": candidate["episode_number"],
        "scene_number": candidate["scene_number"],
        "start_time": candidate["start_time"],
        "end_time": candidate["end_time"],
        "duration": candidate["end_time"] - candidate["start_time"],
        "model": MODEL_NAME,
        "segment_events": segment_events,
        "segment_judgment": judgment,
        "dimension_scores": dimension_scores,
        "overall_score": overall_score,
        "recommended": overall_score >= 65,
        "evidence_verification": evidence_verification,
        "evidence_summary": {
            "verified": verified_count,
            "unverified": total_evidence - verified_count,
            "total": total_evidence
        }
    }


# ============================================================
# TEST THREE CANDIDATES
# ============================================================

def evaluate_test_candidates():

    candidates = load_json(SCORED_FILE)
    episodes = load_json(PARSED_FILE)

    candidate_lookup = {
        item["candidate_id"]: item
        for item in candidates
    }

    results = []
    failures = []

    print("\n========== FULL-SEGMENT EVALUATION ==========")
    print("Model:", MODEL_NAME)

    for candidate_id in TEST_CANDIDATE_IDS:

        print("\n----------------------------------------")
        print("Candidate:", candidate_id)

        try:

            if candidate_id not in candidate_lookup:
                raise ValueError("Candidate not found")

            candidate = candidate_lookup[candidate_id]

            result = evaluate_segment(candidate, episodes)

            results.append(result)

            print("Duration:", result["duration"], "seconds")
            print("Dimension scores:")

            for dimension, details in result["dimension_scores"].items():
                print(
                    f"  {dimension}: {details['score']}/10 "
                    f"({details['score_out_of_100']}/100), "
                    f"weight={details['weight']}%"
                )

            print("Overall score:", result["overall_score"])
            print("Recommended:", result["recommended"])
            print("Overall explanation:",
                  result["segment_judgment"]["overall_explanation"])

            print("Main strength:",
                  result["segment_judgment"]["main_strength"])

            print("Main weakness:",
                  result["segment_judgment"]["main_weakness"])

            print("Ending type:",
                  result["segment_judgment"]["ending_type"])

            print("Ending effect:",
                  result["segment_judgment"]["ending_effect"])

            print("\nEvidence verification:")          
        

            for item in result["evidence_verification"]:

                status = "VERIFIED" if item["verified"] else "UNVERIFIED"
                print(f"[{status}] {item['evidence']}")

            print("Evidence summary:", result["evidence_summary"])

        except Exception as error:

            print("Evaluation failed:", error)

            failures.append({
                "candidate_id": candidate_id,
                "error": str(error)
            })

    save_json(OUTPUT_FILE, results)

    failure_file = OUTPUT_FILE.with_name(
        "segment_evaluation_failures.json"
    )
    save_json(failure_file, failures)

    print("\n========== SEGMENT EVALUATION SUMMARY ==========")
    print("Successful:", len(results))
    print("Failed:", len(failures))
    print("Saved results:", OUTPUT_FILE)
    print("Saved failures:", failure_file)


if __name__ == "__main__":
    evaluate_test_candidates()
