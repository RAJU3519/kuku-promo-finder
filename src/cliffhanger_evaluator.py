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

OUTPUT_FILE = BASE_DIR / "outputs" / "cliffhanger_evaluations.json"

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
# EXTRACT ONLY THE FINAL FIVE SECONDS
# ============================================================

def get_ending_events(candidate, episodes):

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

    start = candidate["end_time"] - 5
    end = candidate["end_time"]

    ending_events = []

    for event in scene["events"]:

        if (
            event["start_time"] < end
            and event["end_time"] > start
        ):

            ending_events.append({
                "event_type": event["event_type"],
                "speaker": event.get("speaker"),
                "delivery": event.get("delivery"),
                "start_time": event["start_time"],
                "end_time": event["end_time"],
                "content": event["content"]
            })

    if not ending_events:
        raise ValueError("No events found in ending window")

    return ending_events


# ============================================================
# BUILD CLIFFHANGER-ONLY PROMPT
# ============================================================

def build_cliffhanger_prompt(candidate, ending_events):

    evidence = json.dumps(
        ending_events,
        indent=2,
        ensure_ascii=False
    )

    return f"""
You are an expert microdrama promo cliffhanger evaluator.

Your ONLY task is to evaluate the ending of a short-form
promotional video.

CANDIDATE:
ID: {candidate['candidate_id']}
Episode: {candidate['episode_number']}
Ending window: {candidate['end_time'] - 5} to
{candidate['end_time']} seconds.

IMPORTANT:
You are evaluating ONLY the final FIVE seconds.

ENDING SCREENPLAY EVIDENCE:

{evidence}


EVALUATION CRITERIA:

Determine whether these final five seconds make viewers
want to continue watching.

Consider:

1. An unresolved question.
2. A shocking revelation.
3. An unexpected decision.
4. A threat or confrontation.
5. An emotional reaction to new information.
6. An action whose outcome remains uncertain.

IMPORTANT DISTINCTIONS:

- A dramatic moment is not automatically a cliffhanger.
- A completed revelation can still create anticipation
  if it introduces a new unresolved conflict.
- An ordinary continuation of an action sequence may
  have a weak cliffhanger.
- A character's emotional reaction can strengthen an ending
  when it follows meaningful new information.


SCORING CATEGORIES:

very_weak:
The ending feels complete and creates almost no anticipation.

weak:
Limited anticipation; the story can end without much curiosity.

moderate:
Some unresolved tension or anticipation, but the impact
is limited.

strong:
A compelling ending that creates a clear reason to continue.

exceptional:
A highly powerful unresolved moment that strongly compels
viewers to watch the next part.


STRICT EVIDENCE RULES:

- Use only the supplied final-five-second events.
- Do not invent dialogue, characters or actions.
- Do not use events occurring before this window as proof.
- Do not use events occurring after the candidate's end time.
- Quote exact dialogue when relevant.
- Explain what remains unresolved.
- If nothing meaningful remains unresolved, say so.
- Be critical and consistent.


RETURN ONLY VALID JSON:

{{
    "cliffhanger_level": "strong",
    "cliffhanger_explanation": "Explanation based on actual evidence",
    "evidence": [
        "Exact dialogue or action from the ending"
    ],
    "unresolved_question": "What viewers are left wondering",
    "main_weakness": "Main weakness or None"
}}

Allowed cliffhanger_level values:
very_weak, weak, moderate, strong, exceptional.

Do not return numerical scores.
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
                    "You are a precise screenplay evaluator. "
                    "Evaluate only the supplied ending evidence. "
                    "Never invent events. Return valid JSON only."
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
# VALIDATE MODEL RESPONSE
# ============================================================

def validate_cliffhanger_judgment(judgment):

    allowed_levels = [
        "very_weak",
        "weak",
        "moderate",
        "strong",
        "exceptional"
    ]

    level = judgment.get("cliffhanger_level")

    if not isinstance(level, str):
        raise ValueError("Missing or invalid cliffhanger_level")

    level = (
        level.strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )

    if level not in allowed_levels:
        raise ValueError(f"Invalid cliffhanger category: {level}")

    judgment["cliffhanger_level"] = level

    string_fields = [
        "cliffhanger_explanation",
        "unresolved_question",
        "main_weakness"
    ]

    for field in string_fields:

        if not isinstance(judgment.get(field), str):
            judgment[field] = "Not provided by the model."

    if not isinstance(judgment.get("evidence"), list):
        judgment["evidence"] = []

    return judgment


# ============================================================
# DETERMINISTIC EVIDENCE VERIFICATION
# ============================================================

def normalize_text(text):
    """Normalize text for reliable comparison."""

    return re.sub(
        r"[^a-z0-9]+",
        " ",
        text.lower()
    ).strip()


def verify_evidence(evidence_items, ending_events):
    """
    Check whether each LLM evidence item appears
    in an actual event inside the ending window.
    """

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

        for event in ending_events:

            normalized_content = normalize_text(
                event["content"]
            )

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
                    "inside the ending window."
                )
            })

    return verification_results





# ============================================================
# EVALUATE ONE CANDIDATE
# ============================================================


# ============================================================
# TEST THREE CANDIDATES
# ============================================================

def evaluate_cliffhanger(candidate, episodes):

    ending_events = get_ending_events(candidate, episodes)

    prompt = build_cliffhanger_prompt(candidate, ending_events)

    judgment = evaluate_with_ollama(prompt)

    judgment = validate_cliffhanger_judgment(judgment)

    # Verify the model's evidence against actual screenplay events.
    evidence_verification = verify_evidence(
        judgment["evidence"],
        ending_events
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
        "ending_start": candidate["end_time"] - 5,
        "ending_end": candidate["end_time"],
        "model": MODEL_NAME,

        "ending_events": ending_events,

        "cliffhanger_judgment": judgment,

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

    print("\\n========== CLIFFHANGER-ONLY EVALUATION ==========")
    print("Model:", MODEL_NAME)

    for candidate_id in TEST_CANDIDATE_IDS:

        print("\\n----------------------------------------")
        print("Candidate:", candidate_id)

        try:
            if candidate_id not in candidate_lookup:
                raise ValueError("Candidate not found")

            result = evaluate_cliffhanger(
                candidate_lookup[candidate_id],
                episodes
            )
            results.append(result)

            judgment = result["cliffhanger_judgment"]

            print("Cliffhanger level:", judgment["cliffhanger_level"])
            print("Explanation:", judgment["cliffhanger_explanation"])
            print("Evidence:", judgment["evidence"])
            print("Unresolved question:", judgment["unresolved_question"])
            print("Weakness:", judgment["main_weakness"])

            print("\\nEvidence verification:")
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

    print("\\n========== CLIFFHANGER EVALUATION SUMMARY ==========")
    print("Successful:", len(results))
    print("Failed:", len(failures))
    print("Saved results:", OUTPUT_FILE)

    if failures:
        error_file = (
            BASE_DIR / "outputs" / "cliffhanger_evaluation_errors.json"
        )
        save_json(error_file, failures)
        print("Saved errors:", error_file)


# MAIN
# ============================================================

if __name__ == "__main__":
    evaluate_test_candidates()
