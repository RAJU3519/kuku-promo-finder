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

OUTPUT_FILE = BASE_DIR / "outputs" / "hook_evaluations.json"

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "qwen2.5:3b"

from src.candidate_shortlist import get_top_candidate_ids

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
        json.dump(
            data,
            file,
            indent=4,
            ensure_ascii=False
        )


# ============================================================
# EXTRACT ONLY THE FIRST THREE SECONDS
# ============================================================

def get_opening_events(candidate, episodes):

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
    end = start + 3

    opening_events = []

    for event in scene["events"]:

        # Include only events overlapping the first 3 seconds.
        if (
            event["start_time"] < end
            and event["end_time"] > start
        ):

            opening_events.append({
                "event_type": event["event_type"],
                "speaker": event.get("speaker"),
                "delivery": event.get("delivery"),
                "start_time": event["start_time"],
                "end_time": event["end_time"],
                "content": event["content"]
            })

    if not opening_events:
        raise ValueError("No events found in opening window")

    return opening_events


# ============================================================
# BUILD HOOK-ONLY PROMPT
# ============================================================

def build_hook_prompt(candidate, opening_events):

    evidence = json.dumps(
        opening_events,
        indent=2,
        ensure_ascii=False
    )

    return f"""
You are an expert microdrama promo hook evaluator.

Your ONLY task is to evaluate the opening hook of a
short-form promotional video.

CANDIDATE:
ID: {candidate['candidate_id']}
Episode: {candidate['episode_number']}
Opening window: {candidate['start_time']} to
{candidate['start_time'] + 3} seconds.

IMPORTANT:
You are evaluating ONLY the first THREE seconds.

You must not assume what happens later in the story.

You must not use events outside the supplied opening window.

OPENING SCREENPLAY EVIDENCE:

{evidence}


EVALUATION CRITERIA:

Evaluate how effectively these first three seconds
capture a viewer's attention.

Consider:

1. Immediate conflict or confrontation.
2. Threats, shocking statements or accusations.
3. Mystery or unanswered questions.
4. Strong emotional expressions.
5. Unusual or visually interesting actions.
6. Dialogue that makes viewers want to know what happens next.

Physical violence is NOT necessary for a strong hook.

A confrontation, threat, mystery or emotionally intense
statement can independently create a strong hook.


SCORING CATEGORIES:

very_weak:
Almost no attention-grabbing element.

weak:
Limited interest or an ordinary opening.

moderate:
Some attention-grabbing evidence, but limited impact.

strong:
Clearly compelling opening that creates immediate interest.

exceptional:
Extremely powerful opening that strongly encourages
viewers to continue watching.


STRICT EVIDENCE RULES:

- Use only the supplied opening events.
- Do not invent characters, dialogue or actions.
- Do not describe events that occur later.
- Do not confuse a camera description with an action.
- Do not assume an emotion unless supported by the text.
- Quote exact dialogue when relevant.
- Explain why the opening attracts attention.
- Be honest if the evidence is insufficient.


RETURN ONLY VALID JSON:

{{
    "hook_level": "strong",
    "hook_explanation": "Explanation based on actual evidence",
    "evidence": [
        "Exact dialogue or action from the opening"
    ],
    "main_strength": "Main strength of the opening",
    "main_weakness": "Main weakness or None"
}}

Allowed hook_level values:
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
                    "Evaluate only the supplied opening evidence. "
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

def validate_hook_judgment(judgment):

    allowed_levels = [
        "very_weak",
        "weak",
        "moderate",
        "strong",
        "exceptional"
    ]

    level = judgment.get("hook_level")

    if not isinstance(level, str):
        raise ValueError("Missing or invalid hook_level")

    level = (
        level.strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )

    if level not in allowed_levels:
        raise ValueError(f"Invalid hook category: {level}")

    judgment["hook_level"] = level

    explanation = judgment.get("hook_explanation")

    if not isinstance(explanation, str) or not explanation.strip():
        judgment["hook_explanation"] = (
            "Explanation not provided by the model."
        )

    evidence = judgment.get("evidence")

    if not isinstance(evidence, list):
        judgment["evidence"] = []

    for field in ["main_strength", "main_weakness"]:

        if not isinstance(judgment.get(field), str):
            judgment[field] = "Not provided by the model."

    return judgment

# ============================================================
# EVIDENCE VERIFICATION
# ============================================================

def normalize_text(text):

    return re.sub(
        r"[^a-z0-9]+",
        " ",
        text.lower()
    ).strip()


def verify_hook_evidence(evidence_items, opening_events):

    verification_results = []

    for item in evidence_items:

        if not isinstance(item, str) or not item.strip():

            verification_results.append({
                "evidence": str(item),
                "verified": False
            })
            continue

        normalized_evidence = normalize_text(item)

        matched_event = None

        for event in opening_events:

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
                "matched_content": matched_event["content"],
                "matched_start_time": matched_event["start_time"],
                "matched_end_time": matched_event["end_time"]
            })

        else:

            verification_results.append({
                "evidence": item,
                "verified": False,
                "reason": "No matching opening event found."
            })

    return verification_results
# ============================================================
# EVALUATE ONE CANDIDATE
# ============================================================

def evaluate_hook(candidate, episodes):

    opening_events = get_opening_events(candidate, episodes)

    prompt = build_hook_prompt(candidate, opening_events)

    judgment = evaluate_with_ollama(prompt)

    judgment = validate_hook_judgment(judgment)

    evidence_verification = verify_hook_evidence(
        judgment["evidence"],
        opening_events
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
        "opening_start": candidate["start_time"],
        "opening_end": candidate["start_time"] + 3,
        "model": MODEL_NAME,
        "opening_events": opening_events,
        "hook_judgment": judgment,
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

    print("\n========== HOOK-ONLY EVALUATION ==========")
    print("Model:", MODEL_NAME)

    for candidate_id in TEST_CANDIDATE_IDS:

        print("\n----------------------------------------")
        print("Candidate:", candidate_id)

        try:

            if candidate_id not in candidate_lookup:
                raise ValueError("Candidate not found")

            candidate = candidate_lookup[candidate_id]

            result = evaluate_hook(candidate, episodes)

            results.append(result)

            judgment = result["hook_judgment"]

            print("Hook level:", judgment["hook_level"])
            print("Explanation:", judgment["hook_explanation"])
            print("Evidence:", judgment["evidence"])
            print("Strength:", judgment["main_strength"])
            print("Weakness:", judgment["main_weakness"])

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

    print("\n========== HOOK EVALUATION SUMMARY ==========")
    print("Successful:", len(results))
    print("Failed:", len(failures))
    print("Saved results:", OUTPUT_FILE)

    if failures:

        error_file = (
            BASE_DIR / "outputs" / "hook_evaluation_errors.json"
        )

        save_json(error_file, failures)

        print("Saved errors:", error_file)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    evaluate_test_candidates()
