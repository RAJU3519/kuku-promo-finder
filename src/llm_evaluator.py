
import json
import urllib.request
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

SCORED_FILE = BASE_DIR / "outputs" / "scored_candidates.json"
PARSED_FILE = BASE_DIR / "outputs" / "parsed_episodes.json"
OUTPUT_FILE = BASE_DIR / "outputs" / "llm_evaluations.json"

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "qwen2.5:3b"

# Evaluate these three different promo styles.
TEST_CANDIDATE_IDS = [
    "EP3_SC3_74_134",
    "EP43_SC1_10_70",
    "EP16_SC1_5_55"
]

# Qwen selects a category; Python assigns its numerical value.
LEVEL_SCORES = {
    "very_weak": 10,
    "weak": 30,
    "moderate": 50,
    "strong": 70,
    "exceptional": 90
}

# Keep the scoring weights consistent and deterministic.
WEIGHTS = {
    "hook": 30,
    "cliffhanger": 30,
    "emotional_impact": 15,
    "context_clarity": 10,
    "promo_suitability": 15
}


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
# CANDIDATE EVENT EXTRACTION
# ============================================================

def get_candidate_events(candidate, episodes):
    """Extract screenplay events overlapping a candidate window."""

    episode_number = candidate["episode_number"]
    scene_number = candidate["scene_number"]

    start_time = candidate["start_time"]
    end_time = candidate["end_time"]

    for episode in episodes:

        if episode["episode_number"] != episode_number:
            continue

        for scene in episode["scenes"]:

            if scene["scene_number"] != scene_number:
                continue

            selected_events = []

            for event in scene["events"]:

                if (
                    event["start_time"] < end_time
                    and event["end_time"] > start_time
                ):
                    selected_events.append({
                        "event_type": event["event_type"],
                        "speaker": event.get("speaker"),
                        "delivery": event.get("delivery"),
                        "start_time": event["start_time"],
                        "end_time": event["end_time"],
                        "content": event["content"]
                    })

            return episode, scene, selected_events

    raise ValueError(
        f"Could not find episode/scene for {candidate['candidate_id']}"
    )


# ============================================================
# PROMPT GENERATION
# ============================================================

def build_prompt(candidate, episode, scene, events):
    """Ask Qwen for categorical judgments, not numerical scores."""

    start = candidate["start_time"]
    end = candidate["end_time"]

    opening_events = [
        event for event in events
        if event["start_time"] < start + 3
        and event["end_time"] > start
    ]

    ending_events = [
        event for event in events
        if event["start_time"] < end
        and event["end_time"] > end - 5
    ]

    opening_data = json.dumps(
        opening_events,
        ensure_ascii=False,
        indent=2
    )

    ending_data = json.dumps(
        ending_events,
        ensure_ascii=False,
        indent=2
    )

    full_data = json.dumps(
        events,
        ensure_ascii=False,
        indent=2
    )

    return f"""
You are an expert microdrama advertising evaluator.

Evaluate this screenplay segment as a potential short-form
social media promo.

CANDIDATE:
ID: {candidate['candidate_id']}
Episode: {candidate['episode_number']}
Scene: {candidate['scene_number']}
Start: {start} seconds
End: {end} seconds
Duration: {candidate['duration']} seconds

EPISODE CONTEXT:
Logline: {episode.get('logline', '')}
Continuity: {episode.get('continuity_notes', '')}
Scene heading: {scene.get('scene_heading', '')}


==================================================
OPENING HOOK - FIRST 3 SECONDS
==================================================

Time window: {start} to {start + 3} seconds.

{opening_data}

Evaluate the opening using dialogue, delivery, action,
conflict, mystery and visual descriptions.

Physical violence and shocking reveals are NOT required
for a strong hook.

Confrontational dialogue, threats, mystery and emotional
tension can independently create a strong hook.


==================================================
ENDING CLIFFHANGER - FINAL 5 SECONDS
==================================================

Time window: {end - 5} to {end} seconds.

{ending_data}

Evaluate whether the ending creates an unresolved question,
conflict, decision, revelation or anticipation.

A dramatic action is not automatically a cliffhanger
if the story already feels resolved.


==================================================
COMPLETE SCREENPLAY EVENTS
==================================================

{full_data}


==================================================
CATEGORICAL SCORING SYSTEM
==================================================

For EACH dimension, select exactly ONE category.

VERY_WEAK:
Almost no meaningful evidence of quality.

WEAK:
Limited evidence. Some interest, but significant weaknesses.

MODERATE:
Noticeable quality, but important limitations remain.

STRONG:
Clear and compelling evidence of quality.

EXCEPTIONAL:
Outstanding evidence and highly compelling execution.

Evaluate these FIVE dimensions:

1. hook_level:
Only the first 3 seconds.

2. cliffhanger_level:
Only the final 5 seconds.

3. emotional_impact_level:
Emotional intensity throughout the segment.

4. context_clarity_level:
Can someone unfamiliar with the series understand
the basic situation and why it matters?

5. promo_suitability_level:
Is the complete segment suitable as a short-form promo?
Consider duration, coherence, engagement and story appeal.


IMPORTANT RULES:

- Return categories, NOT numerical scores.
- Use only supplied screenplay events as evidence.
- Never invent dialogue, characters or actions.
- Do not attribute later events to the opening.
- Do not attribute events outside the candidate's end time
  to the ending.
- Episode context can explain the story but is not proof
  of an event occurring inside the promo.
- Be critical and consistent.
- A strong hook does not require physical violence.
- Explain each classification using actual evidence.

Return ONLY valid JSON:

{{
    "hook_level": "strong",
    "cliffhanger_level": "moderate",
    "emotional_impact_level": "strong",
    "context_clarity_level": "moderate",
    "promo_suitability_level": "strong",

    "hook_explanation": "Evidence from the opening",
    "cliffhanger_explanation": "Evidence from the ending",
    "emotional_impact_explanation": "Explanation",
    "context_clarity_explanation": "Explanation",
    "promo_suitability_explanation": "Explanation",

    "main_weakness": "Main weakness or None"
}}

Allowed category values:
very_weak, weak, moderate, strong, exceptional.

Do not return any numerical scores.
"""


# ============================================================
# OLLAMA API
# ============================================================

def evaluate_with_ollama(prompt):
    """Send a prompt to the local Ollama model."""

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a careful and consistent screenplay "
                    "evaluator. Select categorical quality levels "
                    "using the supplied evidence. Never invent "
                    "evidence. Return valid JSON only."
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
# VALIDATE CATEGORICAL JUDGMENTS
# ============================================================

def validate_judgment(judgment):
    """Validate Qwen's categorical response."""

    level_fields = [
        "hook_level",
        "cliffhanger_level",
        "emotional_impact_level",
        "context_clarity_level",
        "promo_suitability_level"
    ]

    explanation_fields = [
        "hook_explanation",
        "cliffhanger_explanation",
        "emotional_impact_explanation",
        "context_clarity_explanation",
        "promo_suitability_explanation",
        "main_weakness"
    ]

    # Step 1: Validate all five scoring categories.
    for field in level_fields:

        if field not in judgment:
            raise ValueError(f"Missing field: {field}")

        if not isinstance(judgment[field], str):
            raise ValueError(f"{field} must be a string")

        # Normalize category formatting.
        judgment[field] = (
            judgment[field]
            .strip()
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
        )

        if judgment[field] not in LEVEL_SCORES:
            raise ValueError(
                f"Invalid category for {field}: {judgment[field]}"
            )

    # Step 2: Missing explanations are not fatal.
    for field in explanation_fields:

        if field not in judgment or not isinstance(judgment[field], str):
            judgment[field] = "Explanation not provided by the model."

        elif not judgment[field].strip():
            judgment[field] = "Explanation not provided by the model."

    return judgment

# ============================================================
# DETERMINISTIC SCORE CALCULATION
# ============================================================

def calculate_final_scores(judgment):
    """Convert categories into scores using Python."""

    scores = {
        "hook": LEVEL_SCORES[judgment["hook_level"]],
        "cliffhanger": LEVEL_SCORES[judgment["cliffhanger_level"]],
        "emotional_impact": LEVEL_SCORES[
            judgment["emotional_impact_level"]
        ],
        "context_clarity": LEVEL_SCORES[
            judgment["context_clarity_level"]
        ],
        "promo_suitability": LEVEL_SCORES[
            judgment["promo_suitability_level"]
        ]
    }

    weighted_total = sum(
        scores[name] * weight
        for name, weight in WEIGHTS.items()
    ) / sum(WEIGHTS.values())

    scores["overall_quality"] = round(weighted_total, 2)

    return scores


# ============================================================
# RECOMMENDATION LOGIC
# ============================================================

def make_recommendation(scores):
    """Generate recommendation using deterministic rules."""

    overall = scores["overall_quality"]

    if (
        overall >= 60
        and (
            scores["hook"] >= 50
            or scores["cliffhanger"] >= 70
        )
    ):
        return True

    return False


# ============================================================
# EVALUATE ONE CANDIDATE
# ============================================================

def evaluate_candidate(candidate, episodes):
    """Evaluate one candidate using its actual screenplay."""

    episode, scene, events = get_candidate_events(
        candidate,
        episodes
    )

    if not events:
        raise ValueError(
            f"No events found for {candidate['candidate_id']}"
        )

    prompt = build_prompt(
        candidate,
        episode,
        scene,
        events
    )

    judgment = evaluate_with_ollama(prompt)

    judgment = validate_judgment(judgment)

    scores = calculate_final_scores(judgment)

    recommended = make_recommendation(scores)

    return {
        "candidate_id": candidate["candidate_id"],
        "episode_number": candidate["episode_number"],
        "scene_number": candidate["scene_number"],
        "start_time": candidate["start_time"],
        "end_time": candidate["end_time"],
        "duration": candidate["duration"],
        "model": MODEL_NAME,

        # Original model classifications and explanations.
        "llm_judgments": judgment,

        # Scores calculated by Python.
        "llm_scores": scores,

        # Recommendation calculated by Python.
        "recommended": recommended
    }


# ============================================================
# EVALUATE THREE DIVERSE CANDIDATES
# ============================================================

def evaluate_test_candidates():
    """Evaluate three manually selected promo candidates."""

    candidates = load_json(SCORED_FILE)
    episodes = load_json(PARSED_FILE)

    candidate_lookup = {
        candidate["candidate_id"]: candidate
        for candidate in candidates
    }

    results = []
    failed_candidates = []

    print("\n========== MULTI-CANDIDATE LLM EVALUATION ==========")
    print("Model:", MODEL_NAME)

    for candidate_id in TEST_CANDIDATE_IDS:

        print("\n----------------------------------------")
        print("Candidate:", candidate_id)

        # Check whether the candidate exists.
        if candidate_id not in candidate_lookup:

            print("Candidate not found. Skipping.")

            failed_candidates.append({
                "candidate_id": candidate_id,
                "error": "Candidate not found"
            })

            continue

        candidate = candidate_lookup[candidate_id]

        print("Rule-based score:", candidate["total_score"])
        print("Sending screenplay to Ollama...")

        try:
            # Evaluate this candidate.
            result = evaluate_candidate(candidate, episodes)

            results.append(result)

            print("\nLLM classifications:")

            for field, value in result["llm_judgments"].items():

                if field.endswith("_level"):
                    print(f"{field}: {value}")

            print("\nPython-calculated scores:")

            for field, value in result["llm_scores"].items():
                print(f"{field}: {value}")

            print("Recommended:", result["recommended"])

        except Exception as error:

            print("Evaluation failed:", error)

            failed_candidates.append({
                "candidate_id": candidate_id,
                "error": str(error)
            })

            # Continue to the next candidate.
            continue

    # Rank successful evaluations by overall quality.
    results.sort(
        key=lambda item: item["llm_scores"]["overall_quality"],
        reverse=True
    )

    # Save successful evaluations.
    save_json(OUTPUT_FILE, results)

    print("\n========== FINAL TEST RANKING ==========")

    for index, result in enumerate(results, start=1):

        print(
            f"{index}. {result['candidate_id']} | "
            f"LLM Score: "
            f"{result['llm_scores']['overall_quality']} | "
            f"Recommended: {result['recommended']}"
        )

    print("\n========== EVALUATION SUMMARY ==========")

    print("Successful evaluations:", len(results))
    print("Failed evaluations:", len(failed_candidates))
    print("Saved results to:", OUTPUT_FILE)

    # Save failures separately for debugging.
    if failed_candidates:

        error_file = (
            BASE_DIR / "outputs" / "llm_evaluation_errors.json"
        )

        save_json(error_file, failed_candidates)

        print("Saved errors to:", error_file)

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    evaluate_test_candidates()
