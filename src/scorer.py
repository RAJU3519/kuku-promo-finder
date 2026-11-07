from pathlib import Path
import json
import re


WEIGHTS = {
    "hook": 30,
    "cliffhanger": 30,
    "emotional_intensity": 15,
    "context": 10,
    "promo_suitability": 15
}


HOOK_PHRASES = [
    "who are you",
    "stand aside",
    "let her go",
    "let him go",
    "i said",
    "arrest him",
    "you already know",
    "how dare you",
    "what have you done",
    "you will not",
    "we will not",
    "you cannot",
    "help me",
    "stay away",
    "get away"
]

HOOK_WORDS = [
    "betray", "kill", "die", "murder",
    "attack", "revenge", "traitor",
    "stop", "run", "help", "dead"
]
IDENTITY_REVELATION_PHRASES = [
    "this is my son",
    "rightful duke",
    "rightful king",
    "rightful heir",
    "he is your duke",
    "she is the princess",
    "he is the prince",
    "i am your father",
    "i am your son",
    "i am alive",
    "he is alive",
    "she is alive",
]
HOOK_ACTIONS = [
    "slap", "punch", "strike", "attack",
    "collapse", "fall", "grab", "throw",
    "charge", "draw", "burst", "rush",
    "strikes", "attacks", "collapses"
]

CLIFFHANGER_PHRASES = [
    "what if",
    "what happened",
    "who is",
    "who was",
    "you don't know",
    "you have no idea",
    "the truth is",
    "when you believed",
    "before it is too late",
    "i will tell you",
    "then let the crown decide",
]

CLIFFHANGER_ACTIONS = [
    "raises his hand",
    "raises her hand",
    "draws his sword",
    "draws her sword",
    "opens the door",
    "reaches for",
    "pulls out",
    "reveals",
    "kneels",
    "offering the ring",
    "offering a ring",
    "raises the ring",
    "steps forward",
    "turns toward",
    "gasps",
    "freezes",
    "sinks to one knee",
    "one knee",
    "offering the silver ring",
    "offering a silver ring",
    "offering her a ring",
    "kneeling before",
    "proposes",

]
REVEAL_ACTIONS = [
    "brings forth the dossier",
    "sealed with red wax",
    "stamped war office",
    "top secret",
]
EMOTION_WORDS = [
    "love", "hate", "fear", "cry", "tears",
    "betrayal", "lost", "heartbroken", "desperate",
    "terrified", "furious", "shocked", "devastated",
    "revenge", "protect", "worth", "forgive"
]

INTENSE_DELIVERIES = [
    "furious", "terrified", "desperate",
    "devastated", "screaming", "shouting",
    "heartbroken", "aggressive", "panicked",
    "threatening", "anguished"
]


def normalize_text(text):
    return re.sub(r"\s+", " ", str(text or "").lower()).strip()


def phrase_matches(text, phrases):
    """Match complete words or phrases, avoiding accidental substrings."""
    hits = []

    for phrase in phrases:
        pattern = r"(?<!\w)" + re.escape(phrase) + r"(?!\w)"

        if re.search(pattern, text):
            hits.append(phrase)

    return hits


def get_events_in_range(candidate, start, end):
    return [
        event for event in candidate.get("events", [])
        if event["start_time"] < end
        and event["end_time"] > start
    ]


def event_text(events, event_type=None):
    selected = events

    if event_type:
        selected = [
            event for event in events
            if event["event_type"] == event_type
        ]

    return " ".join(
        normalize_text(
            f"{event.get('content', '')} {event.get('delivery', '')}"
        )
        for event in selected
    )

def get_fully_contained_events(candidate, start, end):
    events = get_events_in_range(candidate, start, end)

    return [
        event for event in events
        if event["start_time"] >= start
        and event["end_time"] <= end
    ]
def get_partial_events(candidate, start, end):
    events = get_events_in_range(candidate, start, end)

    return [
        event for event in events
        if event["start_time"] < start
        or event["end_time"] > end
    ]

def score_hook(candidate):
    start = candidate["start_time"]
    end = min(start + 3, candidate["end_time"])
    events = get_fully_contained_events(candidate, start, end)
    partial_events = get_partial_events(candidate, start, end)


    dialogue = event_text(events, "DIALOGUE")
    actions = event_text(events, "ACTION")
    camera = event_text(events, "CAMERA")

    score = 0
    reasons = []

    phrase_hits = phrase_matches(dialogue, HOOK_PHRASES)
    word_hits = phrase_matches(dialogue, HOOK_WORDS)
    revelation_hits = phrase_matches(
        dialogue,
        IDENTITY_REVELATION_PHRASES
    )
    action_hits = phrase_matches(actions, HOOK_ACTIONS)

    delivery_hits = phrase_matches(
        dialogue,
        INTENSE_DELIVERIES
    )

    visual_hits = phrase_matches(
        camera,
        ["close-up", "extreme close-up", "low-angle", "sudden"]
    )

    if phrase_hits:
        score += min(len(phrase_hits) * 25, 50)
        reasons.append(
            "Confrontational or attention-grabbing dialogue: "
            + ", ".join(phrase_hits)
        )

    if word_hits:
        score += min(len(word_hits) * 15, 30)
        reasons.append("High-impact dialogue detected")

    if action_hits:
        score += min(len(action_hits) * 20, 40)
        reasons.append(
            "Dramatic physical action: " + ", ".join(action_hits)
        )
    if revelation_hits:
        score += min(len(revelation_hits) * 25, 50)
        reasons.append(
            "Identity or status revelation detected: "
            + ", ".join(revelation_hits)
    )

    if delivery_hits:
        score += min(len(delivery_hits) * 15, 30)
        reasons.append("Intense character delivery detected")

    if visual_hits:
        score += 10
        reasons.append("Dramatic camera framing detected")
    if partial_events:
        reasons.append(
            f"Uncertain hook evidence: {len(partial_events)} event(s) "
            "overlap the 3-second boundary and were not fully scored"
    )


    score = min(score, 100)

    if not reasons:
        reasons.append("No strong opening hook signals detected")

    return score, reasons


def score_cliffhanger(candidate):
    end = candidate["end_time"]
    start = max(candidate["start_time"], end - 5)

    events = get_events_in_range(candidate, start, end)

    dialogue = event_text(events, "DIALOGUE")
    actions = event_text(events, "ACTION")
    camera = event_text(events, "CAMERA")

    score = 0
    reasons = []

    phrase_hits = phrase_matches(
        dialogue,
        CLIFFHANGER_PHRASES
    )

    action_hits = phrase_matches(
        actions,
        CLIFFHANGER_ACTIONS
    )
    reveal_hits = phrase_matches(
        actions + " " + camera,
        REVEAL_ACTIONS
    )

    visual_hits = phrase_matches(
        camera,
        ["close-up", "extreme close-up", "reaction shot"]
    )

    # Incomplete dialogue is a useful cliffhanger signal.
    dialogue_events = [
        event for event in events
        if event["event_type"] == "DIALOGUE"
    ]

    incomplete_dialogue = any(
        event.get("content", "").rstrip().endswith(("...", "…"))
        for event in dialogue_events
    )

    if phrase_hits:
        score += min(len(phrase_hits) * 20, 40)
        reasons.append(
            "Unresolved dialogue setup: " + ", ".join(phrase_hits)
        )

    if incomplete_dialogue:
        score += 30
        reasons.append("Dialogue ends on an incomplete thought")

    if action_hits:
        score += min(len(action_hits) * 20, 40)
        reasons.append(
            "Potential unresolved visual action: "
            + ", ".join(action_hits)
        )

    if reveal_hits:
        score += 30
        reasons.append(
           f"Unresolved evidence or document reveal: {', '.join(reveal_hits)}"
        )

    if visual_hits:
        score += 10
        reasons.append("Dramatic final visual framing detected")

    score = min(score, 100)

    if not reasons:
        reasons.append("No strong ending cliffhanger signals detected")

    return score, reasons


def score_emotional_intensity(candidate):
    events = [
        event for event in candidate.get("events", [])
        if event["event_type"] in ("DIALOGUE", "ACTION")
    ]

    combined = event_text(events)

    emotion_hits = phrase_matches(
        combined,
        EMOTION_WORDS
    )

    delivery_hits = phrase_matches(
        combined,
        INTENSE_DELIVERIES
    )

    score = min(
        len(emotion_hits) * 12 + len(delivery_hits) * 15,
        100
    )

    reasons = []

    if emotion_hits:
        reasons.append(
            "Emotional language: " + ", ".join(emotion_hits)
        )

    if delivery_hits:
        reasons.append("Intense emotional delivery detected")

    if not reasons:
        reasons.append("Limited emotional signals detected")

    return score, reasons


def score_context(candidate):
    events = candidate.get("events", [])

    dialogue = [
        event for event in events
        if event["event_type"] == "DIALOGUE"
    ]

    speakers = {
        event.get("speaker")
        for event in dialogue
        if event.get("speaker")
    }

    score = 0
    reasons = []

    if len(dialogue) >= 3:
        score += 40
        reasons.append("Multiple dialogue exchanges")
    elif dialogue:
        score += 20
        reasons.append("Dialogue provides some story context")

    if len(speakers) >= 2:
        score += 30
        reasons.append("Multiple identifiable speakers")
    elif len(speakers) == 1:
        score += 15
        reasons.append("One identifiable speaker")

    if any(
        event["event_type"] == "ACTION"
        for event in events
    ):
        score += 20
        reasons.append("Physical action supports the scene")

    if any(
        event["event_type"] == "CAMERA"
        for event in events
    ):
        score += 10
        reasons.append("Visual descriptions are available")

    return min(score, 100), reasons


def score_promo_suitability(candidate):
    events = candidate.get("events", [])

    event_types = {
        event["event_type"]
        for event in events
    }

    duration = candidate["duration"]

    score = 0
    reasons = []

    if 20 <= duration <= 60:
        score += 40
        reasons.append("Within the required duration")

    if "DIALOGUE" in event_types:
        score += 20
        reasons.append("Dialogue available")

    if "ACTION" in event_types:
        score += 20
        reasons.append("Physical action available")

    if "CAMERA" in event_types:
        score += 10
        reasons.append("Visual descriptions available")

    if len(events) >= 5:
        score += 10
        reasons.append("Enough screenplay events for analysis")

    return min(score, 100), reasons


def score_candidate(candidate):
    hook, hook_reasons = score_hook(candidate)
    cliff, cliff_reasons = score_cliffhanger(candidate)
    emotion, emotion_reasons = score_emotional_intensity(candidate)
    context, context_reasons = score_context(candidate)
    suitability, suitability_reasons = score_promo_suitability(candidate)

    scores = {
        "hook": hook,
        "cliffhanger": cliff,
        "emotional_intensity": emotion,
        "context": context,
        "promo_suitability": suitability
    }

    total = sum(
        scores[key] * WEIGHTS[key] / 100
        for key in WEIGHTS
    )

    return {
        "candidate_id": candidate["candidate_id"],
        "episode_number": candidate["episode_number"],
        "scene_number": candidate["scene_number"],
        "start_time": candidate["start_time"],
        "end_time": candidate["end_time"],
        "duration": candidate["duration"],
        "scores": scores,
        "total_score": round(total, 2),
        "reasons": {
            "hook": hook_reasons,
            "cliffhanger": cliff_reasons,
            "emotional_intensity": emotion_reasons,
            "context": context_reasons,
            "promo_suitability": suitability_reasons
        }
    }


def main():
    root = Path(__file__).resolve().parents[1]

    input_path = root / "outputs" / "promo_candidates.json"
    output_path = root / "outputs" / "scored_candidates.json"

    with input_path.open("r", encoding="utf-8") as file:
        candidates = json.load(file)

    scored = [
        score_candidate(candidate)
        for candidate in candidates
    ]

    scored.sort(
        key=lambda item: item["total_score"],
        reverse=True
    )

    output_path.write_text(
        json.dumps(scored, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print("\n========== PROMO SCORING REPORT V2 ==========")
    print(f"Candidates scored: {len(scored)}")

    for rank, item in enumerate(scored[:10], 1):
        print(f"\n{rank}. {item['candidate_id']}")
        print(f"   Total: {item['total_score']}/100")

        for key, value in item["scores"].items():
            print(f"   {key}: {value}")

    print(f"\nSaved to: {output_path}")
    print("SCORING V2 COMPLETE")


if __name__ == "__main__":
    main()