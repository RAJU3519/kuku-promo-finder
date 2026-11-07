import json
from pathlib import Path

from src.scorer import score_hook, score_cliffhanger, score_context

ROOT = Path(__file__).resolve().parents[1]


def test_hook_excludes_dialogue_crossing_three_second_boundary():
    candidate = {
        "start_time": 64.0,
        "end_time": 77.5,
        "events": [
            {
                "event_type": "DIALOGUE",
                "start_time": 64.2,
                "end_time": 72.4,
                "content": "This is my son. The rightful Duke of Ashburn!",
                "delivery": "Authoritative",
            },
            {
                "event_type": "ACTION",
                "start_time": 64.0,
                "end_time": 72.5,
                "content": "His mother points proudly toward Alexander.",
            },
        ],
    }
    score, reasons = score_hook(candidate)
    assert score == 0
    assert any("Uncertain hook evidence: 2 event(s)" in reason for reason in reasons)


def test_hook_scores_fully_contained_hook_event():
    candidate = {
        "start_time": 10.0,
        "end_time": 40.0,
        "events": [
            {
                "event_type": "DIALOGUE",
                "start_time": 10.2,
                "end_time": 12.2,
                "content": "Arrest him!",
                "delivery": "Furious",
            }
        ],
    }
    score, reasons = score_hook(candidate)
    assert score > 0
    assert any("arrest him" in reason.lower() for reason in reasons)


def test_cliffhanger_detects_decision_and_document_reveal():
    candidate = {
        "start_time": 0.0,
        "end_time": 30.0,
        "events": [
            {
                "event_type": "DIALOGUE",
                "start_time": 26.1,
                "end_time": 28.0,
                "content": "Then let the Crown decide.",
                "delivery": None,
            },
            {
                "event_type": "ACTION",
                "start_time": 28.2,
                "end_time": 30.0,
                "content": "Alexander brings forth the dossier stamped WAR OFFICE - TOP SECRET.",
                "delivery": None,
            },
            {
                "event_type": "CAMERA",
                "start_time": 28.0,
                "end_time": 30.0,
                "content": "Close-up of the dossier sealed with red wax.",
                "delivery": None,
            },
        ],
    }
    score, reasons = score_cliffhanger(candidate)
    assert score == 60
    assert any("Unresolved dialogue setup" in reason for reason in reasons)
    assert any("Unresolved evidence or document reveal" in reason for reason in reasons)
    assert any("Dramatic final visual framing" in reason for reason in reasons)


def test_project_outputs_have_expected_candidate_count():
    path = ROOT / "outputs" / "promo_candidates.json"
    if not path.exists():
        return
    candidates = json.loads(path.read_text(encoding="utf-8"))
    assert len(candidates) == 3140
