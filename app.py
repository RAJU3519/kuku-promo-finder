from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from src.parser import extract_episode_metadata, extract_scenes, extract_events
from src.candidate_generator import generate_all_candidates
from src.scorer import score_candidate
from src.multi_segment_generator import (
    select_segment_pool, generate_multi_segment_candidates,
)
from src.multi_segment_scorer import (
    score_multi_candidate,
    diversify_multi_rankings,
)


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"


def load_json(path: Path):
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []


def parse_uploaded_screenplays(uploaded_files):
    """Parse uploaded .txt screenplays using the project's existing parser."""
    episodes, errors = [], []
    with tempfile.TemporaryDirectory() as temp_dir:
        for uploaded in uploaded_files:
            if not uploaded.name.lower().endswith(".txt"):
                errors.append(f"{uploaded.name}: only .txt files are supported")
                continue
            file_path = Path(temp_dir) / Path(uploaded.name).name
            file_path.write_bytes(uploaded.getvalue())
            try:
                metadata = extract_episode_metadata(file_path)
                text = file_path.read_text(encoding="utf-8-sig")
                scenes = extract_scenes(file_path)
                scene_matches = list(re.finditer(
                    r"^[ \t]*SCENE[ \t]+\d+[ \t]*$", text,
                    re.MULTILINE | re.IGNORECASE
                ))
                if len(scene_matches) != len(scenes):
                    raise ValueError("Scene heading count does not match extracted scenes")
                for i, scene in enumerate(scenes):
                    left = scene_matches[i].start()
                    right = scene_matches[i + 1].start() if i + 1 < len(scene_matches) else len(text)
                    scene["events"] = extract_events(text[left:right])
                if metadata["episode_number"] is None:
                    raise ValueError("Episode number could not be extracted")
                episodes.append({
                    "episode_number": metadata["episode_number"],
                    "source_file": uploaded.name,
                    "logline": metadata["logline"],
                    "continuity_notes": metadata["continuity_notes"],
                    "scenes": scenes,
                })
            except Exception as exc:
                errors.append(f"{uploaded.name}: {exc}")
    episodes.sort(key=lambda x: x["episode_number"])
    return episodes, errors


def score_episodes(episodes):
    """Generate and rank single-scene and multi-segment promo candidates."""
    candidates = generate_all_candidates(episodes)
    scored = [score_candidate(candidate) for candidate in candidates]
    scored.sort(key=lambda x: x["total_score"], reverse=True)

    segment_pool = select_segment_pool(candidates, scored)
    multi_candidates = generate_multi_segment_candidates(segment_pool)
    multi_scored = [
        score_multi_candidate(candidate)
        for candidate in multi_candidates
    ]
    multi_scored.sort(
        key=lambda item: item["total_score"],
        reverse=True
    )
    for rank, item in enumerate(multi_scored, 1):
        item["rank"] = rank

    return candidates, scored, multi_candidates, multi_scored


def _ranking_table_row(record, rank):
    """Build a useful table row for both single- and multi-segment results."""
    segments = record.get("segments") or []
    if segments:
        episode_label = " → ".join(f"EP{s.get('episode_number', '?')}" for s in segments)
        scene_label = " → ".join(f"SC{s.get('scene_number', '?')}" for s in segments)
        start_label = " → ".join(str(s.get("start_time", "?")) for s in segments)
        end_label = " → ".join(str(s.get("end_time", "?")) for s in segments)
    else:
        episode_label = record.get("episode_number", "—")
        scene_label = record.get("scene_number", "—")
        start_label = record.get("start_time", "—")
        end_label = record.get("end_time", "—")

    return {
        "Rank": rank,
        "Candidate": record.get("candidate_id", "Unknown"),
        "Episode(s)": episode_label,
        "Scene(s)": scene_label,
        "Start(s)": start_label,
        "End(s)": end_label,
        "Duration(s)": record.get("duration", "—"),
        "Score": record.get("total_score", "—"),
    }


def show_rankings(records, title):
    st.subheader(title)
    if not records:
        st.info("No ranked candidates are available yet.")
        return

    frame = pd.DataFrame([
        _ranking_table_row(record, i + 1)
        for i, record in enumerate(records)
    ])
    st.dataframe(frame, width="stretch", hide_index=True)

    ids = [r["candidate_id"] for r in records]
    selected_id = st.selectbox("Inspect a promo candidate", ids, key=title)
    selected = next(r for r in records if r["candidate_id"] == selected_id)
    st.markdown(f"**{selected_id}**")
    st.write(
        f"Duration: {selected.get('duration')} seconds · "
        f"Score: {selected.get('total_score')}/100"
    )

    segments = selected.get("segments") or []
    if segments:
        st.markdown("**Promo segments in playback order**")
        for i, segment in enumerate(segments, 1):
            st.write(
                f"{i}. Episode {segment.get('episode_number', '?')}, "
                f"Scene {segment.get('scene_number', '?')}: "
                f"{segment.get('start_time', '?')}–"
                f"{segment.get('end_time', '?')} seconds "
                f"({segment.get('duration', '?')} sec)"
            )

    scores = selected.get("scores", {})
    if scores:
        st.markdown("**Score breakdown**")
        for dimension, value in scores.items():
            numeric = max(0, min(100, int(round(float(value)))))
            label = dimension.replace("_", " ").title()
            st.write(f"{label}: **{value}/100**")
            st.progress(numeric)

    for dimension, reasons in selected.get("reasons", {}).items():
        with st.expander(dimension.replace("_", " ").title()):
            if isinstance(reasons, str):
                st.write(reasons)
            else:
                for reason in reasons:
                    st.write(f"• {reason}")


st.set_page_config(page_title="Kuku Promo Finder", page_icon="🎬", layout="wide")
st.title("🎬 Kuku Promo Finder")
st.caption("Screenplay-based promo discovery · 20–60 second candidates · evidence-aware heuristic ranking")
st.warning(
    "Scores are decision-support signals, not objective truth. This tool analyses screenplay text and timestamps; "
    "it cannot assess acting, actual visual execution, music, or edit pacing."
)

tab_upload, tab_existing, tab_method = st.tabs(["Analyze screenplays", "Existing results", "How ranking works"])

with tab_existing:
    rankings = load_json(OUTPUTS / "scored_candidates.json")
    if not rankings:
        st.info("No existing scored_candidates.json found. Use the Analyze screenplays tab.")
    else:
        st.metric("Ranked candidates", len(rankings))
        show_rankings(rankings[:100], "Existing single-scene rankings")
    multi = load_json(OUTPUTS / "scored_multi_segment_candidates.json")
    if multi:
        st.divider()
        show_rankings(
            diversify_multi_rankings(multi, max_items=100),
            "Existing multi-segment rankings (near-duplicates suppressed)"
        )

with tab_upload:
    st.write("Upload one or more screenplay `.txt` files in the same format as the supplied dataset.")
    uploaded = st.file_uploader(
        "Screenplay files",
        type=["txt"],
        accept_multiple_files=True,
    )
    if st.button("Analyze uploaded screenplays", type="primary", disabled=not uploaded):
        with st.spinner("Parsing screenplays and generating promo candidates..."):
            episodes, errors = parse_uploaded_screenplays(uploaded)
            (
                candidates, scored, multi_candidates, multi_scored
            ) = score_episodes(episodes) if episodes else ([], [], [], [])
        st.session_state["upload_episodes"] = episodes
        st.session_state["upload_candidates"] = candidates
        st.session_state["upload_scored"] = scored
        st.session_state["upload_multi_candidates"] = multi_candidates
        st.session_state["upload_multi_scored"] = multi_scored
        st.session_state["upload_errors"] = errors

    if "upload_scored" in st.session_state:
        episodes = st.session_state["upload_episodes"]
        candidates = st.session_state["upload_candidates"]
        scored = st.session_state["upload_scored"]
        multi_candidates = st.session_state.get("upload_multi_candidates", [])
        multi_scored = st.session_state.get("upload_multi_scored", [])
        errors = st.session_state["upload_errors"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Episodes parsed", len(episodes))
        c2.metric("Single-scene candidates", len(candidates))
        c3.metric("Single-scene ranked", len(scored))
        c4.metric("Multi-segment ranked", len(multi_scored))
        for error in errors:
            st.error(error)
        show_rankings(scored[:100], "Uploaded screenplay rankings")
        if scored:
            export = json.dumps({
                "episodes_processed": len(episodes),
                "single_scene_rankings": scored,
                "multi_segment_rankings": multi_scored,
                "notes": [
                    "Scores are heuristic decision-support signals.",
                    "Multi-segment candidates are assembled from selected windows in chronological screenplay order.",
                    "Human review is required to confirm narrative continuity between segments.",
                ],
            }, indent=2, ensure_ascii=False)
            st.download_button(
                "Download all ranked results (JSON)",
                data=export,
                file_name="promo_rankings.json",
                mime="application/json",
            )

        st.subheader("Uploaded multi-segment promo rankings")
        if multi_scored:
            show_rankings(
                diversify_multi_rankings(multi_scored, max_items=100),
                "Uploaded multi-segment rankings (near-duplicates suppressed)"
            )
            st.caption(
                "The table suppresses highly overlapping segment combinations for variety. "
                "The JSON export retains the full ranked candidate list."
            )
        else:
            st.info(
                "No multi-segment candidates could be assembled from this upload. "
                "The current generator requires segments from at least two distinct scenes. "
                "Try uploading multiple episodes or a screenplay containing multiple scenes."
            )

with tab_method:
    st.subheader("Evaluation dimensions")
    st.markdown("""
    - **Hook:** dialogue, action, delivery, and camera signals in the first 3 seconds.
    - **Cliffhanger:** unresolved dialogue, action, and visual reveal signals in the final 5 seconds.
    - **Emotional intensity:** emotional language and delivery markers.
    - **Context:** dialogue exchanges, identifiable speakers, action, and visual descriptions.
    - **Promo suitability:** duration, screenplay event types, and available event detail.
    - **Segment continuity (multi-segment only):** combines structural links (chronology, recurring credited speakers, shared setting cues) with dialogue-level edit-junction checks (cut-off lines, context-dependent reaction openings, and traceable dialogue handoffs). This is a transparent screenplay heuristic, not a first-time-viewer comprehension test.

    The current ranking is heuristic. Boundary-crossing events are withheld from opening-hook scoring because
    the screenplay format does not provide word-level timing. Such events are surfaced as uncertain evidence.
    """)
    st.info(
        "Context is a coverage checklist (dialogue exchanges, speaker variety, action and camera descriptions), "
        "not a semantic first-time-viewer clarity score. Promo suitability checks format and available event "
        "types rather than predicting ad performance. For multi-segment promos, continuity signals use "
        "chronological order, recurring credited speakers and shared setting cues. These heuristics support "
        "shortlisting; an editor must still judge story clarity and the cut between segments. "
        "Local Ollama semantic evaluation remains a separate optional pipeline step."
    )
