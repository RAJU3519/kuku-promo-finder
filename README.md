# Kuku Promo Finder

A Streamlit application that analyses uploaded microdrama screenplay `.txt` files and ranks 20–60 second promotional clip candidates.

## Prototype

The app is designed for Streamlit Community Cloud. The repository intentionally excludes the supplied screenplay pack and generated raw outputs. Users upload screenplay files through the app.

## Features

- Parses episode metadata, scene boundaries, timestamps, dialogue, action and camera descriptions.
- Generates 20/30/40/50/60-second single-scene candidates.
- Scores hook, cliffhanger, emotional intensity, context and promo suitability.
- Generates two- and three-segment combinations across distinct scenes/episodes with a 20–60 second total duration.
- Scores multi-segment continuity using chronology, recurring credited speakers, setting cues and dialogue-level edit-junction checks.
- Suppresses highly overlapping variants in the visible shortlist while retaining all scored candidates in JSON export.
- Exports full ranked results as JSON.

## Run locally (Windows PowerShell)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local URL printed by Streamlit, usually `http://localhost:8501`.

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub.
2. Sign in at https://share.streamlit.io/ with GitHub.
3. Choose **Create app** → **Yup, I have an app**.
4. Select this repository, branch `main`, and entrypoint `app.py`.
5. In Advanced settings, select Python 3.12.
6. Deploy. Once it starts, use the app's sharing settings to ensure it is public for reviewers.

## Tests

Install development dependencies and run:

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

## Important limitations

- This tool analyses screenplay text and timestamps, not rendered video. It cannot assess acting, actual visual execution, music, shot quality or edit pacing.
- Script event timestamps are not word-level timestamps. The opening-hook scorer excludes partially overlapping events from positive evidence and reports uncertainty.
- Multi-segment continuity is a transparent screenplay heuristic. It is not proof of first-time-viewer comprehension or a substitute for human editorial review.
- Context and promo-suitability scores are heuristic coverage signals; they do not predict audience response or ad performance.
- Scores are decision-support signals. Review the original source segments before choosing a promo.

## Privacy and repository contents

The GitHub deployment repository excludes `data/` (source screenplays), `outputs/` (generated JSON results), `.venv/`, and local secrets. The app processes uploaded screenplay text in memory/temp files for the current analysis; do not upload confidential or restricted material unless you are authorized to do so.
