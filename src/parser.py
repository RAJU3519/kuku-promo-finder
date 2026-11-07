
from pathlib import Path
from collections import defaultdict
import hashlib
import re


# ============================================================
# 1. FILE HASHING
# ============================================================

def calculate_file_hash(file_path: Path) -> str:
    """
    Calculate SHA-256 hash of a file.

    Identical files will have identical hashes.
    """

    sha256 = hashlib.sha256()

    with file_path.open("rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


# ============================================================
# 2. FILE DISCOVERY AND DUPLICATE DETECTION
# ============================================================

def discover_episode_files(folder: Path):
    """
    Discover screenplay files and identify exact duplicates.

    Returns:
        unique_files: One file per unique content hash.
        duplicate_groups: Groups of identical files.
        total_files: Total number of files discovered.
    """

    if not folder.exists():
        raise FileNotFoundError(
            f"Episode folder not found: {folder}"
        )

    if not folder.is_dir():
        raise NotADirectoryError(
            f"Expected a directory: {folder}"
        )

    text_files = sorted(
        [
            file for file in folder.iterdir()
            if file.is_file()
            and file.suffix.lower() == ".txt"
        ],
        key=lambda file: file.name.lower()
    )

    if not text_files:
        raise FileNotFoundError(
            "No .txt screenplay files found."
        )

    hash_groups = defaultdict(list)

    for file_path in text_files:
        file_hash = calculate_file_hash(file_path)
        hash_groups[file_hash].append(file_path)

    unique_files = []
    duplicate_groups = []

    for files in hash_groups.values():

        unique_files.append(files[0])

        if len(files) > 1:
            duplicate_groups.append(files)

    return (
        unique_files,
        duplicate_groups,
        len(text_files)
    )


# ============================================================
# 3. EPISODE METADATA EXTRACTION
# ============================================================

def extract_episode_metadata(file_path: Path) -> dict:
    """
    Extract episode number, logline and continuity notes
    from a screenplay file.
    """

    text = file_path.read_text(encoding="utf-8-sig")

    def extract_field(label: str):
        """
        Extract text between a metadata label and
        the next metadata label or screenplay marker.
        """

        pattern = (
            rf"^{re.escape(label)}:\s*"
            rf"(.*?)"
            rf"(?=^(?:EPISODE NUMBER|EPISODE LOGLINE|"
            rf"CONTINUITY NOTES):"
            rf"|^--- SCRIPT START ---|\Z)"
        )

        match = re.search(
            pattern,
            text,
            re.MULTILINE | re.DOTALL | re.IGNORECASE
        )

        return match.group(1).strip() if match else None

    episode_text = extract_field("EPISODE NUMBER")
    logline = extract_field("EPISODE LOGLINE")
    continuity = extract_field("CONTINUITY NOTES")

    # Extract numeric episode ID.
    episode_match = (
        re.search(r"\d+", episode_text)
        if episode_text else None
    )

    episode_number = (
        int(episode_match.group())
        if episode_match else None
    )

    # Basic metadata validation.
    warnings = []

    if episode_number is None:
        warnings.append("Episode number is missing.")

    if not logline:
        warnings.append("Episode logline is missing.")

    if not continuity:
        warnings.append("Continuity notes are missing.")

    return {
        "episode_number": episode_number,
        "logline": logline,
        "continuity_notes": continuity,
        "source_file": file_path.name,
        "warnings": warnings
    }


# ============================================================
# 4. MAIN FUNCTION
# ============================================================

def timestamp_to_seconds(timestamp: str) -> float:
    """
    Convert MM:SS.ss into seconds.
    Example: 01:23.45 -> 83.45
    """

    timestamp = timestamp.strip()

    match = re.fullmatch(
        r"(\d+):(\d{2}(?:\.\d+)?)",
        timestamp
    )

    if not match:
        raise ValueError(f"Invalid timestamp: {timestamp}")

    minutes = int(match.group(1))
    seconds = float(match.group(2))

    if seconds >= 60:
        raise ValueError(f"Invalid seconds: {timestamp}")

    return minutes * 60 + seconds


def extract_scenes(file_path: Path) -> list:
    """
    Extract scene headings and timestamps from a screenplay.
    """

    text = file_path.read_text(encoding="utf-8-sig")

    scene_pattern = re.compile(
        r"^[ \t]*SCENE[ \t]+(\d+)[ \t]*$",
        re.MULTILINE | re.IGNORECASE
    )

    matches = list(scene_pattern.finditer(text))

    if not matches:
        raise ValueError(f"No scenes found: {file_path.name}")

    scenes = []

    for index, scene_match in enumerate(matches):

        scene_number = int(scene_match.group(1))

        block_start = scene_match.start()

        block_end = (
            matches[index + 1].start()
            if index + 1 < len(matches)
            else len(text)
        )

        block = text[block_start:block_end]

        def find_field(pattern):
            return re.search(
                pattern,
                block,
                re.MULTILINE | re.IGNORECASE
            )

        heading = find_field(
            r"^[ \t]*Scene heading:[ \t]*(.+?)[ \t]*$"
        )

        start = find_field(
            r"^[ \t]*Scene start:[ \t]*(.+?)[ \t]*$"
        )

        end = find_field(
            r"^[ \t]*Scene end:[ \t]*(.+?)[ \t]*$"
        )

        if not heading or not start or not end:
            raise ValueError(
                f"Missing scene data in {file_path.name}, "
                f"Scene {scene_number}"
            )

        raw_start = start.group(1).strip()
        raw_end = end.group(1).strip()

        start_time = timestamp_to_seconds(raw_start)
        end_time = timestamp_to_seconds(raw_end)

        if end_time <= start_time:
            raise ValueError(
                f"Invalid duration in {file_path.name}, "
                f"Scene {scene_number}"
            )

        scenes.append({
            "scene_number": scene_number,
            "scene_heading": heading.group(1).strip(),
            "start_time": start_time,
            "end_time": end_time,
            "raw_start": raw_start,
            "raw_end": raw_end,
            "events": []
        })

    return scenes


def extract_events(scene_text: str) -> list:
    """
    Extract camera, action, sound effects and dialogue
    from a single scene.
    """

    event_header_pattern = re.compile(
        r"^[ \t]*\*\*(.+?):\*\*[ \t]*$",
        re.MULTILINE
    )

    headers = list(event_header_pattern.finditer(scene_text))

    events = []

    for index, header in enumerate(headers):

        block_start = header.start()

        block_end = (
            headers[index + 1].start()
            if index + 1 < len(headers)
            else len(scene_text)
        )

        block = scene_text[block_start:block_end]

        header_text = header.group(1).strip()

        time_match = re.search(
            r"Start:\s*\(([^)]+)\)\s*\|\s*End:\s*\(([^)]+)\)",
            block
        )

        if not time_match:
            continue

        raw_start = time_match.group(1).strip()
        raw_end = time_match.group(2).strip()

        start_time = timestamp_to_seconds(raw_start)
        end_time = timestamp_to_seconds(raw_end)

        if end_time <= start_time:
            raise ValueError(
                f"Invalid event duration: {header_text}"
            )

        content = block[time_match.end():].strip()

        speaker = None
        delivery = None

        if header_text.upper() == "CAMERA":
            event_type = "CAMERA"

        elif header_text.upper() == "ACTION":
            event_type = "ACTION"

        elif header_text.upper() in ["SOUND FX", "SOUND EFFECT"]:
            event_type = "SOUND_FX"

        else:
            event_type = "DIALOGUE"

            character_match = re.match(
                r"^(.*?)\s*\((.*?)\)\s*$",
                header_text
            )

            if character_match:
                speaker = character_match.group(1).strip()
                delivery = character_match.group(2).strip()
            else:
                speaker = header_text.strip()

        if event_type == "DIALOGUE":
            if content.startswith('"') and content.endswith('"'):
                content = content[1:-1]

        events.append({
            "event_type": event_type,
            "speaker": speaker,
            "delivery": delivery,
            "start_time": start_time,
            "end_time": end_time,
            "raw_start": raw_start,
            "raw_end": raw_end,
            "content": content
        })

    return events




def main():

    # Find the project directory.
    project_root = Path(__file__).resolve().parent.parent

    # Locate screenplay files.
    episodes_folder = project_root / "data" / "episodes"

    try:

        # ----------------------------------------------------
        # FILE DISCOVERY
        # ----------------------------------------------------

        unique_files, duplicates, total = (
            discover_episode_files(episodes_folder)
        )

        print("\n========== KUKU FILE DISCOVERY ==========")

        print(f"Total files discovered : {total}")
        print(f"Unique files            : {len(unique_files)}")
        print(f"Duplicate files         : {total - len(unique_files)}")

        if duplicates:

            print("\nDuplicate groups:")

            for group in duplicates:

                print("\nIdentical files:")

                for file_path in group:
                    print(f"  - {file_path.name}")

        else:
            print("\nNo exact duplicate files found.")

        print("\nFirst five unique episodes:")

        for file_path in unique_files[:5]:
            print(f"  - {file_path.name}")

        print("\nFile discovery completed successfully.")

        # ----------------------------------------------------
        # METADATA EXTRACTION
        # ----------------------------------------------------

        # Select the first file for our initial test.
        sample_file = unique_files[0]

        metadata = extract_episode_metadata(sample_file)

        print("\n========== METADATA EXTRACTION ==========")

        for key, value in metadata.items():
            print(f"\n{key}: {value}")

        print("\nMetadata extraction completed successfully.")

    except (FileNotFoundError, NotADirectoryError) as error:

        print(f"\nERROR: {error}")

    except (OSError, UnicodeError) as error:

        print(f"\nFILE ERROR: {error}")


# ============================================================
# 5. PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
