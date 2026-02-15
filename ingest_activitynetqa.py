import json
import os
import shutil
import socket

import audio_processing
import config
import main as ingest_main
import video_processing
from huggingface_hub import HfApi, hf_hub_download
from hf_auth import hf_login

ACTIVITYNETQA_DATASET_NAME = "lmms-lab/ActivityNetQA"
_HF_AUTH_READY = False
_REPO_FILES_CACHE = {}
_LOCAL_VIDEO_INDEX = None


def resolve_dataset_name(dataset_name=None):
    if dataset_name:
        return dataset_name
    try:
        import config as app_config

        configured_name = getattr(app_config, "DATASET_NAME", None)
        if configured_name:
            return configured_name
    except Exception:
        pass
    return ACTIVITYNETQA_DATASET_NAME


def _ensure_hf_login():
    global _HF_AUTH_READY
    if _HF_AUTH_READY:
        return
    hf_login()
    _HF_AUTH_READY = True


def _list_repo_files(dataset_name):
    if dataset_name in _REPO_FILES_CACHE:
        return _REPO_FILES_CACHE[dataset_name]
    _ensure_hf_login()
    api = HfApi()
    files = api.list_repo_files(repo_id=dataset_name, repo_type="dataset")
    _REPO_FILES_CACHE[dataset_name] = files
    return files


def _build_local_video_index(root="."):
    index = {}
    root_abs = os.path.abspath(root)
    for current_root, dirs, files in os.walk(root_abs):
        dirs[:] = [d for d in dirs if d.lower() not in {"venv", ".git", "__pycache__"}]
        for name in files:
            if not name.lower().endswith(".mp4"):
                continue
            index.setdefault(name.lower(), os.path.join(current_root, name))
    return index


def _find_local_video(candidate_names):
    global _LOCAL_VIDEO_INDEX
    for name in candidate_names:
        if not name:
            continue
        local_name = name if name.lower().endswith(".mp4") else f"{name}.mp4"
        if os.path.exists(local_name):
            return os.path.abspath(local_name)

    if _LOCAL_VIDEO_INDEX is None:
        _LOCAL_VIDEO_INDEX = _build_local_video_index(".")

    for name in candidate_names:
        if not name:
            continue
        local_name = name if name.lower().endswith(".mp4") else f"{name}.mp4"
        found = _LOCAL_VIDEO_INDEX.get(os.path.basename(local_name).lower())
        if found:
            return found
    return None


def _find_repo_video_path(repo_files, candidate_names, data_dir=None):
    prefixes = []
    if data_dir:
        norm = str(data_dir).replace("\\", "/").strip().strip("/")
        if norm:
            prefixes.append(norm + "/")

    candidate_basenames = []
    for name in candidate_names:
        if not name:
            continue
        n = str(name).replace("\\", "/").strip().lstrip("./")
        if not n.lower().endswith(".mp4"):
            n = f"{n}.mp4"
        candidate_basenames.append(os.path.basename(n))

    for path in repo_files:
        p = str(path).replace("\\", "/").strip()
        if not p.lower().endswith(".mp4"):
            continue
        if prefixes and not any(p.startswith(prefix) for prefix in prefixes):
            continue
        base = os.path.basename(p)
        if base in candidate_basenames:
            return p
    return None


def get_dataset_video(data_dir, target_filenames, dataset_name=None):
    dataset_name = resolve_dataset_name(dataset_name)
    if not dataset_name:
        return None

    candidate_names = [x for x in (target_filenames or []) if x]
    print(f"--- Searching for {candidate_names} in {data_dir} ---")

    local_match = _find_local_video(candidate_names)
    if local_match:
        print(f"Using local video: {local_match}")
        return local_match

    try:
        repo_files = _list_repo_files(dataset_name)
        repo_video_path = _find_repo_video_path(repo_files, candidate_names, data_dir=data_dir)
        if not repo_video_path:
            print(f"Video not found in dataset repo for candidates: {candidate_names}")
            return None

        _ensure_hf_login()
        downloaded_path = hf_hub_download(
            repo_id=dataset_name,
            repo_type="dataset",
            filename=repo_video_path,
        )

        local_output_path = os.path.basename(repo_video_path)
        if os.path.abspath(downloaded_path) != os.path.abspath(local_output_path):
            shutil.copyfile(downloaded_path, local_output_path)
            print(f"Found and saved to: {local_output_path}")
            return os.path.abspath(local_output_path)

        print(f"Found and saved to: {downloaded_path}")
        return downloaded_path
    except socket.gaierror as e:
        print(f"Network/DNS error while reaching Hugging Face: {e}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def select_project_folder(default_folder="ActivityNetQA_data"):
    root = os.getcwd()
    if default_folder and os.path.isdir(os.path.join(root, default_folder)):
        use_default = input(f"Use default folder '{default_folder}'? [Y/n]: ").strip().lower()
        if use_default in ("", "y", "yes"):
            return default_folder

    folders = sorted(
        name
        for name in os.listdir(root)
        if os.path.isdir(os.path.join(root, name))
    )
    if not folders:
        raise RuntimeError("No folders found in the project directory.")

    print("Select a folder:")
    for idx, name in enumerate(folders, start=1):
        print(f"{idx}. {name}")

    while True:
        choice = input("Enter number: ").strip()
        if not choice.isdigit():
            print("Please enter a valid number.")
            continue
        index = int(choice)
        if 1 <= index <= len(folders):
            return folders[index - 1]
        print("Choice out of range.")


def list_json_files(folder):
    return sorted(
        name
        for name in os.listdir(folder)
        if name.lower().endswith(".json") and os.path.isfile(os.path.join(folder, name))
    )


def load_json_entries(json_path):
    entries = []
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            entries.extend(item for item in data if isinstance(item, dict))
        elif isinstance(data, dict):
            for key in ("data", "annotations", "items", "examples"):
                value = data.get(key)
                if isinstance(value, list):
                    entries.extend(item for item in value if isinstance(item, dict))
                    break
            else:
                print("Skipping: JSON dict does not contain a supported list key.")
        else:
            print("Skipping: JSON root is not a list/dict.")
    except Exception as exc:
        print(f"Skipping: {exc}")

    return entries


def _video_id_from_entry(entry):
    for key in ("video_name", "video", "video_id", "youtube_id", "video_path"):
        value = str(entry.get(key) or "").strip()
        if not value:
            continue
        base = os.path.basename(value.replace("\\", "/"))
        stem, ext = os.path.splitext(base)
        if ext.lower() == ".mp4":
            return stem[2:] if stem.startswith("v_") else stem
        return base[2:] if base.startswith("v_") else base
    return ""


def _candidate_video_names(entry):
    video_id = _video_id_from_entry(entry)
    candidates = []
    if video_id:
        candidates.append(f"{video_id}.mp4")
        candidates.append(f"v_{video_id}.mp4")
    raw_path = str(entry.get("video_path") or entry.get("video") or "").strip()
    if raw_path:
        candidates.append(os.path.basename(raw_path.replace("\\", "/")))
    return list(dict.fromkeys(candidates))


def _canonical_video_filename(entry, fallback_path):
    video_id = _video_id_from_entry(entry)
    if video_id:
        return f"{video_id}.mp4"

    base = os.path.basename(str(fallback_path or "").replace("\\", "/"))
    stem, ext = os.path.splitext(base)
    ext = ext or ".mp4"
    if stem.startswith("v_"):
        stem = stem[2:]
    return f"{stem}{ext}" if stem else base


def _ensure_canonical_video_alias(source_video_path, canonical_filename):
    if not source_video_path:
        return source_video_path

    source_abs = os.path.abspath(source_video_path)
    source_base = os.path.basename(source_abs)
    if canonical_filename and source_base == canonical_filename:
        return source_abs

    alias_dir = os.path.join("data", "video_aliases", "activitynetqa")
    os.makedirs(alias_dir, exist_ok=True)
    alias_path = os.path.abspath(os.path.join(alias_dir, canonical_filename))
    if not os.path.exists(alias_path):
        shutil.copyfile(source_abs, alias_path)
    return alias_path


def extract_query_text(entry):
    question = str(entry.get("question") or "").strip()
    answer = str(entry.get("answer") or "").strip()
    qtype = str(entry.get("type") or "").strip()

    parts = []
    if question:
        parts.append(question)
    if answer:
        parts.append(f"Ground truth answer: {answer}")
    if qtype:
        parts.append(f"Question type: {qtype}")
    return "\n".join(parts).strip()


def set_pipeline_context(video_path, query_text, frame_dir, audio_out):
    config.VIDEO_INPUT = video_path
    config.USER_DESCRIPTION = query_text or config.USER_DESCRIPTION
    config.FRAME_DIR = frame_dir
    config.AUDIO_OUTPUT = audio_out

    ingest_main.VIDEO_INPUT = video_path
    ingest_main.USER_DESCRIPTION = query_text or ingest_main.USER_DESCRIPTION
    audio_processing.VIDEO_INPUT = video_path
    audio_processing.AUDIO_OUTPUT = audio_out
    video_processing.VIDEO_INPUT = video_path
    video_processing.FRAME_DIR = frame_dir


def main_cli():
    folder = select_project_folder(default_folder="ActivityNetQA_data")
    json_files = list_json_files(folder)
    if not json_files:
        raise RuntimeError("No JSON files found in the selected folder.")

    global_index = 1
    cache = {}

    for json_name in json_files:
        json_path = os.path.join(folder, json_name)
        entries = load_json_entries(json_path)
        if not entries:
            continue

        json_stem = os.path.splitext(os.path.basename(json_name))[0]
        for entry in entries:
            candidate_names = _candidate_video_names(entry)
            if not candidate_names:
                print(f"[{global_index}] Skipping: no video_name/video path field.")
                global_index += 1
                continue

            cache_key = candidate_names[0]
            if cache_key in cache and os.path.exists(cache[cache_key]):
                video_path = cache[cache_key]
            else:
                video_path = None
                for data_dir in ("videos", "activitynet_videos", "video", None):
                    video_path = get_dataset_video(
                        data_dir=data_dir,
                        target_filenames=candidate_names,
                        dataset_name=ACTIVITYNETQA_DATASET_NAME,
                    )
                    if video_path:
                        cache[cache_key] = video_path
                        break

            if not video_path:
                print(f"[{global_index}] Missing video for: {candidate_names}")
                global_index += 1
                continue

            canonical_filename = _canonical_video_filename(entry, video_path)
            video_path = _ensure_canonical_video_alias(video_path, canonical_filename)

            question_id = str(entry.get("question_id") or "").strip()
            entry_id = question_id or os.path.splitext(os.path.basename(video_path))[0]
            frame_dir = os.path.join(
                "data",
                "frames",
                "hf_ingest",
                "activitynetqa",
                json_stem,
                entry_id,
            )
            audio_out = os.path.join(
                "data",
                "audio",
                "hf_ingest",
                f"activitynetqa_{entry_id}.mp3",
            )
            os.makedirs(frame_dir, exist_ok=True)
            os.makedirs(os.path.dirname(audio_out), exist_ok=True)

            query_text = extract_query_text(entry)
            set_pipeline_context(video_path, query_text, frame_dir, audio_out)

            print(f"\n[{global_index}] Ingesting {entry_id}")
            ingest_main.run_ingestion_pipeline()
            global_index += 1


if __name__ == "__main__":
    main_cli()
