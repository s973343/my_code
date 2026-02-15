import json
import os
import re
import shutil
import socket
import zipfile

import audio_processing
import config
import main as ingest_main
import video_processing
from huggingface_hub import HfApi, hf_hub_download
from hf_auth import hf_login

MVBENCH_DATASET_NAME = "OpenGVLab/MVBench"
_HF_AUTH_READY = False
_REPO_FILES_CACHE = {}
_LOCAL_VIDEO_INDEX = None
_ZIP_INDEX_CACHE = {}
VIDEO_EXTS = (".mp4", ".avi", ".mkv", ".webm", ".mov", ".m4v")
TASK_ARCHIVE_HINTS = {
    "episodic_reasoning": "video/tvqa.zip",
}


def _normalize_task_key(task_name):
    key = str(task_name or "").strip().lower()
    key = re.sub(r"\.json$", "", key)
    key = re.sub(r"\s*\(\d+\)\s*$", "", key)
    key = re.sub(r"[^a-z0-9_]+", "_", key).strip("_")
    return key


def _resolve_task_archive_name(task_name):
    norm = _normalize_task_key(task_name)
    if norm in TASK_ARCHIVE_HINTS:
        return TASK_ARCHIVE_HINTS[norm]
    for task_key, archive in TASK_ARCHIVE_HINTS.items():
        if task_key in norm:
            return archive
    return None


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
    print("No dataset found in config.")
    return None


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


def _normalize_target(target_filename):
    return str(target_filename or "").replace("\\", "/").strip().lstrip("./")


def _find_repo_video_path(repo_files, target_filename, data_dir=None):
    target_norm = _normalize_target(target_filename)
    if not target_norm:
        return None

    target_base = os.path.basename(target_norm)
    target_stem, target_ext = os.path.splitext(target_base)
    target_ext = target_ext.lower()

    dir_prefix = (str(data_dir).replace("\\", "/").strip().strip("/") + "/") if data_dir else None
    basename_matches = []

    for path in repo_files:
        path_norm = str(path).replace("\\", "/").strip()
        if not path_norm:
            continue
        ext = os.path.splitext(path_norm)[1].lower()
        if ext not in VIDEO_EXTS:
            continue
        if dir_prefix and not path_norm.startswith(dir_prefix):
            continue

        path_base = os.path.basename(path_norm)
        path_stem, _ = os.path.splitext(path_base)

        if path_norm == target_norm or path_norm.endswith(f"/{target_norm}"):
            return path_norm

        if target_ext:
            if path_base == target_base:
                basename_matches.append(path_norm)
        else:
            if path_stem == target_stem:
                basename_matches.append(path_norm)

    if not basename_matches:
        return None

    # Prefer conventional video folders if several files share the same basename.
    for path in basename_matches:
        low = f"/{path.lower()}/"
        if "/video/" in low or "/videos/" in low:
            return path

    return basename_matches[0]


def _build_local_video_index(root="."):
    index = {}
    root_abs = os.path.abspath(root)
    for current_root, dirs, files in os.walk(root_abs):
        dirs[:] = [d for d in dirs if d.lower() not in {"venv", ".git", "__pycache__", "node_modules"}]
        for name in files:
            stem, ext = os.path.splitext(name)
            if ext.lower() not in VIDEO_EXTS:
                continue
            path = os.path.join(current_root, name)
            index.setdefault(name.lower(), path)
            index.setdefault(stem.lower(), path)
    return index


def _find_local_video(target_filename):
    global _LOCAL_VIDEO_INDEX
    target_norm = _normalize_target(target_filename)
    if not target_norm:
        return None

    if os.path.isabs(target_norm) and os.path.exists(target_norm):
        return os.path.abspath(target_norm)

    if os.path.exists(target_norm):
        return os.path.abspath(target_norm)

    target_base = os.path.basename(target_norm)
    target_stem, target_ext = os.path.splitext(target_base)

    if target_ext:
        if os.path.exists(target_base):
            return os.path.abspath(target_base)
    else:
        for ext in VIDEO_EXTS:
            direct = f"{target_base}{ext}"
            if os.path.exists(direct):
                return os.path.abspath(direct)

    if _LOCAL_VIDEO_INDEX is None:
        _LOCAL_VIDEO_INDEX = _build_local_video_index(".")

    if target_ext:
        return _LOCAL_VIDEO_INDEX.get(target_base.lower())
    return _LOCAL_VIDEO_INDEX.get(target_stem.lower()) or _LOCAL_VIDEO_INDEX.get(target_base.lower())


def _build_zip_member_index(zip_path):
    if zip_path in _ZIP_INDEX_CACHE:
        return _ZIP_INDEX_CACHE[zip_path]

    member_index = {}
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.namelist():
            if member.endswith("/"):
                continue
            base = os.path.basename(member)
            stem, ext = os.path.splitext(base)
            if ext.lower() not in VIDEO_EXTS:
                continue
            member_index.setdefault(base.lower(), member)
            member_index.setdefault(stem.lower(), member)

    _ZIP_INDEX_CACHE[zip_path] = member_index
    return member_index


def _extract_from_zip(zip_path, member_name):
    cache_root = os.path.join(".cache", "mvbench_videos")
    os.makedirs(cache_root, exist_ok=True)

    target_base = os.path.basename(member_name)
    target_path = os.path.join(cache_root, target_base)
    if os.path.exists(target_path):
        return os.path.abspath(target_path)

    with zipfile.ZipFile(zip_path, "r") as zf:
        with zf.open(member_name) as src, open(target_path, "wb") as dst:
            shutil.copyfileobj(src, dst)
    return os.path.abspath(target_path)


def _resolve_from_archive(task_name, target_filename, dataset_name):
    archive_name = _resolve_task_archive_name(task_name)
    if not archive_name:
        return None

    target_norm = _normalize_target(target_filename)
    if not target_norm:
        return None

    target_base = os.path.basename(target_norm)
    target_stem, target_ext = os.path.splitext(target_base)
    target_ext = target_ext.lower()

    try:
        print(f"Trying archive fallback for task '{task_name}': {archive_name}")
        _ensure_hf_login()
        zip_path = hf_hub_download(
            repo_id=dataset_name,
            repo_type="dataset",
            filename=archive_name,
        )
        member_index = _build_zip_member_index(zip_path)

        member_name = None
        if target_ext:
            member_name = member_index.get(target_base.lower())
        if not member_name:
            member_name = member_index.get(target_stem.lower())
        if not member_name and not target_ext:
            member_name = member_index.get(f"{target_stem}.mp4".lower())

        if not member_name:
            print(
                f"Video {target_filename} not found inside archive {archive_name}. "
                "Please verify your JSON file and task mapping."
            )
            return None

        local_path = _extract_from_zip(zip_path, member_name)
        print(f"Extracted from {archive_name}: {local_path}")
        return local_path
    except Exception as e:
        print(f"An error occurred while resolving archive video {target_filename}: {e}")
        return None


def get_dataset_video(data_dir, target_filename, max_search=5000, dataset_name=None):
    """
    Finds and downloads a target video from a Hugging Face dataset repo.
    Returns the local path to the video if found, else None.
    """
    del max_search
    dataset_name = resolve_dataset_name(dataset_name)
    if not dataset_name:
        return None

    target_norm = _normalize_target(target_filename)
    if not target_norm:
        return None

    target_base = os.path.basename(target_norm)
    stem, ext = os.path.splitext(target_base)
    local_output_path = target_base if ext.lower() in VIDEO_EXTS else f"{stem or target_base}.mp4"

    print(f"--- Searching for {target_filename} in {data_dir} ---")

    try:
        local_match = _find_local_video(target_filename)
        if local_match:
            print(f"Using local video: {local_match}")
            return local_match

        repo_files = _list_repo_files(dataset_name)
        repo_video_path = _find_repo_video_path(repo_files, target_norm, data_dir=data_dir)

        if not repo_video_path and not ext:
            repo_video_path = _find_repo_video_path(repo_files, f"{target_norm}.mp4", data_dir=data_dir)

        if not repo_video_path:
            print(f"Video {target_filename} not found in dataset repo.")
            return None

        _ensure_hf_login()
        downloaded_path = hf_hub_download(
            repo_id=dataset_name,
            repo_type="dataset",
            filename=repo_video_path,
        )

        if os.path.abspath(downloaded_path) != os.path.abspath(local_output_path):
            shutil.copyfile(downloaded_path, local_output_path)
            print(f"Found and saved to: {local_output_path}")
            return local_output_path

        print(f"Found and saved to: {downloaded_path}")
        return downloaded_path

    except socket.gaierror as e:
        print(f"Network/DNS error while reaching Hugging Face: {e}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def select_project_folder(default_folder="MVBench_jsons"):
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


def extract_query_text(entry):
    question = str(entry.get("question") or entry.get("query") or "").strip()
    candidates = entry.get("candidates") or entry.get("options")

    if isinstance(candidates, list) and candidates:
        options = "; ".join(str(x) for x in candidates)
        if question:
            return f"{question} Options: {options}"
        return f"Options: {options}"

    return question


def _normalize_json_stem(name):
    stem = os.path.splitext(os.path.basename(name))[0]
    stem = re.sub(r"^\d+_", "", stem)
    return stem.strip()


def build_data_dir_candidates(entry, json_name):
    candidates = []

    task = str(entry.get("task") or entry.get("task_type") or "").strip()
    if task:
        candidates.append(f"video/{task}")
        candidates.append(f"MVBench/video/{task}")

    json_stem = _normalize_json_stem(json_name)
    if json_stem:
        candidates.append(f"video/{json_stem}")
        candidates.append(f"MVBench/video/{json_stem}")

    candidates.append("video")
    candidates.append("MVBench/video")

    deduped = []
    seen = set()
    for cand in candidates:
        key = cand.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(cand)

    deduped.append(None)
    return deduped


def resolve_mvbench_video(target_filename, data_dir_candidates, cache, task_name):
    raw = str(target_filename or "").strip()
    if not raw:
        return None

    key = os.path.basename(raw)
    if key in cache and os.path.exists(cache[key]):
        return cache[key]

    # MVBench video assets are largely stored in zip archives.
    # If we know the task->archive mapping, prefer archive resolution first.
    archive_path = _resolve_from_archive(task_name, raw, MVBENCH_DATASET_NAME)
    if archive_path:
        cache[key] = archive_path
        return archive_path

    candidates = [raw]
    stem, ext = os.path.splitext(raw)
    if not ext:
        candidates.append(f"{raw}.mp4")
        base = os.path.basename(raw)
        if base != raw:
            candidates.append(f"{base}.mp4")
            candidates.append(base)

    for data_dir in data_dir_candidates:
        for video_name in candidates:
            video_path = get_dataset_video(
                data_dir=data_dir,
                target_filename=video_name,
                dataset_name=MVBENCH_DATASET_NAME,
            )
            if video_path:
                cache[key] = video_path
                return video_path

    return None


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
    folder = select_project_folder(default_folder="MVBench_jsons")
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
            filename = entry.get("video") or entry.get("video_path") or entry.get("video_id")
            if not filename:
                print(f"[{global_index}] Skipping: no video field.")
                global_index += 1
                continue

            data_dir_candidates = build_data_dir_candidates(entry, json_name)
            video_path = resolve_mvbench_video(filename, data_dir_candidates, cache, json_stem)
            if not video_path:
                print(f"[{global_index}] Missing video in HF dataset: {filename}")
                global_index += 1
                continue

            video_stem = os.path.splitext(os.path.basename(video_path))[0]
            frame_dir = os.path.join("data", "frames", "hf_ingest", "mvbench", json_stem, video_stem)
            audio_out = os.path.join("data", "audio", "hf_ingest", f"mvbench_{json_stem}_{video_stem}.mp3")
            os.makedirs(frame_dir, exist_ok=True)
            os.makedirs(os.path.dirname(audio_out), exist_ok=True)

            query_text = extract_query_text(entry)
            set_pipeline_context(video_path, query_text, frame_dir, audio_out)

            print(f"\n[{global_index}] Ingesting {video_stem}")
            ingest_main.run_ingestion_pipeline()
            global_index += 1


if __name__ == "__main__":
    main_cli()
