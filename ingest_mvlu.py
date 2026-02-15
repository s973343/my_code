import json
import os
import re
import shutil
import socket

import audio_processing
import config
import main as ingest_main
import video_processing
from huggingface_hub import HfApi, hf_hub_download
from hf_auth import hf_login

MVLU_DATASET_NAME = "MLVU/MVLU"
_HF_AUTH_READY = False
_REPO_FILES_CACHE = {}
_LOCAL_VIDEO_INDEX = None

QUESTION_TYPE_DATA_DIRS = {
    "plotQA": ["MLVU/video/1_plotQA"],
    "findNeedle": ["MLVU/video/2_needle"],
    "ego": ["MLVU/video/3_ego"],
    "count": ["MLVU/video/4_count"],
    "order": ["MLVU/video/5_order"],
    "anomaly_reco": ["MLVU/video/6_anomaly_reco"],
    "topic_reasoning": ["MLVU/video/7_topic_reasoning"],
    "subPlot": ["MLVU/video/8_sub_scene"],
    "summary": ["MLVU/video/9_summary"],
}


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


def _find_repo_video_path(repo_files, target_filename, data_dir=None):
    target_norm = str(target_filename).replace("\\", "/").strip().lstrip("./")
    target_base = os.path.basename(target_norm)
    dir_prefix = (str(data_dir).replace("\\", "/").strip().strip("/") + "/") if data_dir else None
    basename_matches = []

    for path in repo_files:
        path_norm = str(path).replace("\\", "/").strip()
        if not path_norm.lower().endswith(".mp4"):
            continue
        if dir_prefix and not path_norm.startswith(dir_prefix):
            continue

        if path_norm == target_norm or path_norm.endswith(f"/{target_norm}"):
            return path_norm
        if os.path.basename(path_norm) == target_base:
            basename_matches.append(path_norm)

    if not basename_matches:
        return None
    return basename_matches[0]


def _build_local_video_index(root="."):
    index = {}
    root_abs = os.path.abspath(root)
    for current_root, dirs, files in os.walk(root_abs):
        dirs[:] = [d for d in dirs if d.lower() not in {"venv", ".git", "__pycache__"}]
        for name in files:
            if not name.lower().endswith(".mp4"):
                continue
            key = name.lower()
            index.setdefault(key, os.path.join(current_root, name))
    return index


def _find_local_video(target_filename):
    global _LOCAL_VIDEO_INDEX
    target_base = os.path.basename(str(target_filename).strip())
    if not target_base:
        return None

    # Fast-path: file already in current working dir.
    local_output_path = target_base if target_base.lower().endswith(".mp4") else f"{target_base}.mp4"
    if os.path.exists(local_output_path):
        return os.path.abspath(local_output_path)

    if _LOCAL_VIDEO_INDEX is None:
        _LOCAL_VIDEO_INDEX = _build_local_video_index(".")

    return _LOCAL_VIDEO_INDEX.get(target_base.lower())


def get_dataset_video(data_dir, target_filename, max_search=5000, dataset_name=None):
    """
    Finds and downloads a target MP4 from a Hugging Face dataset repo.
    Returns the local path to the video if found, else None.
    """
    del max_search
    dataset_name = resolve_dataset_name(dataset_name)
    if not dataset_name:
        return None
    target_base = os.path.basename(str(target_filename).strip())
    local_output_path = target_base if target_base.lower().endswith(".mp4") else f"{target_base}.mp4"

    print(f"--- Searching for {target_filename} in {data_dir} ---")

    try:
        local_match = _find_local_video(target_filename)
        if local_match:
            print(f"Using local video: {local_match}")
            return local_match

        repo_files = _list_repo_files(dataset_name)
        repo_video_path = _find_repo_video_path(repo_files, target_filename, data_dir=data_dir)
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


def select_project_folder(default_folder="MVLU_jsons"):
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
    question = str(entry.get("question") or "").strip()
    candidates = entry.get("candidates")

    if isinstance(candidates, list) and candidates:
        options = "; ".join(str(x) for x in candidates)
        if question:
            return f"{question} Options: {options}"
        return f"Options: {options}"

    return question


def _normalize_stem_for_dir(name):
    stem = os.path.splitext(os.path.basename(name))[0]
    stem = re.sub(r"^\d+_", "", stem)
    return stem.strip()


def _task_dir_from_json_name(json_name):
    stem = os.path.splitext(os.path.basename(json_name))[0]
    if re.match(r"^\d+_", stem):
        return f"MLVU/video/{stem}"
    return None


def build_data_dir_candidates(entry, json_name):
    candidates = []

    question_type = str(entry.get("question_type") or "").strip()
    if question_type:
        candidates.extend(QUESTION_TYPE_DATA_DIRS.get(question_type, []))

    task_dir = _task_dir_from_json_name(json_name)
    if task_dir:
        candidates.append(task_dir)

    json_stem = _normalize_stem_for_dir(json_name)
    if json_stem:
        candidates.append(f"MLVU/video/{json_stem}")

    de_duplicated = []
    seen = set()
    for cand in candidates:
        if not cand:
            continue
        key = cand.lower()
        if key in seen:
            continue
        seen.add(key)
        de_duplicated.append(cand)

    # Fallback: global basename match in the whole dataset.
    de_duplicated.append(None)
    return de_duplicated


def resolve_mvlu_video(target_filename, data_dir_candidates, cache):
    key = os.path.basename(str(target_filename).strip())
    if key in cache and os.path.exists(cache[key]):
        return cache[key]

    for data_dir in data_dir_candidates:
        video_path = get_dataset_video(
            data_dir=data_dir,
            target_filename=target_filename,
            dataset_name=MVLU_DATASET_NAME,
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
    folder = select_project_folder(default_folder="MVLU_jsons")
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
            filename = entry.get("video") or entry.get("video_path")
            if not filename:
                print(f"[{global_index}] Skipping: no video field.")
                global_index += 1
                continue

            data_dir_candidates = build_data_dir_candidates(entry, json_name)
            video_path = resolve_mvlu_video(filename, data_dir_candidates, cache)
            if not video_path:
                print(f"[{global_index}] Missing video in HF dataset: {filename}")
                global_index += 1
                continue

            video_stem = os.path.splitext(os.path.basename(video_path))[0]
            frame_dir = os.path.join("data", "frames", "hf_ingest", "mvlu", json_stem, video_stem)
            audio_out = os.path.join("data", "audio", "hf_ingest", f"mvlu_{json_stem}_{video_stem}.mp3")
            os.makedirs(frame_dir, exist_ok=True)
            os.makedirs(os.path.dirname(audio_out), exist_ok=True)

            query_text = extract_query_text(entry)
            set_pipeline_context(video_path, query_text, frame_dir, audio_out)

            print(f"\n[{global_index}] Ingesting {video_stem}")
            ingest_main.run_ingestion_pipeline()
            global_index += 1


if __name__ == "__main__":
    main_cli()
