import json
import os
import shutil

import audio_processing
import config
import main as ingest_main
import video_processing
from huggingface_hub import HfApi, hf_hub_download
from hf_auth import hf_login

MOVIECHAT_DATASET_NAME = "Enxin/MovieChat-1K_train"
_HF_AUTH_READY = False
_REPO_FILES_CACHE = {}


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


def _find_repo_video_path(repo_files, target_filename):
    target_norm = str(target_filename).replace("\\", "/").strip().lstrip("./")
    target_base = os.path.basename(target_norm)
    basename_matches = []

    for path in repo_files:
        path_norm = str(path).replace("\\", "/").strip()
        if not path_norm.lower().endswith(".mp4"):
            continue
        if path_norm == target_norm or path_norm.endswith(f"/{target_norm}"):
            return path_norm
        if os.path.basename(path_norm) == target_base:
            basename_matches.append(path_norm)

    if not basename_matches:
        return None

    for path in basename_matches:
        if "/videos/" in f"/{path.lower()}/":
            return path
    return basename_matches[0]


def get_dataset_video(data_dir, target_filename, max_search=5000, dataset_name=None):
    """
    Locates and downloads a target MP4 from a Hugging Face dataset repo.
    Returns the local path to the video if found, else None.
    """
    del data_dir, max_search
    dataset_name = resolve_dataset_name(dataset_name)
    if not dataset_name:
        return None
    target_base = os.path.basename(str(target_filename).strip())
    local_output_path = target_base if target_base.lower().endswith(".mp4") else f"{target_base}.mp4"

    print(f"--- Searching for {target_filename} in {dataset_name} ---")

    try:
        repo_files = _list_repo_files(dataset_name)
        repo_video_path = _find_repo_video_path(repo_files, target_filename)
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

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def select_project_folder():
    root = os.getcwd()
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


def _normalize_moviechat_dict(data):
    info = data.get("info") if isinstance(data.get("info"), dict) else {}
    global_qa = data.get("global") if isinstance(data.get("global"), list) else []
    if not info or not global_qa:
        return []

    video_path = str(info.get("video_path") or "").strip()
    if not video_path:
        return []

    entries = []
    for qa in global_qa:
        if not isinstance(qa, dict):
            continue
        question = str(qa.get("question") or "").strip()
        if not question:
            continue
        entries.append({"video": video_path, "question": question})
    return entries


def load_json_entries(json_path):
    entries = []
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            entries.extend(data)
        elif isinstance(data, dict):
            entries.extend(_normalize_moviechat_dict(data))
        else:
            print("Skipping: JSON root is not a list/dict.")
    except Exception as exc:
        print(f"Skipping: {exc}")
    return entries


def extract_query_text(entry):
    conversations = entry.get("conversations") or []
    if isinstance(conversations, list):
        for item in conversations:
            if not isinstance(item, dict):
                continue
            if item.get("from") == "human" and item.get("value"):
                return str(item["value"]).strip()

    if entry.get("question"):
        return str(entry["question"]).strip()
    return ""


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
    folder = select_project_folder()
    json_files = list_json_files(folder)
    if not json_files:
        raise RuntimeError("No JSON files found in the selected folder.")

    global_index = 1
    for json_name in json_files:
        json_path = os.path.join(folder, json_name)
        entries = load_json_entries(json_path)
        if not entries:
            continue
        json_stem = os.path.splitext(os.path.basename(json_name))[0]
        for entry in entries:
            filename = entry.get("image") or entry.get("video") or entry.get("video_path")
            if not filename:
                print(f"[{global_index}] Skipping: no image/video field.")
                global_index += 1
                continue

            video_path = get_dataset_video(
                data_dir=None,
                target_filename=filename,
                dataset_name=MOVIECHAT_DATASET_NAME,
            )
            if not video_path:
                global_index += 1
                continue

            video_stem = os.path.splitext(os.path.basename(video_path))[0]
            frame_dir = os.path.join("data", "frames", "hf_ingest", json_stem, video_stem)
            audio_out = os.path.join("data", "audio", "hf_ingest", f"{json_stem}_{video_stem}.mp3")
            os.makedirs(frame_dir, exist_ok=True)
            os.makedirs(os.path.dirname(audio_out), exist_ok=True)

            query_text = extract_query_text(entry)
            set_pipeline_context(video_path, query_text, frame_dir, audio_out)
            print(f"\n[{global_index}] Ingesting {video_stem}")
            ingest_main.run_ingestion_pipeline()
            global_index += 1


if __name__ == "__main__":
    main_cli()
