import json
import os

import audio_processing
import config
import main as ingest_main
import video_processing
from datasets import load_dataset
from hf_auth import hf_login

try:
    from IPython.display import Video, display
except Exception:
    Video = None
    display = None


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


def get_dataset_video(data_dir, target_filename, max_search=5000, dataset_name=None):
    """
    Streams a dataset split, searches for a filename, and saves it locally.
    Returns the local path to the video if found, else None.
    """
    dataset_name = resolve_dataset_name(dataset_name)
    if not dataset_name:
        return None
    target_key = target_filename.replace(".mp4", "")
    local_output_path = f"{target_key}.mp4"

    print(f"--- Searching for {target_filename} in {data_dir} ---")

    try:
        hf_login()
        dataset = load_dataset(dataset_name, data_dir=data_dir, streaming=True, split="train")

        for i, entry in enumerate(dataset):
            current_key = entry.get("__key__")

            if current_key == target_key:
                video_bytes = entry.get("mp4") or entry.get("video")

                if video_bytes:
                    with open(local_output_path, "wb") as f:
                        f.write(video_bytes)
                    print(f"Found and saved to: {local_output_path} (Index: {i})")
                    return local_output_path
                print(f"Found key {target_key}, but no video bytes present.")
                return None

            if i % 1000 == 0 and i > 0:
                print(f"... checked {i} videos ...")

            if i >= max_search:
                break

        print(f"Video {target_filename} not found within first {max_search} entries.")
        return None

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


def load_json_entries(json_path):
    entries = []
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            entries.extend(data)
        else:
            print("Skipping: JSON root is not a list.")
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
            video_path = get_dataset_video(json_stem, filename)
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
