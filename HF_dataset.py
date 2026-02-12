from datasets import load_dataset
from IPython.display import Video, display
import os

from hf_auth import hf_login


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
    Streams VISTA-400K, searches for a filename, and saves it locally.
    Returns the local path to the video if found, else None.
    """
    dataset_name = resolve_dataset_name(dataset_name)
    if not dataset_name:
        return None
    target_key = target_filename.replace('.mp4', '')
    local_output_path = f"{target_key}.mp4"
    
    print(f"--- Searching for {target_filename} in {data_dir} ---")
    
    try:
        hf_login()
        # Load the stream
        dataset = load_dataset(dataset_name, data_dir=data_dir, streaming=True, split="train")
        
        for i, entry in enumerate(dataset):
            current_key = entry.get('__key__')
            
            if current_key == target_key:
                # Handle different key names for video data (mp4 or video)
                video_bytes = entry.get('mp4') or entry.get('video')
                
                if video_bytes:
                    with open(local_output_path, "wb") as f:
                        f.write(video_bytes)
                    print(f"[OK] Found and saved to: {local_output_path} (Index: {i})")
                    return local_output_path
                else:
                    print(f"[ERROR] Found key {target_key}, but no video bytes present.")
                    return None
            
            if i % 1000 == 0 and i > 0:
                print(f"... checked {i} videos ...")
                
            if i >= max_search:
                break
                
        print(f"[ERROR] Video {target_filename} not found within first {max_search} entries.")
        return None

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# --- Example usage ---
# video_path = get_dataset_video("spatiotemporal_niah_qa", "CPLa78IpO90_19_17to138.mp4")
# if video_path:
#     display(Video(video_path, embed=True, width=600))
