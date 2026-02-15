import os
import time
from config import VIDEO_INPUT, USER_DESCRIPTION
from audio_processing import extract_and_transcribe
from video_processing import extract_keyframes
from image_captioning import generate_captions
from causal_analysis import get_causal_knowledge
from knowledge_base import store_frame_embedding, store_audio_segments

def run_ingestion_pipeline():
    print("--- Starting Phase 1: Ingestion & Pre-processing ---")
    start_time = time.time()

    # 1. Audio Processing
    audio_result = extract_and_transcribe(return_segments=True)
    transcript = audio_result["text"]
    audio_segments = audio_result.get("segments", [])
    print(f"Transcript generated: {transcript[:50]}...")
    if audio_segments:
        print(f"Generated {len(audio_segments)} audio segments.")
        store_audio_segments(audio_segments, video_filename=os.path.basename(VIDEO_INPUT))

    # 2. Visual Extraction
    frames_info = extract_keyframes() # returns list of dicts with path + timestamps
    print(f"Extracted {len(frames_info)} keyframes.")

    print("\n--- Starting Phase 2: Reasoning & Fusion ---")
    for frame in frames_info:
        frame_path = frame["path"]
        timestamp = os.path.basename(frame_path).split('.')[0]
        scene_times = f"{frame['start_time']:.2f}-{frame['end_time']:.2f}s"
        
        # 3. Multi-Level Captioning (VLM)
        short_cap, long_cap = generate_captions(frame_path)
        
        # 4. Causal Reasoning Module
        causal_text = get_causal_knowledge(USER_DESCRIPTION, f"{short_cap}. {long_cap}")
        
        # 5. Fusion & Vector Storage
        store_frame_embedding(
            short_cap,
            long_cap,
            causal_text,
            timestamp,
            frame_path,
            frame["start_time"],
            frame["end_time"],
            frame["duration"],
            video_filename=os.path.basename(VIDEO_INPUT)
        )
        print(f"Stored frame at {timestamp} (scene {scene_times}, duration {frame['duration']:.2f}s)")

    total_time = time.time() - start_time
    print(f"\nPipeline execution finished in {total_time:.2f} seconds.")

if __name__ == "__main__":
    run_ingestion_pipeline()
