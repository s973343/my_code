import cv2
import os
from config import VIDEO_INPUT, FRAME_DIR

def extract_keyframes():
    print("-> Extracting Keyframes (scene-change based, until video end)...")
    if not os.path.exists(FRAME_DIR): os.makedirs(FRAME_DIR)

    cap = cv2.VideoCapture(VIDEO_INPUT)
    frames_info = []
    count = 0
    frame_idx = 0

    # Scene-change detection settings
    threshold = 0.6  # Bhattacharyya distance threshold (0..1)
    min_scene_gap = 10  # minimum frames between keyframes
    last_scene_idx = -min_scene_gap
    prev_hist = None
    prev_scene_start_idx = 0

    # Video FPS for timestamp conversion
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 1.0

    last_valid_frame = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        last_valid_frame = frame

        # Compute grayscale histogram for scene-change detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        cv2.normalize(hist, hist)

        if prev_hist is None:
            # Initialize first scene start
            prev_hist = hist
            prev_scene_start_idx = frame_idx
            last_scene_idx = frame_idx
        else:
            diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
            if diff > threshold and (frame_idx - last_scene_idx) >= min_scene_gap:
                # Save keyframe for the previous scene at this boundary frame
                path = os.path.join(FRAME_DIR, f"frame_{count:03d}.jpg")
                cv2.imwrite(path, frame)

                start_time = prev_scene_start_idx / fps
                end_time = frame_idx / fps
                duration = max(0.0, end_time - start_time)

                frames_info.append({
                    "path": path,
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration": duration
                })

                prev_hist = hist
                prev_scene_start_idx = frame_idx
                last_scene_idx = frame_idx
                count += 1

        frame_idx += 1

    # Final scene (if any frames were read)
    if frame_idx > 0 and last_valid_frame is not None:
        path = os.path.join(FRAME_DIR, f"frame_{count:03d}.jpg")
        # Use the last valid frame for the final keyframe
        cv2.imwrite(path, last_valid_frame)

        start_time = prev_scene_start_idx / fps
        end_time = max(frame_idx - 1, prev_scene_start_idx) / fps
        duration = max(0.0, end_time - start_time)

        frames_info.append({
            "path": path,
            "start_time": start_time,
            "end_time": end_time,
            "duration": duration
        })

    cap.release()
    return frames_info
