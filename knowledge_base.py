import clip
import chromadb
import torch
import os
from config import DB_PATH, CLIP_MODEL
from PIL import Image

# Load model globally to avoid reloading in loops
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load(CLIP_MODEL, device=device)

# Initialize Chroma Client globally to prevent re-opening connection in loops
client = chromadb.PersistentClient(path=DB_PATH)
col = client.get_or_create_collection("video_frames_v1")
audio_col = client.get_or_create_collection("audio_segments_v1")

def store_audio_segments(segments, video_filename=None):
    if not segments:
        return

    ids = []
    embeddings = []
    metadatas = []
    documents = []

    for i, seg in enumerate(segments, 1):
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", 0.0))
        duration = max(0.0, end - start)
        text = (seg.get("text") or "").strip()

        formatted = text

        text_token = clip.tokenize([formatted], truncate=True).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text_token)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            vector = text_features.cpu().numpy().tolist()[0]

        vid = (video_filename or "unknown").replace(" ", "_")
        ids.append(f"audio_{vid}_{i:06d}")
        embeddings.append(vector)
        metadatas.append({
            "start": start,
            "end": end,
            "duration": duration,
            "video": video_filename or ""
        })
        documents.append(formatted)

    audio_col.upsert(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas,
        documents=documents
    )

def store_frame_embedding(short_cap, long_cap, causal_text, timestamp, frame_path, scene_start, scene_end, scene_duration, video_filename=None):
    # 1-4. Text Components: Time, Short, Long, Causal
    # (Transcript is kept in metadata but excluded from embedding per request)
    text_content = f"SHORT: {short_cap} | LONG: {long_cap} | CAUSAL: {causal_text}"
    
    # 5. Actual Frame Embedding (Visual)
    image = preprocess(Image.open(frame_path)).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # Compute Text Embedding
        text_token = clip.tokenize([text_content], truncate=True).to(device)
        text_features = model.encode_text(text_token)
        
        # Compute Image Embedding
        image_features = model.encode_image(image)
        
        # Fusion: Normalize and Average both vectors
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        fused_vector = (text_features + image_features) / 2.0
        fused_vector = fused_vector / fused_vector.norm(dim=-1, keepdim=True)
        
        vector = fused_vector.cpu().numpy().tolist()[0]
    
    vid = (video_filename or "unknown").replace(" ", "_")
    col.upsert(
        ids=[f"frame_{vid}_{timestamp}"],
        embeddings=[vector],
        metadatas=[{
            "timestamp": timestamp,
            "scene_start": scene_start,
            "scene_end": scene_end,
            "scene_duration": scene_duration,
            "video": video_filename or "",
        }],
        documents=[text_content]
    )
