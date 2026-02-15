import os
import clip
import torch
import chromadb
import base64
import torch.nn.functional as F
from config import DB_PATH, CLIP_MODEL, VLM_MODEL, GROQ_API_KEY, FRAME_DIR
from groq import Groq

client_db = chromadb.PersistentClient(path=DB_PATH)
frame_collection = client_db.get_collection("video_frames_v1")
audio_collection = client_db.get_collection("audio_segments_v1")
groq_client = Groq(api_key=GROQ_API_KEY)
model, _ = clip.load(CLIP_MODEL, device="cpu")

def query_video_rag(user_query, debug_raw=False, video_filename=None, frame_dir=None, attach_images=True):
    # 1. CLIP Embed Query
    text_token = clip.tokenize([user_query]).to("cpu")
    with torch.no_grad():
        text_features = model.encode_text(text_token)
        text_features /= text_features.norm(dim=-1, keepdim=True)  # Normalize to match stored vectors
        query_vector = text_features.numpy().tolist()[0]

    # 2. Vector Search (frames + audio)
    where_filter = {"video": video_filename} if video_filename else None
    frame_results = frame_collection.query(
        query_embeddings=[query_vector],
        n_results=5,
        include=["metadatas", "documents"],
        where=where_filter
    )
    audio_results = audio_collection.query(
        query_embeddings=[query_vector],
        n_results=5,
        include=["metadatas", "documents"],
        where=where_filter
    )

    if debug_raw:
        print("\n[RAW] Frame Results:")
        print(frame_results)
        print("\n[RAW] Audio Results:")
        print(audio_results)

    frame_metas = frame_results.get("metadatas", [[]])[0]
    frame_docs = frame_results.get("documents", [[]])[0]
    frame_ids = frame_results.get("ids", [[]])[0]
    audio_metas = audio_results.get("metadatas", [[]])[0]
    audio_docs = audio_results.get("documents", [[]])[0]
    audio_ids = audio_results.get("ids", [[]])[0]

    if not frame_metas and not audio_metas:
        return "No matches."

    # Build text list for reranking frames (use documents)
    def _cap(text, limit=400):
        text = (text or "").strip()
        return text[:limit]

    frame_texts = [_cap(d, 800) for d in frame_docs]

    # Encode metadata texts with CLIP for reranking
    ranked_indices = list(range(len(frame_metas)))
    if frame_texts:
        text_tokens = clip.tokenize(frame_texts, truncate=True).to("cpu")
        with torch.no_grad():
            meta_features = model.encode_text(text_tokens)
            meta_features /= meta_features.norm(dim=-1, keepdim=True)

        # Cosine similarity between query and metadata
        sims = F.cosine_similarity(meta_features, text_features)
        ranked_indices = torch.argsort(sims, descending=True).tolist()

    # Reorder frames by similarity
    frame_metas = [frame_metas[i] for i in ranked_indices]
    frame_docs = [frame_docs[i] for i in ranked_indices]
    frame_ids = [frame_ids[i] for i in ranked_indices] if frame_ids else []

    top_k = 3
    frame_context_lines = []
    for i, (m, d, fid) in enumerate(zip(frame_metas[:top_k], frame_docs[:top_k], frame_ids[:top_k]), start=1):
        scene_start = m.get("scene_start")
        scene_end = m.get("scene_end")
        scene_range = ""
        if scene_start is not None and scene_end is not None:
            scene_range = f" ({scene_start:.2f}-{scene_end:.2f}s)"
        frame_context_lines.append(
            f"[F{i}] id={fid} ts={m.get('timestamp','')}{scene_range} | {d}"
        )

    audio_context_lines = []
    for i, (m, d, aid) in enumerate(zip(audio_metas[:top_k], audio_docs[:top_k], audio_ids[:top_k]), start=1):
        audio_context_lines.append(
            f"[A{i}] id={aid} {d} (t={m.get('start',0.0):.2f}-{m.get('end',0.0):.2f}s, dur={m.get('duration',0.0):.2f}s)"
        )

    # 3. Final VLM Reasoning
    # Attach top-k images (optional)
    image_contents = []
    evidence_lines = []
    if attach_images:
        use_frame_dir = frame_dir or FRAME_DIR
        for m in frame_metas[:top_k]:
            ts_i = m.get("timestamp")
            if not ts_i:
                continue
            frame_path = os.path.join(use_frame_dir, f"{ts_i}.jpg")
            if os.path.exists(frame_path):
                with open(frame_path, "rb") as f:
                    img_b64 = base64.b64encode(f.read()).decode('utf-8')
                image_contents.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}})
            scene_start = m.get("scene_start")
            scene_end = m.get("scene_end")
            scene_range = ""
            if scene_start is not None and scene_end is not None:
                scene_range = f"{scene_start:.2f}-{scene_end:.2f}s"
            evidence_lines.append(f"- frame {ts_i} {scene_range}".strip())
    else:
        for m in frame_metas[:top_k]:
            ts_i = m.get("timestamp")
            if not ts_i:
                continue
            scene_start = m.get("scene_start")
            scene_end = m.get("scene_end")
            scene_range = ""
            if scene_start is not None and scene_end is not None:
                scene_range = f"{scene_start:.2f}-{scene_end:.2f}s"
            evidence_lines.append(f"- frame {ts_i} {scene_range}".strip())

    for i, (m, aid) in enumerate(zip(audio_metas[:top_k], audio_ids[:top_k]), start=1):
        evidence_lines.append(
            f"- audio {aid} {m.get('start',0.0):.2f}-{m.get('end',0.0):.2f}s"
        )

    prompt = (
        "You are a video RAG assistant. Answer the question using the provided context and image. "
        "If the answer is not in the context, say you don't know. "
        "Do not repeat yourself. Do not use LaTeX or boxed answers. "
        "Be concise and direct.\n\n"
        f"Question: {user_query}\n"
        "Frame Context:\n" + "\n".join(frame_context_lines) + "\n"
        "Audio Context:\n" + "\n".join(audio_context_lines) + "\n"
    )
    
    response = groq_client.chat.completions.create(
        model=VLM_MODEL,
        messages=[{
            "role": "user",
            "content": [{"type": "text", "text": prompt}] + image_contents
        }]
    )
    evidence_text = "Evidence:\n" + "\n".join(evidence_lines) if evidence_lines else "Evidence: (no frames found)"
    frame_citations = [m.get("timestamp","") for m in frame_metas[:top_k] if m.get("timestamp")]
    audio_citations = [aid for aid in audio_ids[:top_k] if aid]
    citations = ", ".join(frame_citations + audio_citations)
    return f"{response.choices[0].message.content}\n\n{evidence_text}\n\n[Citations: {citations}]"

if __name__ == "__main__":
    q = input("Question: ")
    debug_raw = os.getenv("RAG_DEBUG", "0") == "1"
    print(query_video_rag(q, debug_raw=debug_raw))
