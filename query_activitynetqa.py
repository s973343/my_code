import json
import os
import time
from output_json import create_output_json
from retrieval import query_video_rag


def split_prediction_sections(prediction):
    text = str(prediction or "").strip()
    if not text:
        return "", "Evidence: (no evidence)", "[Citations: ]"

    citations_line = ""
    lines = text.splitlines()
    if lines and lines[-1].strip().startswith("[Citations:"):
        citations_line = lines[-1].strip()
        text = "\n".join(lines[:-1]).strip()

    marker = "\n\nEvidence:\n"
    if marker in text:
        answer_text, evidence_body = text.split(marker, 1)
        evidence_text = "Evidence:\n" + evidence_body.strip()
    else:
        answer_text = text
        evidence_text = "Evidence: (no evidence)"

    if not citations_line:
        citations_line = "[Citations: ]"

    return answer_text.strip(), evidence_text, citations_line


def select_project_folder():
    root = os.getcwd()
    folders = sorted(
        name
        for name in os.listdir(root)
        if os.path.isdir(os.path.join(root, name))
        and name not in {"node_modules", "venv", "__pycache__"}
    )
    if not folders:
        raise RuntimeError("No folders found in the project directory.")

    print("Select a folder that contains ActivityNetQA JSON files:")
    for idx, name in enumerate(folders, start=1):
        print(f"{idx}. {name}")

    while True:
        choice = input("Enter number: ").strip()
        if not choice.isdigit():
            print("Please enter a valid number.")
            continue
        index = int(choice)
        if 1 <= index <= len(folders):
            return os.path.join(root, folders[index - 1])
        print("Choice out of range.")


def list_json_files(folder):
    json_files = []
    for root, _, files in os.walk(folder):
        for name in files:
            if not name.lower().endswith(".json"):
                continue
            if name.lower().endswith("_output.json"):
                continue
            json_files.append(os.path.join(root, name))
    return sorted(json_files)


def load_entries(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in ("data", "entries", "items", "examples"):
            val = data.get(key)
            if isinstance(val, list):
                return val

    return []


def normalize_text(value):
    return str(value or "").strip()


def resolve_video_name(entry):
    raw = (
        entry.get("video_name")
        or entry.get("video")
        or entry.get("video_path")
        or entry.get("id")
        or ""
    )
    raw = normalize_text(raw)
    if not raw:
        return ""

    # ActivityNetQA often stores bare ids (no extension).
    if os.path.splitext(raw)[1]:
        return raw
    return f"{raw}.mp4"


def build_queries(folder):
    json_files = list_json_files(folder)
    if not json_files:
        raise RuntimeError("No JSON files found in the selected folder.")

    rows = []
    for json_path in json_files:
        print(f"\nProcessing file: {json_path}")
        json_name = os.path.basename(json_path)
        json_stem = os.path.splitext(json_name)[0]

        try:
            entries = load_entries(json_path)
        except Exception as exc:
            print(f"Skipping '{json_name}': {exc}")
            continue

        if not entries:
            print(f"Skipping '{json_name}': no list entries found.")
            continue

        for idx, entry in enumerate(entries):
            if not isinstance(entry, dict):
                continue

            query_text = normalize_text(entry.get("question") or entry.get("query"))
            answer_text = normalize_text(entry.get("answer"))
            video_value = resolve_video_name(entry)
            question_id = normalize_text(entry.get("question_id"))
            q_type = normalize_text(entry.get("type"))

            if not query_text and not answer_text:
                continue

            rows.append(
                {
                    "id": question_id or f"{json_stem}_{idx:06d}",
                    "source_json": json_name,
                    "entry_index": idx,
                    "video": video_value,
                    "query": query_text,
                    "answer": answer_text,
                    "question_id": question_id,
                    "type": q_type,
                }
            )

    return rows


def main():
    folder = select_project_folder()
    selected_folder_name = os.path.basename(os.path.normpath(folder))
    print(f"\nReading JSON files from: {folder}")
    rows = build_queries(folder)
    if not rows:
        raise RuntimeError("No valid entries were found to build a query file.")

    print("\nRunning retrieval on extracted entries...")
    success = 0
    failed = 0
    skipped = 0
    latencies_sec = []
    output_filename = f"{selected_folder_name}_output.json"

    for i, row in enumerate(rows, start=1):
        user_query = row.get("query", "")
        video_filename = row.get("video", "")
        expected_answer = normalize_text(row.get("answer"))
        latency_sec = 0.0

        if not user_query:
            skipped += 1
            print(f"\n[{i}] Skipped (empty query)")
            output_text = "Skipped: empty query"
        else:
            start_time = time.perf_counter()
            try:
                prediction = query_video_rag(
                    user_query=user_query,
                    video_filename=video_filename,
                    attach_images=False,
                )
                latency_sec = time.perf_counter() - start_time
                latencies_sec.append(latency_sec)
                success += 1

                print()
                print(f"Video: {video_filename}")
                print(f"Query: {user_query}")
                if expected_answer:
                    print(f"Answer: {expected_answer}")

                answer_text, evidence_text, citations_line = split_prediction_sections(prediction)
                print(f"Output: {answer_text}")
                print(f"Latency: {latency_sec:.3f}s")
                print()
                print(evidence_text)
                print()
                print(citations_line)
                output_text = answer_text
            except Exception as exc:
                latency_sec = time.perf_counter() - start_time
                failed += 1
                print(f"Video: {video_filename}")
                print(f"Query: {user_query}")
                print(f"Latency: {latency_sec:.3f}s")
                print(f"Retrieval failed: {exc}")
                output_text = f"Retrieval failed: {exc}"

        create_output_json(
            video=video_filename,
            query=user_query,
            answer=expected_answer,
            output=output_text,
            latency=latency_sec,
            filename=output_filename,
            default_prefix=selected_folder_name,
        )

    avg_latency_sec = (sum(latencies_sec) / len(latencies_sec)) if latencies_sec else 0.0
    max_latency_sec = max(latencies_sec) if latencies_sec else 0.0
    print(
        f"Done. total={len(rows)} success={success} skipped={skipped} failed={failed} "
        f"avg_latency={avg_latency_sec:.3f}s max_latency={max_latency_sec:.3f}s"
    )


if __name__ == "__main__":
    main()
