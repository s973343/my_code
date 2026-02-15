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

    print("Select a folder that contains VISTA JSON files:")
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
            if name.startswith("vista400k_queries_"):
                continue
            json_files.append(os.path.join(root, name))
    return sorted(json_files)


def load_entries(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in ("entries", "data", "items"):
            if isinstance(data.get(key), list):
                return data[key]

    return []


def extract_human_and_answer(entry):
    human_text = ""
    answer_text = ""

    conversations = entry.get("conversations")
    if not isinstance(conversations, list):
        return human_text, answer_text

    for turn in conversations:
        if not isinstance(turn, dict):
            continue
        role = str(turn.get("from", "")).strip().lower()
        value = str(turn.get("value", "")).strip()
        if not value:
            continue
        if role == "human" and not human_text:
            human_text = value
        elif role in {"gpt", "assistant"} and not answer_text:
            answer_text = value

    return human_text, answer_text


def extract_query_and_answer(entry):
    human_text, answer_text = extract_human_and_answer(entry)
    query_text = extract_final_question(human_text)

    if not query_text:
        for key in ("question", "query", "prompt", "instruction", "caption", "description"):
            value = entry.get(key)
            if value:
                query_text = str(value).strip()
                if query_text:
                    break

    if not answer_text:
        for key in ("answer", "gt_answer", "label", "target", "output", "response"):
            value = entry.get(key)
            if value:
                answer_text = str(value).strip()
                if answer_text:
                    break

    return query_text, human_text, answer_text


def extract_final_question(human_text):
    if not human_text:
        return ""
    lines = [line.strip() for line in human_text.splitlines() if line.strip()]
    if not lines:
        return ""

    for line in reversed(lines):
        if "?" in line:
            return line
    return lines[-1]


def extract_video_id(video_value):
    if not video_value:
        return ""
    filename = os.path.basename(str(video_value).strip())
    return os.path.splitext(filename)[0]


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

            query_text, human_text, answer_text = extract_query_and_answer(entry)
            if not query_text and not answer_text:
                continue

            video_value = (
                entry.get("image")
                or entry.get("video")
                or entry.get("video_path")
                or entry.get("video_id")
                or entry.get("clip_path")
                or entry.get("path")
                or ""
            )
            row = {
                "id": f"{json_stem}_{idx:06d}",
                "source_json": json_name,
                "entry_index": idx,
                "video": video_value,
                "video_id": extract_video_id(video_value),
                "query": query_text,
                "human_prompt": human_text,
                "answer": answer_text,
            }
            rows.append(row)

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
        user_query = row.get("query", "") or row.get("human_prompt", "")
        video_filename = row.get("video", "")
        expected_answer = str(row.get("answer", "")).strip()
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
