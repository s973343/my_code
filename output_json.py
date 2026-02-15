
import json
import os
from datetime import datetime


def append_json_entry(entry, output_dir=None, filename=None, default_prefix="output"):
    if not isinstance(entry, dict):
        raise TypeError("entry must be a dictionary.")

    target_dir = output_dir or os.path.join(os.getcwd(), "output")
    os.makedirs(target_dir, exist_ok=True)

    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{default_prefix}_{timestamp}.json"

    if not filename.lower().endswith(".json"):
        filename = f"{filename}.json"

    output_path = os.path.join(target_dir, filename)
    data = []
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            try:
                existing = json.load(f)
            except json.JSONDecodeError:
                existing = []

        if isinstance(existing, list):
            data = existing
        elif isinstance(existing, dict):
            # Backward compatibility for files created with a single JSON object.
            data = [existing]

    data.append(entry)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return output_path


def create_output_json(
    video,
    query,
    answer,
    output,
    latency=None,
    output_dir=None,
    filename=None,
    default_prefix="output",
):
    # Backward-compatible wrapper for existing VISTA400K call sites.
    entry = {
        "Video": video,
        "Query": query,
        "Answer": answer,
        "Output": output,
    }
    if latency is not None:
        entry["LatencySec"] = round(float(latency), 3)
    return append_json_entry(
        entry=entry,
        output_dir=output_dir,
        filename=filename,
        default_prefix=default_prefix,
    )
