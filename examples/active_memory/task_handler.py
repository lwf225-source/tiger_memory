"""Pure data renderer: no storage, retrieval, files, providers or seed imports."""
import json


def render_task(task, context):
    packet = json.loads(context) if context else {"entries": []}
    entries = packet["entries"]
    style = "default"
    labels = []
    for entry in entries:
        labels.append({"label": entry["topic_key"], "entry_id": entry["entry_id"],
                       "project_id": entry["project_id"], "source_ref": entry["source_ref"]})
        if entry["topic_key"] == "demo.output_style":
            data = json.loads(entry["statement"])
            if data.get("style") in {"compact", "expanded"}:
                style = data["style"]
    checklist = ["check alpha", "check beta", "check gamma"]
    rendered = " | ".join(checklist) if style == "compact" else "\n".join("- " + item for item in checklist)
    return {"task": task, "style": style, "checklist": rendered, "labels": labels}
