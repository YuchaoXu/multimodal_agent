from __future__ import annotations
import copy
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from .common import read_json, write_json


class Recorder:
    def __init__(self, root, corpus, config, question, parent=None):
        self.run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:10]
        self.path = Path(root).resolve() / self.run_id
        self.path.mkdir(parents=True, exist_ok=False)
        self.sequence = 0
        write_json(self.path / "manifest.json", {"schema_version": 1, "run_id": self.run_id,
                   "corpus": os.path.relpath(corpus.root, self.path),
                   "corpus_fingerprint": corpus.manifest["fingerprint"], "question": question,
                   "config": config.public(), "parent": parent,
                   "synthetic_demo": config.provider == "demo", "created_at": datetime.now(timezone.utc).isoformat()})

    def event(self, kind, payload):
        self.sequence += 1
        event = {"seq": self.sequence, "timestamp": datetime.now(timezone.utc).isoformat(),
                 "kind": kind, "payload": payload}
        with (self.path / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def checkpoint(self, state):
        write_json(self.path / "state.json", state)

    def finish(self, state, status, error=None):
        summary = {"run_id": self.run_id, "status": status, "answer": state.get("answer"),
                   "citations": state.get("citations", []), "unresolved": state["unresolved"],
                   "steps": state["step"], "calls": state["calls"], "error": error,
                   "reward": None, "synthetic_demo": state.get("synthetic_demo", False)}
        self.event("terminal", summary)
        self.checkpoint(state)
        write_json(self.path / "summary.json", summary)
        return summary


def events(run):
    with (Path(run) / "events.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def annotation_template(run, output):
    actions = [e["payload"] for e in events(run) if e["kind"] == "retrieval"]
    write_json(output, {"run_id": read_json(Path(run) / "manifest.json")["run_id"],
        "terminal": {"correct": None, "evidence_complete": None, "notes": ""},
        "steps": [{"step": a["state_before"]["step"], "labels": {
            c["view_id"]: "unknown" for c in a["candidates"]}, "notes": ""} for a in actions]})


def export_runs(runs, output, annotations=None, include_demo=False):
    """Export all RL transitions; only explicit positive labels produce retrieval SFT pairs."""
    runs, output = Path(runs), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifests = [runs / "manifest.json"] if (runs / "manifest.json").exists() else sorted(runs.glob("*/manifest.json"))
    sft, transitions, episode_count = [], [], 0
    for manifest_path in manifests:
        run = manifest_path.parent
        manifest = read_json(manifest_path)
        if manifest.get("synthetic_demo") and not include_demo:
            continue
        summary = read_json(run / "summary.json") if (run / "summary.json").exists() else {"status": "interrupted"}
        annotation = {}
        if annotations:
            ann_path = Path(annotations) / (manifest["run_id"] + ".json")
            if ann_path.exists():
                annotation = read_json(ann_path)
                if annotation["run_id"] != manifest["run_id"]:
                    raise ValueError("Annotation run id mismatch")
        labels = {v["step"]: v["labels"] for v in annotation.get("steps", [])}
        terminal = annotation.get("terminal", {})
        correct, complete = terminal.get("correct"), terminal.get("evidence_complete")
        reward = None
        if type(correct) is bool and type(complete) is bool:
            reward = float(correct) + .5 * float(complete) - .02 * summary.get("steps", 0)
        event_list = list(events(run))
        after = {e["payload"]["step"]: e["payload"] for e in event_list if e["kind"] == "observation"}
        for event in event_list:
            if event["kind"] != "retrieval":
                continue
            action = event["payload"]
            step = action["state_before"]["step"]
            candidate_ids = {c["view_id"] for c in action["candidates"]}
            judgment = labels.get(step, {})
            if not set(judgment).issubset(candidate_ids) or any(v not in ("positive", "negative", "unknown") for v in judgment.values()):
                raise ValueError("Invalid annotation: candidate id or label")
            transition = {"schema_version": 1, "run_id": manifest["run_id"], "corpus_fingerprint": manifest["corpus_fingerprint"],
                          "corpus_manifest": str((run / manifest["corpus"] / "manifest.json").resolve()),
                          "synthetic_demo": manifest.get("synthetic_demo", False), **copy.deepcopy(action),
                          "observation": after.get(step), "terminal_reward": reward,
                          "terminal_status": summary["status"], "labels": judgment,
                          "trainable_transition": step in after and summary["status"] not in ("error", "interrupted"),
                          "on_policy_eligible": action["selected"]["policy"] == "sample" and step in after and summary["status"] not in ("error", "interrupted")}
            transitions.append(transition)
            positives = [k for k, v in judgment.items() if v == "positive"]
            negatives = [k for k, v in judgment.items() if v == "negative"]
            if positives and step in after and summary["status"] not in ("error", "interrupted"):
                sft.append({"run_id": manifest["run_id"], "step": step, "query": action["state_query"],
                            "state": action["state_before"], "positive_view_ids": positives,
                            "negative_view_ids": negatives,
                            "unknown_view_ids": sorted(candidate_ids - set(positives) - set(negatives)),
                            "corpus_manifest": transition["corpus_manifest"],
                            "corpus_fingerprint": manifest["corpus_fingerprint"], "synthetic_demo": manifest.get("synthetic_demo", False)})
        episode_count += 1
    for name, rows in (("retriever_sft.jsonl", sft), ("rl_transitions.jsonl", transitions)):
        with (output / name).open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    stats = {"episodes": episode_count, "sft_pairs": len(sft), "transitions": len(transitions)}
    write_json(output / "stats.json", stats)
    return stats
