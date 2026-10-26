from __future__ import annotations
import argparse
import copy
import json
import re
import sys
from pathlib import Path
from .agent import Agent
from .common import Config, read_json, write_json
from .corpus import Corpus, ingest, make_demo, ingest_pages
from .retrieval import build_dense
from .trajectory import annotation_template, events, export_runs


def evaluate(run, gold):
    """Evaluation is post-rollout only. Gold never enters agent state or prompts."""
    run = Path(run)
    summary, state = read_json(run / "summary.json"), read_json(run / "state.json")
    normalize = lambda text: re.sub(r"\s+", " ", str(text).strip().lower())
    expected = gold.get("answers", [gold["answer"]] if "answer" in gold else [])
    if not isinstance(expected, list) or any(not isinstance(v, str) for v in expected):
        raise ValueError("gold.answers must be a list of strings")
    support = set(gold.get("support_view_ids", []))
    visited = set(state["visited"])
    result = {"run_id": summary["run_id"], "exact_match":
              any(normalize(summary.get("answer") or "") == normalize(a) for a in expected) if expected else None,
              "support_view_recall": len(visited & support) / len(support) if support else None,
              "all_support_views_visited": support.issubset(visited) if support else None,
              "steps": summary["steps"], "calls": summary["calls"],
              "note": "Support-view visits do not prove that the evidence was actually understood."}
    write_json(run / "evaluation.json", result)
    return result


def branch(run, step, view_id, output):
    run = Path(run).resolve()
    manifest = read_json(run / "manifest.json")
    corpus = Corpus(run / manifest["corpus"])
    if corpus.manifest["fingerprint"] != manifest["corpus_fingerprint"]:
        raise ValueError("Corpus changed since parent rollout")
    action = next((e["payload"] for e in events(run) if e["kind"] == "retrieval" and e["payload"]["state_before"]["step"] == step), None)
    if action is None:
        raise ValueError("No retrieval at requested step (steps start at 0)")
    if view_id not in [c["view_id"] for c in action["candidates"]]:
        raise ValueError("Alternative view must be in the recorded candidate pool")
    config = Config(**manifest["config"]).validate()
    agent = Agent(corpus, config)
    if action["index_id"] != agent.retriever.index_id or action["policy_id"] != agent.retriever.policy_id:
        raise ValueError("Retriever index changed")
    return agent.run(manifest["question"], output, state=copy.deepcopy(action["state_before"]),
        parent={"run_id": manifest["run_id"], "kind": "counterfactual", "step": step},
        forced_action=view_id, first_retrieval=action)


def resume(run, output):
    run = Path(run).resolve()
    manifest, state = read_json(run / "manifest.json"), read_json(run / "state.json")
    if (run / "summary.json").exists():
        status = read_json(run / "summary.json")["status"]
        if status not in ("error",):
            raise ValueError("Only interrupted or error runs can resume; completed/budget-ended runs are immutable")
    corpus = Corpus(run / manifest["corpus"])
    if corpus.manifest["fingerprint"] != manifest["corpus_fingerprint"]:
        raise ValueError("Corpus changed since checkpoint")
    config = Config(**manifest["config"]).validate()
    return Agent(corpus, config).run(manifest["question"], output, state=state,
        parent={"run_id": manifest["run_id"], "kind": "resume_checkpoint"})


def main(argv=None):
    parser = argparse.ArgumentParser(description="Frozen multimodal evidence agent")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("ingest", help="Render PDF(s), extract text, generate page/region views")
    p.add_argument("source"); p.add_argument("--output", required=True); p.add_argument("--max-pages", type=int)
    p = sub.add_parser("ingest-pages", help="Import benchmark images and optional OCR from JSONL")
    p.add_argument("manifest"); p.add_argument("--output", required=True)
    p = sub.add_parser("ask", help="Run a question against an indexed corpus")
    p.add_argument("--corpus", required=True); p.add_argument("--config", required=True)
    p.add_argument("--question", required=True); p.add_argument("--runs", default="runs")
    p = sub.add_parser("batch", help="Run JSONL rows containing id and question; gold is never passed to agent")
    p.add_argument("dataset"); p.add_argument("--corpus", required=True); p.add_argument("--config", required=True)
    p.add_argument("--runs", default="runs")
    p = sub.add_parser("demo", help="Offline deterministic two-hop smoke test, NOT a real VLM")
    p.add_argument("--output", default="data/demo")
    p = sub.add_parser("inspect"); p.add_argument("run")
    p = sub.add_parser("annotate-template"); p.add_argument("run"); p.add_argument("--output", required=True)
    p = sub.add_parser("export"); p.add_argument("runs"); p.add_argument("--output", required=True)
    p.add_argument("--annotations"); p.add_argument("--include-demo", action="store_true")
    p = sub.add_parser("evaluate"); p.add_argument("run"); p.add_argument("--gold", required=True)
    p = sub.add_parser("branch"); p.add_argument("run"); p.add_argument("--step", type=int, required=True)
    p.add_argument("--view-id", required=True); p.add_argument("--runs", default="runs")
    p = sub.add_parser("resume"); p.add_argument("run"); p.add_argument("--runs", default="runs")
    p = sub.add_parser("build-dense"); p.add_argument("--corpus", required=True)
    p.add_argument("--encoder-factory", required=True); p.add_argument("--encoder-id", required=True)
    p.add_argument("--output", required=True); p.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args(argv)
    try:
        result = dispatch(args)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if isinstance(result, dict) and (result.get("status") == "error" or result.get("errors", 0)):
            return 1
        return 0
    except (ValueError, OSError, KeyError, ImportError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2


def dispatch(args):
    if args.command == "ingest-pages":
        corpus = ingest_pages(args.manifest, args.output)
        return {"corpus": str(corpus.root), "views": len(corpus.views)}
    if args.command == "ingest":
        corpus = ingest(args.source, args.output, args.max_pages)
        return {"corpus": str(corpus.root), "views": len(corpus.views),
                "pages_without_extracted_text": len({(v["doc_id"], v["page"]) for v in corpus.views if not v["text"].strip()}),
                "note": "No OCR is run. Scanned PDFs need OCR page context or a visual embedding plugin for useful retrieval."}
    if args.command in ("ask", "batch"):
        corpus, config = Corpus(args.corpus), Config.load(args.config)
        agent = Agent(corpus, config)
        if args.command == "ask":
            return agent.run(args.question, args.runs)
        results = []
        with Path(args.dataset).open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                item = json.loads(line)
                # Explicit allowlist: answers/support annotations never enter the rollout.
                result = agent.run(item["question"], args.runs)
                write_json(Path(result["run_dir"]) / "dataset_reference.json", {"id": item.get("id")})
                if "gold" in item:
                    evaluate(result["run_dir"], item["gold"])
                results.append({"id": item.get("id"), **result})
        return {"results": results, "errors": sum(r["status"] == "error" for r in results)}
    if args.command == "demo":
        root = Path(args.output).resolve()
        root.mkdir(parents=True, exist_ok=False)
        corpus = make_demo(root / "corpus")
        config = Config(provider="demo", max_steps=5, candidate_k=9)
        result = Agent(corpus, config).run("How much did Aurora sensor efficiency change from 2024 to 2025?", root / "runs")
        return {**result, "note": "Synthetic scripted smoke test; no API call and no real visual model evaluation."}
    if args.command == "inspect":
        run = Path(args.run)
        return {"summary": read_json(run / "summary.json"), "state": read_json(run / "state.json")}
    if args.command == "annotate-template":
        if Path(args.output).exists():
            raise ValueError("Annotation file already exists")
        annotation_template(args.run, args.output)
        return {"template": args.output}
    if args.command == "export":
        return export_runs(args.runs, args.output, args.annotations, args.include_demo)
    if args.command == "evaluate":
        return evaluate(args.run, read_json(args.gold))
    if args.command == "branch":
        return branch(args.run, args.step, args.view_id, args.runs)
    if args.command == "resume":
        return resume(args.run, args.runs)
    if args.command == "build-dense":
        build_dense(Corpus(args.corpus), args.encoder_factory, args.encoder_id, args.output, args.batch_size)
        return {"index": args.output, "encoder_id": args.encoder_id}
