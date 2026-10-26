import copy
import json
import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from mmagent.agent import Agent
from mmagent.common import Config, read_json, write_json
from mmagent.corpus import Corpus, make_demo, ingest
from mmagent.model import DemoModel, OpenAIModel, parse_json
from mmagent.retrieval import Retriever, build_dense
from mmagent.trajectory import events, annotation_template, export_runs
from mmagent.cli import branch, resume, evaluate, main


class FixtureEncoder:
    def encode_views(self, views):
        return [[1, v["page"], len(v["text"]) / 100] for v in views]

    def encode_queries(self, queries):
        return [[1, 1, 1] for _ in queries]


def make_encoder():
    return FixtureEncoder()


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.corpus = make_demo(self.root / "corpus")
        self.config = Config(provider="demo", max_steps=5, candidate_k=9)
        self.question = "How much did Aurora sensor efficiency change from 2024 to 2025?"

    def tearDown(self):
        self.temp.cleanup()

    def run_agent(self, config=None, model=None):
        return Agent(self.corpus, config or self.config, model).run(self.question, self.root / "runs")

    def test_end_to_end_two_hop_with_citations(self):
        result = self.run_agent()
        self.assertEqual(result["status"], "completed", result)
        self.assertEqual(result["steps"], 2)
        self.assertIn("15 percentage points", result["answer"])
        self.assertEqual(set(result["citations"]), {"demo-report:p1:full", "demo-report:p2:full"})
        trace = list(events(result["run_dir"]))
        self.assertEqual([e["seq"] for e in trace], list(range(1, len(trace) + 1)))
        retrieval = next(e["payload"] for e in trace if e["kind"] == "retrieval")
        self.assertEqual(retrieval["selected"]["behavior_probability"], 1.0)
        self.assertEqual(retrieval["state_before"]["facts"], [])
        self.assertEqual(result["calls"], 5)

    def test_budget_without_fabricated_answer(self):
        config = copy.copy(self.config); config.max_calls = 1
        result = self.run_agent(config)
        self.assertEqual(result["status"], "budget_exhausted")
        self.assertIsNone(result["answer"])
        self.assertEqual(result["calls"], 1)
        self.assertEqual(result["steps"], 0)

    def test_max_steps_forces_final_answer(self):
        config = copy.copy(self.config); config.max_steps = 1
        result = self.run_agent(config)
        self.assertEqual(result["status"], "budget_exhausted")
        self.assertEqual(result["steps"], 1)
        self.assertIn("Insufficient", result["answer"])

    def test_revisit_region_not_exact_duplicate(self):
        result = self.run_agent()
        state = read_json(Path(result["run_dir"]) / "state.json")
        state["visited"] = ["demo-report:p1:full"]
        _, candidates = Retriever(self.corpus, self.config).candidates(state, "Aurora sensor", ["demo-report:p1:full"])
        by_id = {c["view_id"]: c for c in candidates}
        self.assertIn("exact_repeat", by_id["demo-report:p1:full"]["channels"])
        self.assertNotIn("exact_repeat", by_id["demo-report:p1:top"]["channels"])
        self.assertEqual(len(candidates), 9)

    def test_sampling_probability_is_actual_behavior(self):
        config = copy.copy(self.config); config.selection = "sample"
        r = Retriever(self.corpus, config)
        candidates = [{"view_id": "a", "softmax_probability": .25}, {"view_id": "b", "softmax_probability": .75}]
        action = r.choose(candidates, 2)
        self.assertEqual(action["behavior_probability"], action["softmax_probability"])
        self.assertAlmostEqual(action["behavior_logprob"], math.log(action["behavior_probability"]))
        self.assertEqual(action, r.choose(candidates, 2))

    def test_counterfactual_does_not_leak_future(self):
        parent = self.run_agent()
        child = branch(parent["run_dir"], 0, "demo-report:p3:full", self.root / "branches")
        action = next(e["payload"] for e in events(child["run_dir"]) if e["kind"] == "retrieval")
        self.assertEqual(action["state_before"]["facts"], [])
        self.assertEqual(action["state_before"]["visited"], [])
        self.assertEqual(action["selected"]["policy"], "intervention")
        self.assertIsNone(action["selected"]["behavior_logprob"])
        self.assertEqual(action["state_before"]["calls"], 1)
        parent_action = next(e["payload"] for e in events(parent["run_dir"]) if e["kind"] == "retrieval")
        self.assertEqual(action["candidates"], parent_action["candidates"])

    def test_branch_rejects_unknown_action(self):
        parent = self.run_agent()
        with self.assertRaises(ValueError):
            branch(parent["run_dir"], 0, "made-up", self.root / "branches")

    def test_unlabeled_and_demo_not_silently_sft(self):
        result = self.run_agent()
        stats = export_runs(self.root / "runs", self.root / "export")
        self.assertEqual(stats["episodes"], 0)
        stats = export_runs(result["run_dir"], self.root / "export2", include_demo=True)
        self.assertEqual(stats["sft_pairs"], 0)
        self.assertEqual(stats["transitions"], 2)
        rows = [json.loads(x) for x in (self.root / "export2/rl_transitions.jsonl").read_text().splitlines()]
        self.assertIsNone(rows[0]["terminal_reward"])
        self.assertFalse(rows[0]["on_policy_eligible"])

    def test_annotations_preserve_unknown(self):
        result = self.run_agent()
        path = self.root / "annotations" / (result["run_id"] + ".json")
        annotation_template(result["run_dir"], path)
        ann = read_json(path)
        ann["terminal"].update(correct=True, evidence_complete=True)
        ann["steps"][0]["labels"]["demo-report:p1:full"] = "positive"
        ann["steps"][0]["labels"]["demo-report:p3:full"] = "negative"
        write_json(path, ann)
        stats = export_runs(result["run_dir"], self.root / "export", path.parent, True)
        self.assertEqual(stats["sft_pairs"], 1)
        row = json.loads((self.root / "export/retriever_sft.jsonl").read_text().splitlines()[0])
        self.assertIn("demo-report:p1:top", row["unknown_view_ids"])
        self.assertEqual(row["negative_view_ids"], ["demo-report:p3:full"])

    def test_gold_not_used_during_rollout(self):
        result = self.run_agent()
        gold = {"answer": "SECRET_GOLD_ANSWER", "support_view_ids": ["demo-report:p2:full"]}
        metric = evaluate(result["run_dir"], gold)
        self.assertFalse(metric["exact_match"])
        self.assertEqual(metric["support_view_recall"], 1)
        self.assertNotIn("SECRET_GOLD_ANSWER", (Path(result["run_dir"]) / "events.jsonl").read_text())

    def test_hallucinated_citation_records_failure(self):
        class BadModel(DemoModel):
            def generate(self, task, payload, images, record):
                record("model_request", {"task": task})
                return {"action": "finish", "answer": "made up", "citations": ["unknown"], "unresolved": []}
        result = self.run_agent(model=BadModel())
        self.assertEqual(result["status"], "error")
        self.assertIsNone(result["answer"])
        self.assertIn("not read", result["error"])

    def test_resume_retains_spent_calls(self):
        class FailingModel(DemoModel):
            def generate(self, task, payload, images, record):
                record("model_request", {"task": task})
                raise RuntimeError("test transient failure")
        parent = self.run_agent(model=FailingModel())
        child = resume(parent["run_dir"], self.root / "resumed")
        self.assertEqual(child["status"], "completed")
        self.assertEqual(child["calls"], 6)

    def test_immutable_completed_runs(self):
        result = self.run_agent()
        with self.assertRaises(ValueError):
            resume(result["run_dir"], self.root / "resumed")

    def test_asset_tampering_rejected(self):
        path = self.corpus.image("demo-report:p1:full")
        path.write_bytes(b"corrupt")
        with self.assertRaises(ValueError):
            self.corpus.image("demo-report:p1:full")

    def test_dense_encoder_plugin_and_index_fingerprint(self):
        config = copy.copy(self.config)
        config.retriever = "dense"; config.encoder_factory = "test_agent:make_encoder"
        config.encoder_id = "fixture-v1"; config.dense_path = str(self.root / "dense")
        build_dense(self.corpus, config.encoder_factory, config.encoder_id, config.dense_path, 3)
        r = Retriever(self.corpus, config)
        from mmagent.agent import initial_state
        _, candidates = r.candidates(initial_state(self.question, config), "K7")
        self.assertEqual(len(candidates), 9)
        self.assertAlmostEqual(sum(c["softmax_probability"] for c in candidates), 1)
        config.encoder_id = "changed-v2"
        with self.assertRaises(ValueError):
            Retriever(self.corpus, config)

    def test_pdf_ingestion_and_crops(self):
        from pypdf import PdfWriter
        pdf = self.root / "source.pdf"
        writer = PdfWriter(); writer.add_blank_page(width=600, height=800)
        with pdf.open("wb") as stream:
            writer.write(stream)
        corpus = ingest(pdf, self.root / "pdf-corpus")
        self.assertEqual(len(corpus.views), 3)
        self.assertEqual(corpus.views[0]["page"], 1)
        self.assertEqual(corpus.views[1]["bbox"], [0, 0, 1, .55])
        self.assertTrue(corpus.image(corpus.views[0]["id"]).exists())
        with self.assertRaises(FileExistsError):
            ingest(pdf, self.root / "pdf-corpus")

    def test_openai_payload_images_and_no_key_in_trace(self):
        class Response:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read(self):
                return json.dumps({"choices": [{"message": {"content": '{"action":"search","query":"K7"}'}}], "usage": {"total_tokens": 10}}).encode()
        config = Config(model="test-vision")
        logs = []
        with patch.dict(os.environ, {"MMAGENT_API_KEY": "secret-fixture"}), patch("urllib.request.urlopen", return_value=Response()) as send:
            model = OpenAIModel(config)
            model.generate("plan", {"state": {}}, [("SELECTED x", self.corpus.image("demo-report:p1:full"))], lambda k, p: logs.append((k, p)))
            body = json.loads(send.call_args.args[0].data)
            self.assertTrue(body["messages"][1]["content"][-1]["image_url"]["url"].startswith("data:image/png;base64,"))
            self.assertIn("untrusted evidence", body["messages"][0]["content"])
            self.assertNotIn("secret-fixture", json.dumps(logs))

    def test_api_retry_budget_is_counted(self):
        class BrokenResponse:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read(self): return b'{"choices":[{"message":{"content":"invalid json"}}]}'
        config = Config(model="test", max_calls=1)
        with patch.dict(os.environ, {"MMAGENT_API_KEY": "fixture"}), patch("urllib.request.urlopen", return_value=BrokenResponse()) as send, patch("time.sleep"):
            result = Agent(self.corpus, config).run(self.question, self.root / "runs")
        self.assertEqual(send.call_count, 1)
        self.assertEqual(result["calls"], 1)
        self.assertEqual(result["status"], "budget_exhausted")

    def test_image_benchmark_ingestion(self):
        from mmagent.corpus import ingest_pages
        manifest = self.root / "pages.jsonl"
        manifest.write_text(json.dumps({"document": "report-A", "page": 3,
                            "image": str(self.corpus.image("demo-report:p1:full")), "text": "Aurora"}) + "\n")
        corpus = ingest_pages(manifest, self.root / "images-corpus")
        self.assertEqual(len(corpus.views), 3)
        self.assertEqual(corpus.views[0]["page"], 3)
        self.assertEqual(corpus.views[0]["text_scope"], "page_context")

    def test_http_embedding_contract(self):
        from mmagent.encoders import HTTPEncoder
        class Response:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read(self): return b'{"embeddings":[[1,2,3]]}'
        with patch("urllib.request.urlopen", return_value=Response()) as send:
            encoder = HTTPEncoder()
            self.assertEqual(encoder.encode_queries(["need K7"]), [[1, 2, 3]])
            payload = json.loads(send.call_args.args[0].data)
            self.assertEqual(payload["kind"], "queries")
            view = dict(self.corpus.views[0], image_path=str(self.corpus.image(self.corpus.views[0]["id"])))
            encoder.encode_views([view])
            payload = json.loads(send.call_args.args[0].data)
            self.assertEqual(payload["kind"], "views")
            self.assertTrue(payload["inputs"][0]["image_base64"])

    def test_batch_gold_is_allowlisted_out(self):
        from argparse import Namespace
        from mmagent.cli import dispatch
        write_json(self.root / "config.json", self.config.public())
        dataset = self.root / "questions.jsonl"
        dataset.write_text(json.dumps({"id": "sample1", "question": self.question,
                          "gold": {"answer": "DO_NOT_LEAK"}}) + "\n")
        result = dispatch(Namespace(command="batch", corpus=str(self.corpus.root), config=str(self.root / "config.json"),
                          dataset=str(dataset), runs=str(self.root / "runs")))
        run = Path(result["results"][0]["run_dir"])
        self.assertNotIn("DO_NOT_LEAK", (run / "events.jsonl").read_text())
        self.assertFalse(read_json(run / "evaluation.json")["exact_match"])

    def test_json_and_config_validation(self):
        self.assertEqual(parse_json('```json\n{"ok": true}\n```'), {"ok": True})
        with self.assertRaises(ValueError): parse_json('[]')
        with self.assertRaises(ValueError): Config(provider="demo", candidate_k=0).validate()
        with self.assertRaises(ValueError): Config(provider="demo", selection="random").validate()


if __name__ == "__main__":
    unittest.main()
