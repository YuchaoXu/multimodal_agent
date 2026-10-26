from __future__ import annotations
import copy
import time
from .model import OpenAIModel, DemoModel
from .retrieval import Retriever
from .trajectory import Recorder


class BudgetExceeded(Exception):
    pass


def strings(value, name, limit=32):
    if not isinstance(value, list) or len(value) > limit or any(not isinstance(x, str) or len(x) > 4000 for x in value):
        raise ValueError(f"Invalid {name}: expected a bounded list of strings")
    return value


def initial_state(question, config):
    return {"question": question, "step": 0, "calls": 0, "elapsed_seconds": 0.0,
            "facts": [], "unresolved": [question], "failures": [], "visited": [], "queries": [],
            "answer": None, "citations": [], "synthetic_demo": config.provider == "demo"}


class Agent:
    def __init__(self, corpus, config, model=None):
        self.corpus, self.config = corpus, config.validate()
        self.retriever = Retriever(corpus, config)
        self.model = model or (DemoModel() if config.provider == "demo" else OpenAIModel(config))
        if config.provider == "demo" and not corpus.manifest.get("synthetic_demo"):
            raise ValueError("DemoModel is restricted to the synthetic demo corpus")

    def run(self, question, runs, state=None, parent=None, forced_action=None, first_retrieval=None):
        if not isinstance(question, str) or not question.strip() or len(question) > 12000:
            raise ValueError("Question must contain 1..12000 characters")
        state = copy.deepcopy(state) if state is not None else initial_state(question, self.config)
        if state["question"] != question:
            raise ValueError("Checkpoint question mismatch")
        recorder = Recorder(runs, self.corpus, self.config, question, parent)
        started = time.monotonic()
        inherited_elapsed = state.get("elapsed_seconds", 0)
        status, error = "budget_exhausted", None

        def record(kind, payload):
            state["elapsed_seconds"] = inherited_elapsed + time.monotonic() - started
            if kind == "model_request":
                if state["calls"] >= self.config.max_calls or state["elapsed_seconds"] >= self.config.max_seconds:
                    raise BudgetExceeded()
                state["calls"] += 1
                # Persist budget even if the request/process fails before receiving a response.
                recorder.checkpoint(state)
            recorder.event(kind, payload)

        def generate(task, payload, images):
            return self.model.generate(task, payload, images, record)

        def anchors():
            ids = list(dict.fromkeys(f["view_id"] for f in state["facts"]))
            ids = ids[-self.config.anchor_images:] if self.config.anchor_images else []
            return [("ANCHOR " + v, self.corpus.image(v)) for v in ids]

        def accept_answer(value):
            answer = value.get("answer")
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError("Model returned an empty answer")
            citations = strings(value.get("citations", []), "citations")
            if not set(citations).issubset(set(state["visited"])):
                raise ValueError("Model cited evidence it has not read")
            unresolved = strings(value.get("unresolved", []), "unresolved")
            if not citations and not unresolved:
                raise ValueError("Answer needs citations or explicit unresolved requirements")
            state.update(answer=answer, citations=list(dict.fromkeys(citations)), unresolved=unresolved)

        recorder.event("start", {"state": copy.deepcopy(state), "index_id": self.retriever.index_id})
        recorder.checkpoint(state)
        try:
            while state["step"] < self.config.max_steps:
                if inherited_elapsed + time.monotonic() - started >= self.config.max_seconds:
                    raise BudgetExceeded()
                if first_retrieval is not None:
                    plan = copy.deepcopy(first_retrieval["plan"])
                else:
                    plan = generate("plan", {"state": copy.deepcopy(state),
                        "budget_remaining": {"steps": self.config.max_steps - state["step"],
                                             "calls": self.config.max_calls - state["calls"]}}, anchors())
                    record("plan", plan)
                if plan.get("action") == "finish":
                    accept_answer(plan)
                    status = "insufficient_evidence" if state["unresolved"] else "completed"
                    break
                if plan.get("action") != "search" or not isinstance(plan.get("query"), str) or not plan["query"].strip():
                    raise ValueError("Plan must finish or search with a nonempty query")
                focus = strings(plan.get("focus", []), "focus")
                if not set(focus).issubset(set(state["visited"])):
                    raise ValueError("Focus must reference previously visited views")
                query, candidates = self.retriever.candidates(state, plan["query"], focus)
                if first_retrieval is not None:
                    if candidates != first_retrieval["candidates"] or query != first_retrieval["state_query"]:
                        raise ValueError("Branch requires identical candidate support, state, and index")
                    first_retrieval = None
                selected = self.retriever.choose(candidates, state["step"], forced_action)
                forced_action = None
                action = {"state_before": copy.deepcopy(state), "state_query": query, "plan": plan,
                          "candidates": candidates, "selected": selected, "index_id": self.retriever.index_id,
                          "policy_id": self.retriever.policy_id,
                          "temperature": self.config.temperature, "candidate_k": self.config.candidate_k,
                          "probability_scope": "conditional_on_recorded_top_k"}
                record("retrieval", action)
                view = self.corpus.by_id[selected["view_id"]]
                image = self.corpus.image(view["id"])
                observation = generate("read", {"state": copy.deepcopy(state), "query": plan["query"],
                    "selected_view": view}, anchors() + [("SELECTED " + view["id"], image)])
                facts = observation.get("facts", [])
                if not isinstance(facts, list) or len(facts) > 20:
                    raise ValueError("Invalid observed facts")
                new_facts = []
                for fact in facts:
                    confidence = fact.get("confidence", .5)
                    if not isinstance(fact.get("text"), str) or not 0 < len(fact["text"]) <= 4000:
                        raise ValueError("Invalid fact text")
                    if type(confidence) not in (float, int) or not 0 <= confidence <= 1:
                        raise ValueError("Invalid confidence")
                    new_facts.append({"text": fact["text"], "confidence": confidence,
                                      "view_id": view["id"], "step": state["step"], "verified": False})
                state["unresolved"] = strings(observation.get("unresolved", []), "unresolved")
                if type(observation.get("useful")) is not bool:
                    raise ValueError("Observation useful must be boolean")
                step = state["step"]
                state["facts"].extend(new_facts)
                state["visited"].append(view["id"])
                state["queries"].append(plan["query"])
                if not observation["useful"]:
                    state["failures"].append(f"{view['id']}: {str(observation.get('reason', 'No useful evidence'))[:1000]}")
                state["step"] += 1
                record("observation", {"step": step, "view_id": view["id"], "observation": observation,
                                       "state_after": copy.deepcopy(state)})
                recorder.checkpoint(state)
            else:
                answer = generate("answer", {"state": copy.deepcopy(state)}, anchors())
                accept_answer(answer)
                status = "budget_exhausted"
        except BudgetExceeded:
            state["unresolved"] = state["unresolved"] or ["Budget ended before answer verification"]
        except Exception as exc:
            status, error = "error", type(exc).__name__ + ": " + str(exc)
        summary = recorder.finish(state, status, error)
        summary["run_dir"] = str(recorder.path)
        return summary
