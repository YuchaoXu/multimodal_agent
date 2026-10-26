from __future__ import annotations
import importlib
import math
import random
import re
from collections import Counter
from pathlib import Path
from .common import digest, read_json, write_json


def tokens(text):
    return re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", text.lower())


def state_query(state, query):
    facts = [f["text"] for f in state["facts"][-12:]]
    return "\n".join(["Question: " + state["question"], "Current need: " + query,
                      "Evidence: " + "; ".join(facts),
                      "Unresolved: " + "; ".join(state["unresolved"]),
                      "Failed attempts: " + "; ".join(state["failures"][-4:])])


def load_encoder(factory):
    if ":" not in factory:
        raise ValueError("encoder_factory must be 'module:factory'")
    module, name = factory.split(":", 1)
    return getattr(importlib.import_module(module), name)()


def normalized(values):
    import numpy as np
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError("Encoder must return a finite [batch, dimension] matrix")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if (norms <= 0).any():
        raise ValueError("Zero embedding")
    return array / norms


def build_dense(corpus, factory, encoder_id, output, batch_size=16):
    """The plugin receives image paths + page context, and owns model-specific processing."""
    import numpy as np
    if not encoder_id or batch_size < 1:
        raise ValueError("encoder_id and positive batch_size are required")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    encoder = load_encoder(factory)
    chunks = []
    for start in range(0, len(corpus.views), batch_size):
        batch = [dict(v, image_path=str(corpus.image(v["id"]))) for v in corpus.views[start:start + batch_size]]
        vectors = normalized(encoder.encode_views(batch))
        if len(vectors) != len(batch):
            raise ValueError("Encoder returned incorrect batch size")
        chunks.append(vectors)
    matrix = np.concatenate(chunks)
    np.save(output / "vectors.npy", matrix, allow_pickle=False)
    write_json(output / "metadata.json", {"corpus_fingerprint": corpus.manifest["fingerprint"],
               "encoder_id": encoder_id, "encoder_factory": factory, "view_ids": [v["id"] for v in corpus.views],
               "vectors_sha256": digest((output / "vectors.npy").read_bytes())})


class Retriever:
    """State-conditioned retriever boundary; the agent never owns trainable weights."""
    def __init__(self, corpus, config):
        self.corpus, self.config = corpus, config
        if not corpus.views:
            raise ValueError("Corpus has no views")
        self.counts = [Counter(tokens(v["text"])) for v in corpus.views]
        self.lengths = [sum(c.values()) for c in self.counts]
        self.average = sum(self.lengths) / len(self.lengths) or 1
        self.df = Counter(t for c in self.counts for t in c)
        if config.retriever == "dense":
            import numpy as np
            root = Path(config.dense_path)
            meta = read_json(root / "metadata.json")
            if (meta["corpus_fingerprint"] != corpus.manifest["fingerprint"] or
                meta["encoder_id"] != config.encoder_id or
                meta["vectors_sha256"] != digest((root / "vectors.npy").read_bytes()) or
                meta["view_ids"] != [v["id"] for v in corpus.views]):
                raise ValueError("Dense index does not match corpus/encoder/checksum")
            self.encoder = load_encoder(config.encoder_factory)
            self.matrix = normalized(np.load(root / "vectors.npy", allow_pickle=False))
            if self.matrix.shape[0] != len(corpus.views):
                raise ValueError("Dense index row count mismatch")
            self.index_id = digest(meta)
        else:
            self.index_id = "bm25-v1:" + corpus.manifest["fingerprint"]

        self.policy_id = digest({"index": self.index_id, "query_encoder": config.query_encoder_id or config.encoder_id,
                                 "factory": config.encoder_factory, "scorer_code": digest(Path(__file__).read_bytes()),
                                 "selection": config.selection, "temperature": config.temperature,
                                 "candidate_k": config.candidate_k, "duplicate_penalty": config.duplicate_penalty})

    def candidates(self, state, query, focus=None):
        text = state_query(state, query)
        if self.config.retriever == "dense":
            query_vectors = normalized(self.encoder.encode_queries([text]))
            if query_vectors.shape != (1, self.matrix.shape[1]):
                raise ValueError("Query embedding dimension mismatch")
            scores = (self.matrix @ query_vectors[0]).tolist()
        else:
            # Current need carries more weight than accumulated context.
            terms = Counter(tokens(query))
            for token in set(tokens(text)):
                terms[token] += .15
            scores = []
            for counts, length in zip(self.counts, self.lengths):
                score = 0.0
                for token, weight in terms.items():
                    freq = counts[token]
                    idf = math.log(1 + (len(self.counts) - self.df[token] + .5) / (self.df[token] + .5))
                    score += weight * idf * freq * 2.5 / (freq + 1.5 * (.25 + .75 * length / self.average))
                scores.append(score)
        focus = focus or []
        result = []
        for view, score in zip(self.corpus.views, scores):
            reasons = ["global"]
            # Focus only affects known views; it never removes global candidates.
            for anchor in focus:
                origin = self.corpus.by_id.get(anchor)
                if origin and view["doc_id"] == origin["doc_id"] and abs(view["page"] - origin["page"]) <= 1:
                    score += .25
                    reasons.append("local_or_revisit")
                    break
            if view["region"] == "full":
                score += .05
            if view["id"] in state["visited"]:
                score -= self.config.duplicate_penalty * state["visited"].count(view["id"])
                reasons.append("exact_repeat")
            result.append({"view_id": view["id"], "score": float(score), "channels": reasons})
        result.sort(key=lambda v: (-v["score"], v["view_id"]))
        # Probabilities are conditional on this exact truncated candidate pool.
        result = result[:self.config.candidate_k]
        logits = [v["score"] / self.config.temperature for v in result]
        peak = max(logits)
        weights = [math.exp(max(-700, x - peak)) for x in logits]
        denominator = sum(weights)
        for v, weight in zip(result, weights):
            v["softmax_probability"] = weight / denominator
        return text, result

    def choose(self, candidates, step, forced=None):
        if forced:
            if forced not in [x["view_id"] for x in candidates]:
                raise ValueError("Forced action is outside recorded candidate support")
            selected = next(x for x in candidates if x["view_id"] == forced)
            return dict(selected, behavior_probability=None, behavior_logprob=None, policy="intervention")
        if self.config.selection == "sample":
            rng = random.Random(self.config.seed + step)
            selected = rng.choices(candidates, weights=[x["softmax_probability"] for x in candidates])[0]
            probability = selected["softmax_probability"]
        else:
            selected, probability = candidates[0], 1.0
        return dict(selected, behavior_probability=probability, behavior_logprob=math.log(probability),
                    policy=self.config.selection)
