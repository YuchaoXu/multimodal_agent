from __future__ import annotations
import hashlib
import json
import os
import tempfile
from pathlib import Path
from dataclasses import dataclass, asdict


def digest(value):
    raw = value if isinstance(value, bytes) else json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(raw).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=".write-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


@dataclass
class Config:
    provider: str = "openai"
    base_url: str = "https://api.openai.com/v1"
    model: str = ""
    api_key_env: str = "MMAGENT_API_KEY"
    timeout: int = 120
    retries: int = 2
    max_output_tokens: int = 1600
    json_mode: bool = True
    output_token_field: str = "max_completion_tokens"
    max_steps: int = 8
    max_calls: int = 24
    max_seconds: int = 900
    candidate_k: int = 12
    selection: str = "greedy"
    temperature: float = 1.0
    seed: int = 7
    anchor_images: int = 2
    duplicate_penalty: float = 1.0
    retriever: str = "bm25"
    encoder_factory: str = ""
    encoder_id: str = ""
    query_encoder_id: str = ""
    dense_path: str = ""

    def validate(self):
        for key in ("timeout", "max_output_tokens", "max_steps", "max_calls", "max_seconds", "candidate_k"):
            if not isinstance(getattr(self, key), int) or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be a positive integer")
        if self.output_token_field not in ("max_tokens", "max_completion_tokens"):
            raise ValueError("Unsupported output_token_field")
        if self.selection not in ("greedy", "sample") or self.temperature <= 0:
            raise ValueError("selection must be greedy/sample; temperature must be positive")
        if self.provider not in ("openai", "demo") or self.retriever not in ("bm25", "dense"):
            raise ValueError("Unsupported provider or retriever")
        if not 0 <= self.anchor_images <= 8 or not 0 <= self.retries <= 5:
            raise ValueError("anchor_images must be 0..8 and retries 0..5")
        if self.provider == "openai" and not self.model:
            raise ValueError("Set model in your configuration")
        return self

    @classmethod
    def load(cls, path):
        return cls(**read_json(path)).validate()

    def public(self):
        return asdict(self)
