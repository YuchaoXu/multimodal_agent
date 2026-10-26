"""HTTP bridge to a separately hosted, trainable multimodal embedding model.

This is a project-defined /encode contract, not the OpenAI embedding schema.
The encoder service owns its model, batching, pooling, dtype and GPU placement.
"""
import base64
import json
import os
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit


class HTTPEncoder:
    def __init__(self):
        self.url = os.environ.get("MMAGENT_ENCODER_URL", "http://localhost:8001/encode")
        parts = urlsplit(self.url)
        if parts.username or parts.password or parts.query or parts.fragment:
            raise ValueError("Encoder URL must not embed credentials")
        if parts.scheme != "https" and not (parts.scheme == "http" and parts.hostname in ("localhost", "127.0.0.1", "::1")):
            raise ValueError("Remote encoder endpoint must use HTTPS")
        self.token = os.environ.get("MMAGENT_ENCODER_KEY", "")

    def _encode(self, kind, inputs):
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = "Bearer " + self.token
        req = urllib.request.Request(self.url, data=json.dumps({"kind": kind, "inputs": inputs}).encode(), headers=headers)
        with urllib.request.urlopen(req, timeout=120) as response:
            return json.load(response)["embeddings"]

    def encode_queries(self, queries):
        return self._encode("queries", [{"text": query} for query in queries])

    def encode_views(self, views):
        return self._encode("views", [{"view_id": v["id"], "text": v["text"], "bbox": v["bbox"],
            "image_base64": base64.b64encode(Path(v["image_path"]).read_bytes()).decode(),
            "mime_type": "image/png"} for v in views])


def http_encoder():
    return HTTPEncoder()
