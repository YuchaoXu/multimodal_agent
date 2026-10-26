# Validation

Date: 2026-10-23

Python: 3.12.14

- pypdf: 6.10.0
- pypdfium2: 5.14.0
- Pillow: 12.3.0
- numpy: 2.3.5

- 22 unittest cases passed.
- Offline synthetic two-hop rollout: completed in 2 retrieval steps and 5 scripted model calls.
- Real ITER PDF: first 3 pages parsed into 9 page/region images; extracted 10,237 text characters; retrieval and image integrity checks passed.
- Multimodal HTTP requests and embedding transport were tested with mocked responses.
- No external VLM, live embedding checkpoint, GPU training or benchmark evaluation was performed.
