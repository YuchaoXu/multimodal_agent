from __future__ import annotations
import shutil
from pathlib import Path
from .common import digest, read_json, write_json

# Normalized coordinates use the top-left origin. Re-reading another region is a new action.
REGIONS = {"full": [0, 0, 1, 1], "top": [0, 0, 1, .55], "bottom": [0, .45, 1, 1]}


class Corpus:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.manifest = read_json(self.root / "manifest.json")
        if self.manifest["fingerprint"] != digest(self.manifest["views"]):
            raise ValueError("Corpus manifest fingerprint mismatch")
        self.views = self.manifest["views"]
        self.by_id = {v["id"]: v for v in self.views}
        if len(self.by_id) != len(self.views):
            raise ValueError("Duplicate view ids")

    def image(self, view_id):
        v = self.by_id[view_id]
        path = (self.root / v["image"]).resolve()
        if not path.is_relative_to(self.root) or digest(path.read_bytes()) != v["image_sha256"]:
            raise ValueError("Image missing, outside corpus, or changed")
        return path


def _add_page(root, doc_id, source_name, page_no, image, text, views):
    for name, box in REGIONS.items():
        crop = image.crop(tuple(round(x * (image.width if i % 2 == 0 else image.height)) for i, x in enumerate(box)))
        crop.thumbnail((1400, 1800) if name != "full" else (1000, 1400))
        relative = f"images/{doc_id}-p{page_no}-{name}.png"
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        crop.convert("RGB").save(path)
        views.append({"id": f"{doc_id}:p{page_no}:{name}", "doc_id": doc_id,
                      "source": source_name, "page": page_no, "region": name, "bbox": box,
                      "width": crop.width, "height": crop.height, "image": relative,
                      "image_sha256": digest(path.read_bytes()),
                      "text": text, "text_scope": "page_context"})


def ingest(source, output, max_pages=None):
    """Build a new immutable corpus. Existing output is never silently replaced."""
    from pypdf import PdfReader
    import pypdfium2 as pdfium
    source, output = Path(source).resolve(), Path(output).resolve()
    files = sorted(p for p in source.rglob("*") if p.is_file() and p.suffix.lower() == ".pdf") if source.is_dir() else [source]
    if not files or any(p.suffix.lower() != ".pdf" or not p.is_file() for p in files):
        raise ValueError("Input must be a PDF or a directory containing PDFs")
    if max_pages is not None and max_pages < 1:
        raise ValueError("max_pages must be positive")
    output.mkdir(parents=True, exist_ok=False)
    views, documents = [], []
    seen = set()
    try:
        for path in files:
            file_hash = digest(path.read_bytes())
            if file_hash in seen:
                continue
            seen.add(file_hash)
            doc_id = file_hash[:20]
            copied = output / "sources" / f"{doc_id}.pdf"
            copied.parent.mkdir(exist_ok=True)
            shutil.copy2(path, copied)
            pdf = pdfium.PdfDocument(str(copied))
            reader = PdfReader(copied)
            count = min(len(pdf), max_pages or len(pdf))
            documents.append({"id": doc_id, "name": path.name, "sha256": file_hash,
                              "pages": len(pdf), "indexed_pages": count, "file": str(copied.relative_to(output))})
            try:
                for i in range(count):
                    page = pdf[i]
                    scale = min(2.0, 2400 / max(page.get_size()))
                    bitmap = page.render(scale=scale)
                    image = bitmap.to_pil().copy()
                    try:
                        text = reader.pages[i].extract_text() or ""
                        _add_page(output, doc_id, path.name, i + 1, image, text, views)
                    finally:
                        image.close()
                        bitmap.close()
                        page.close()
            finally:
                pdf.close()
        write_json(output / "manifest.json", {"schema_version": 1, "documents": documents,
                   "views": views, "fingerprint": digest(views), "ocr": False})
    except Exception:
        write_json(output / "ingest_error.json", {"status": "incomplete", "usable": False})
        raise
    return Corpus(output)


def make_demo(output):
    """Synthetic two-hop fixture, not an evaluation of visual reasoning."""
    from PIL import Image, ImageDraw
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    texts = ["Project Aurora uses sensor K7. Its performance is documented separately.",
             "K7 sensor efficiency table: 2024 = 60%; 2025 = 75%. Change = 15 percentage points.",
             "Other sensors: M2 measures temperature. This page is unrelated to Aurora efficiency."]
    views = []
    for i, text in enumerate(texts, 1):
        image = Image.new("RGB", (1000, 1300), "white")
        draw = ImageDraw.Draw(image)
        draw.text((60, 80), f"Synthetic report / page {i}", fill="black", font_size=32)
        import textwrap
        for row, line in enumerate(textwrap.wrap(text, 45)):
            draw.text((60, 200 + row * 45), line, fill="black", font_size=26)
        _add_page(root, "demo-report", "demo-report", i, image, text, views)
    write_json(root / "manifest.json", {"schema_version": 1, "documents": [], "views": views,
               "fingerprint": digest(views), "synthetic_demo": True})
    return Corpus(root)


def ingest_pages(manifest, output):
    """Import benchmark page images + optional OCR text from JSONL, without an LLM call."""
    import json
    from PIL import Image, ImageOps
    manifest, output = Path(manifest).resolve(), Path(output).resolve()
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("Page manifest is empty")
    output.mkdir(parents=True, exist_ok=False)
    views, seen = [], set()
    for row in rows:
        document = row["document"]
        page_number = row["page"]
        if not isinstance(document, str) or not document or type(page_number) is not int or page_number < 1:
            raise ValueError("Each page requires a nonempty document name and a positive integer page")
        key = (document, page_number)
        if key in seen:
            raise ValueError("Duplicate document/page in manifest")
        seen.add(key)
        text = row.get("text", "")
        if not isinstance(text, str):
            raise ValueError("Page text must be a string")
        image_path = (manifest.parent / row["image"]).resolve()
        with Image.open(image_path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            # Prevent excessively large benchmark images from exhausting memory downstream.
            image.thumbnail((2400, 3200))
            _add_page(output, digest(document)[:20], document, page_number, image, text, views)
    write_json(output / "manifest.json", {"schema_version": 1, "documents": [], "views": views,
               "fingerprint": digest(views), "source_type": "page_manifest", "ocr": "user_supplied"})
    return Corpus(output)
