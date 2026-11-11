"""Content-addressed local PDF library. Files and indexes never enter Git."""

import hashlib
import json
import re
from collections import OrderedDict
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter

from pypdf import PdfReader

from .evidence import load_pdf
from .storage import outside_repository

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_PAGES = 500


class DocumentStore:
    def __init__(self, directory: Path):
        self.directory = outside_repository(outside_repository(directory) / "documents")
        self.directory.mkdir(parents=True, exist_ok=True)
        self.index_directory = outside_repository(self.directory / "indexes")
        self.index_directory.mkdir(exist_ok=True)
        self.indexes = OrderedDict()
        self.default_id = None

    def metadata(self, document_id):
        if not isinstance(document_id, str) or not re.fullmatch(
            r"[0-9a-f]{64}", document_id
        ):
            raise ValueError("Choose an uploaded document.")
        path = self.directory / (document_id + ".json")
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raise ValueError("This document is not in the local library.") from None
        if value.get("id") != document_id:
            raise ValueError("Invalid local document metadata.")
        return value

    def list(self):
        values = []
        for path in self.directory.glob("*.json"):
            try:
                value = self.metadata(path.stem)
            except ValueError:
                continue
            values.append(value)
        return sorted(values, key=lambda value: value["name"].casefold())

    def path(self, document_id):
        self.metadata(document_id)
        return self.directory / (document_id + ".pdf")

    def read_pdf(self, document_id):
        raw = self.path(document_id).read_bytes()
        if hashlib.sha256(raw).hexdigest() != document_id:
            raise ValueError("The stored PDF changed. Upload the original again.")
        return raw

    def get_index(self, document_id):
        path = self.path(document_id)
        stamp = (path.stat().st_mtime_ns, path.stat().st_size)
        cached = self.indexes.get(document_id)
        if cached and cached[0] == stamp:
            self.indexes.move_to_end(document_id)
            return cached[1], True
        index, hit = load_pdf(path, self.index_directory)
        if index.fingerprint != document_id:
            raise ValueError("The stored PDF changed. Upload the original again.")
        self.indexes[document_id] = (stamp, index)
        while len(self.indexes) > 2:
            self.indexes.popitem(last=False)
        return index, hit

    def register_path(self, path):
        path = outside_repository(path)
        if path.stat().st_size > MAX_UPLOAD_BYTES:
            raise ValueError("PDF uploads must be 25 MB or smaller.")
        result = self.upload(path.read_bytes(), path.name)
        self.default_id = result["document"]["id"]
        return result

    def upload(self, raw, name):
        start = perf_counter()
        if not isinstance(raw, bytes) or not 1 <= len(raw) <= MAX_UPLOAD_BYTES:
            raise ValueError("PDF uploads must be between 1 byte and 25 MB.")
        if not isinstance(name, str):
            raise ValueError("Choose a PDF filename.")
        name = re.split(r"[/\\]", name)[-1].strip()[:160]
        if not name.lower().endswith(".pdf") or b"%PDF-" not in raw[:1024]:
            raise ValueError("Choose a PDF file. Other formats are not supported yet.")
        try:
            reader = PdfReader(BytesIO(raw))
            if reader.is_encrypted:
                raise ValueError("Password-protected PDFs need an unlocked local copy.")
            pages = len(reader.pages)
            if not 1 <= pages <= MAX_PAGES:
                raise ValueError("PDFs must have between 1 and 500 pages.")
        except ValueError:
            raise
        except Exception:
            raise ValueError(
                "This PDF could not be opened. Try a readable PDF copy."
            ) from None
        document_id = hashlib.sha256(raw).hexdigest()
        pdf = self.directory / (document_id + ".pdf")
        metadata = self.directory / (document_id + ".json")
        if metadata.exists():
            self.get_index(document_id)
            return {
                "document": self.metadata(document_id),
                "duplicate": True,
                "index_ms": round((perf_counter() - start) * 1000, 2),
            }
        temporary = None
        try:
            with NamedTemporaryFile(
                dir=self.directory, suffix=".pdf", delete=False
            ) as handle:
                handle.write(raw)
                temporary = Path(handle.name)
            index, _ = load_pdf(temporary, self.index_directory)
            if sum(len(p.text) for p in index.pages) > 10_000_000:
                raise ValueError(
                    "This PDF contains more text than the current limit permits."
                )
            searchable = len(index.pages)
            value = {
                "id": document_id,
                "name": name,
                "pages": pages,
                "searchable_pages": searchable,
                "bytes": len(raw),
                "warning": ""
                if searchable == pages
                else f"{pages - searchable} pages have no searchable text. "
                "Image-only pages need OCR and are not searched.",
            }
            temporary.replace(pdf)
            temporary = None
            with metadata.open("x", encoding="utf-8") as handle:
                json.dump(value, handle, ensure_ascii=False)
            self.get_index(document_id)
            return {
                "document": value,
                "duplicate": False,
                "index_ms": round((perf_counter() - start) * 1000, 2),
            }
        except ValueError as exc:
            if "No extractable text" in str(exc):
                raise ValueError(
                    "No searchable text was found. This may be a scanned PDF; "
                    "create an OCR/text PDF locally, then upload it again."
                ) from None
            raise
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
