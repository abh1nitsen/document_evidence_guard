"""Local, versioned extraction and retrieval. Page IDs use PDF page order."""

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

import pypdf

STOP = set(
    (
        "a an and are as at be by can did do does for from how i in is it me of "
        "on or report show tell that the this to was were what when where which why with"
    ).split()
)


def terms(value: str) -> set[str]:
    return set(re.findall(r"\w+(?:\.\d+)*", value.casefold())) - STOP


@dataclass(frozen=True)
class Page:
    number: int
    text: str


class Index:
    def __init__(self, pages: list[Page], fingerprint: str):
        self.pages = pages
        self.fingerprint = fingerprint
        self.words = {page.number: terms(page.text) for page in pages}
        self.frequency = Counter(word for words in self.words.values() for word in words)

    def fetch(self, query: str, limit: int = 3) -> list[Page]:
        words = terms(query)
        scores = [
            (
                sum(
                    math.log1p(len(self.pages) / self.frequency[word])
                    for word in sorted(words & self.words[page.number])
                ),
                page,
            )
            for page in self.pages
        ]
        ranked = sorted(scores, key=lambda pair: (-pair[0], pair[1].number))
        return [
            page
            for score, page in ranked
            if score > 0 and not re.search(r"\.{8,}\s*\d+", page.text)
        ][:limit]

    def adjacent(self, pages: list[Page], limit: int = 2) -> list[Page]:
        known = {page.number for page in pages}
        candidates = {n + shift for n in known for shift in (-1, 1)} - known
        ordered = {page.number: page for page in self.pages if page.number in candidates}
        nearby = dict.fromkeys(page.number + shift for page in pages for shift in (-1, 1))
        return [ordered[n] for n in nearby if n in ordered][:limit]

    def excerpts(self, pages: list[Page], question: str) -> list[Page]:
        """Select short, overlapping text windows while retaining source page IDs."""
        words = terms(question)
        excerpts = []
        for page in pages:
            lines = page.text.splitlines()
            windows = [
                " ".join(" ".join(lines[i : i + 4]).split())
                for i in range(0, len(lines), 2)
            ]
            if not windows:
                continue
            best = max(
                windows,
                key=lambda window: (
                    sum(
                        math.log1p(len(self.pages) / self.frequency[word])
                        for word in sorted(words & terms(window) & self.frequency.keys())
                    )
                    - (5 if "Source:" in window else 0)
                ),
            )
            if best:
                excerpts.append(Page(page.number, best[:750]))
        return excerpts


def load_pdf(path: Path, data_dir: Path) -> tuple[Index, bool]:
    with path.open("rb") as handle:
        fingerprint = hashlib.file_digest(handle, "sha256").hexdigest()
    cache = data_dir / f"layout-v1-{pypdf.__version__}-{fingerprint}.json"
    hit = cache.exists()
    if hit:
        values = json.loads(cache.read_text(encoding="utf-8"))
    else:
        reader = pypdf.PdfReader(path)
        values = [page.extract_text(extraction_mode="layout") for page in reader.pages]
        # Replace atomically so concurrent readers never see an incomplete index.
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=data_dir, delete=False
        ) as handle:
            json.dump(values, handle, ensure_ascii=False)
        Path(handle.name).replace(cache)
    pages = [Page(i, value) for i, value in enumerate(values, 1) if value.strip()]
    if not pages:
        raise ValueError("No extractable text. This prototype requires a text PDF.")
    return Index(pages, fingerprint), hit
