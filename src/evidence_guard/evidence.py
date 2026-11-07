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


def definition_target(question: str) -> str | None:
    match = re.search(
        r"\bwhat\s+(?:is\s+(?:the\s+)?([a-z][a-z0-9-]{1,15})|"
        r"([a-z][a-z0-9-]{1,15})\s+is)\s*(?:[?,.!]|$)|"
        r"\b(?:define|explain)\s+(?:the\s+)?([a-z][a-z0-9-]{1,15})"
        r"\s*(?:[?,.!]|$)|"
        r"\bwhat\s+does\s+([a-z][a-z0-9-]{1,15})\s+(?:mean|stand for)\b",
        question,
        re.IGNORECASE,
    )
    return (
        next((part.lower() for part in match.groups() if part), None) if match else None
    )


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

    def expansion(self, target: str) -> str | None:
        for page in self.pages:
            for line in page.text.splitlines():
                match = re.fullmatch(
                    rf"\s*{re.escape(target)}\s{{2,}}([A-Za-z][A-Za-z -]{{3,90}})\s*",
                    line,
                    re.IGNORECASE,
                )
                if match:
                    return match[1].strip()
        return None

    def definition(self, question: str) -> dict | None:
        """Resolve a glossary term and matching definition heading without generation."""
        target = definition_target(question)
        expansion = self.expansion(target) if target else None
        if not expansion:
            return None
        for page in self.pages:
            for paragraph in re.split(r"\n\s*\n", page.text):
                lines = paragraph.strip().splitlines()
                if len(lines) < 2:
                    continue
                heading = re.sub(r"^\d+(?:\.\d+)+\s+", "", lines[0].strip())
                if heading == lines[0].strip():
                    continue
                if not (
                    expansion.casefold() in heading.casefold()
                    and target in terms(heading)
                ):
                    continue
                body = " ".join(" ".join(lines[1:]).split())
                if len(body) > 1400 or not re.search(
                    rf"\b{re.escape(target)}\s+(?:is|means)\b", body, re.IGNORECASE
                ):
                    continue
                return {
                    "page": page.number,
                    "text": f"{expansion} ({target.upper()}). {body}",
                    "quote": " ".join(paragraph.split()),
                }
        return None

    def fetch(self, query: str, limit: int = 3) -> list[Page]:
        words = terms(query)
        target = definition_target(query)
        expansion = self.expansion(target) if target else None
        if expansion:
            words = terms(expansion) | {target}

        def definition_bonus(page: Page) -> float:
            if not target:
                return 0
            flat = " ".join(page.text.split()).lower()
            direct = bool(
                re.search(rf"\b{re.escape(target)}\)?\s+(?:is|means|stands for)\b", flat)
            )
            heading = any(
                re.search(r"^\s*\d+(?:\.\d+)+\s+", line)
                and target in terms(line)
                and len(line.strip()) < 110
                for line in page.text.splitlines()
            )
            return 10 * direct + 16 * heading

        scores = [
            (
                sum(
                    math.log1p(len(self.pages) / self.frequency[word])
                    for word in sorted(words & self.words[page.number])
                )
                + definition_bonus(page),
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
        target = definition_target(question)
        if target:
            words = {target}
        excerpts = []
        for page in pages:
            lines = page.text.splitlines()
            windows = [
                " ".join(" ".join(lines[i : i + 4]).split())
                for i in range(0, len(lines), 2)
            ]
            if target:
                paragraphs = re.split(r"\n\s*\n", page.text)
                headings = [
                    paragraph
                    for paragraph in paragraphs
                    if paragraph.strip()
                    and target in terms(paragraph.strip().splitlines()[0])
                    and re.match(r"\d+(?:\.\d+)+\s+", paragraph.strip())
                ]
                definitions = [
                    " ".join(paragraph.split())
                    for paragraph in headings or paragraphs
                    if re.search(rf"\b{re.escape(target)}\b", paragraph, re.IGNORECASE)
                    and len(" ".join(paragraph.split())) <= 1400
                ]
                if definitions:
                    windows = definitions
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
                    + (
                        12
                        if target
                        and re.search(
                            rf"^\d+(?:\.\d+)+\s+[^.]{{0,100}}\b{re.escape(target)}\b",
                            window,
                            re.IGNORECASE,
                        )
                        else 0
                    )
                    + (
                        8
                        if target
                        and re.search(
                            rf"\b{re.escape(target)}\)?\s+(?:is|means|stands for)\b",
                            window,
                            re.IGNORECASE,
                        )
                        else 0
                    )
                ),
            )
            if best:
                excerpts.append(Page(page.number, best[: 1400 if target else 750]))
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
