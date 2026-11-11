"""Local, versioned extraction and retrieval. Page IDs use PDF page order."""

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from difflib import get_close_matches
from functools import lru_cache
from pathlib import Path
from tempfile import NamedTemporaryFile

import pypdf
import snowballstemmer

STEMMER = snowballstemmer.stemmer("english")


@lru_cache(maxsize=16384)
def stem(word: str) -> str:
    return STEMMER.stemWord(word)


STOP = set(
    (
        "a an and are as at be by can did do does for from how i in is it me of "
        "on or report show tell that the this to was were what when where which why with"
    ).split()
)


def terms(value: str) -> set[str]:
    words = set(re.findall(r"\w+(?:\.\d+)*", value.casefold())) - STOP
    return {stem(word) for word in words}


def definition_target(question: str) -> str | None:
    question = question.strip().replace("_", " ")
    patterns = (
        r"\bwhat\s+(?:is|are)\s+(?:a\s+|an\s+|the\s+)?([^?.,]{1,90})",
        r"\bwhat\s+does\s+(.{1,80}?)\s+(?:mean|stand for)\b",
        r"\bwhat\s+(.{1,80}?)\s+is\s*(?:[?,.]|$)",
        r"\b(?:define|explain)\s+(?:a\s+|an\s+|the\s+)?([^?.,]{1,90})",
    )
    for pattern in patterns:
        match = re.search(pattern, question, re.IGNORECASE)
        if match:
            target = re.sub(
                r"(?:\s+(?:please|in simple words|to me|here|"
                r"in (?:this|the) (?:context|report|document|chapter)|"
                r"within (?:this|the) (?:context|report|document|chapter)))+\s*$",
                "",
                match[1].strip(),
                flags=re.IGNORECASE,
            )
            return target.casefold()
    return None


@dataclass(frozen=True)
class Page:
    number: int
    text: str


@dataclass(frozen=True)
class Passage:
    id: str
    page: int
    heading: str
    kind: str
    text: str
    section_start: bool = False


def normalize(value: str) -> str:
    return " ".join(value.replace("_", " ").split()).casefold()


def passages_from_pages(pages: list[Page]) -> list[Passage]:
    result = []
    heading = ""
    for page in pages:
        pending_section = False
        structured = re.sub(
            r"(?m)^(?=\d{1,2}(?:\.\d{1,2}){1,3}\s+[A-Z])",
            "\n\n",
            page.text,
        )
        blocks = [block for block in re.split(r"\n\s*\n", structured) if block.strip()]
        merged = []
        position = 0
        while position < len(blocks):
            block = blocks[position]
            if block.lstrip().startswith("Table "):
                while position + 1 < len(blocks):
                    following = blocks[position + 1]
                    column_lines = sum(
                        bool(re.search(r"\S {2,}\S", line))
                        for line in following.splitlines()
                    )
                    if column_lines < 1 and not following.lstrip().startswith("Source:"):
                        break
                    block += "\n\n" + following
                    position += 1
            merged.append(block)
            position += 1
        for block in merged:
            lines = block.strip().splitlines()
            if not lines or re.search(r"\.{8,}\s*\d+", block):
                continue
            first = lines[0].strip()
            numbered = re.match(r"^\d+(?:\.\d+)+\s+(.+)", first)
            if numbered:
                heading = numbered[1]
            elif re.match(r"^(?:Chapter \d+|Appendix [A-Z])\b", first) or first in {
                "Abstract",
                "Executive Summary",
                "Key takeaways",
                "References",
                "Abbreviations",
            }:
                heading = first
            kind = "paragraph"
            if first.startswith("Table "):
                kind = "table"
            elif first.startswith("Figure "):
                kind = "figure_caption"
            if len(lines) == 1 and (numbered or first == heading):
                pending_section = numbered is not None
                continue
            text = " ".join(block.split())
            section_start = (numbered is not None or pending_section) and len(
                text
            ) <= 1800
            pending_section = False
            start = 0
            while start < len(text):
                end = min(start + (3200 if kind == "table" else 1800), len(text))
                if end < len(text):
                    boundary = text.rfind(". ", start + 700, end)
                    if boundary >= 0:
                        end = boundary + 1
                    else:
                        boundary = text.rfind(" ", start, end)
                        if boundary > start:
                            end = boundary
                chunk = text[start:end].strip()
                digest = hashlib.sha256(chunk.encode()).hexdigest()[:12]
                result.append(
                    Passage(
                        f"p{page.number}-{len(result) + 1}-{digest}",
                        page.number,
                        first if kind != "paragraph" else heading,
                        kind,
                        chunk,
                        section_start and start == 0,
                    )
                )
                start = end
    return result


class Index:
    search_terms = staticmethod(terms)

    def __init__(self, pages: list[Page], fingerprint: str):
        self.pages = pages
        self.fingerprint = fingerprint
        self.words = {page.number: terms(page.text) for page in pages}
        self.frequency = Counter(word for words in self.words.values() for word in words)
        self.passages = passages_from_pages(pages)

    def outline(self) -> list[dict]:
        """Small document-wide map, separate from potentially misleading top hits."""
        entries = []
        for page in self.pages:
            for line in page.text.splitlines():
                title = line.strip()
                if re.match(r"^(?:Chapter \d+|Appendix [A-Z])\b", title):
                    if title not in {e["title"] for e in entries}:
                        entries.append({"page": page.number, "title": title[:90]})
        return entries[:14]

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

    def definitions(self, question: str) -> list[dict]:
        """Match explicit document definitions, including ordinary multiword terms."""
        target = definition_target(question)
        expansion = self.expansion(target) if target else None
        if not target:
            return []
        matches = []
        for passage in self.passages:
            exact = normalize(passage.heading) == normalize(target)
            acronym = expansion and (
                normalize(expansion) in normalize(passage.heading)
                and target in terms(passage.heading)
            )
            if (
                not (exact or acronym)
                or passage.kind != "paragraph"
                or not passage.section_start
            ):
                continue
            if not re.search(
                r"\b(?:is|are|refers to|means|denotes|describes|designation)\b",
                passage.text,
                re.IGNORECASE,
            ):
                continue
            text = re.sub(
                r"^\d+(?:\.\d+)+\s+" + re.escape(passage.heading) + r"\s*",
                "",
                passage.text,
            )
            if expansion:
                text = f"{expansion} ({target.upper()}). {text}"
            matches.append(
                {
                    "page": passage.page,
                    "text": text,
                    "quote": passage.text,
                    "passage_id": passage.id,
                    "heading": passage.heading,
                }
            )
        return matches

    def definition(self, question: str) -> dict | None:
        matches = self.definitions(question)
        return matches[0] if len(matches) == 1 else None

    def suggestions(self, target: str) -> list[str]:
        headings = {
            normalize(p.heading): p.heading
            for p in self.passages
            if 2 <= len(p.heading) <= 80 and normalize(p.heading) != normalize(target)
        }
        close = get_close_matches(normalize(target), sorted(headings), n=2, cutoff=0.86)
        return [headings[key] for key in close]

    def search(self, query: str, limit: int = 8) -> list[Passage]:
        """BM25-style paragraph retrieval with glossary aliases and heading boosts."""
        query_terms = self.search_terms(query)
        for token in list(query_terms):
            expansion = self.expansion(token)
            if expansion:
                query_terms |= self.search_terms(expansion)
        tokenized = [self.search_terms(p.text) for p in self.passages]
        frequency = Counter(word for words in tokenized for word in words)
        average = sum(len(words) for words in tokenized) / max(1, len(tokenized))
        target = definition_target(query)
        ranked = []
        for passage, words in zip(self.passages, tokenized, strict=True):
            matched = words & query_terms
            score = sum(
                math.log1p(
                    (len(tokenized) - frequency[word] + 0.5) / (frequency[word] + 0.5)
                )
                * 2.2
                / (1 + 1.2 * (0.25 + 0.75 * len(words) / max(1, average)))
                for word in matched
            )
            score += 2 * len(self.search_terms(passage.heading) & query_terms)
            if passage.kind == "figure_caption" or passage.text.startswith("Source:"):
                score *= 0.25
            if passage.heading in {"Abbreviations", "References"}:
                score *= 0.25
            if target and normalize(target) == normalize(passage.heading):
                score += 20
            if matched:
                ranked.append((score, passage))
        ranked.sort(key=lambda item: (-item[0], item[1].page, item[1].id))
        return [passage for _, passage in ranked[:limit]]

    def search_queries(
        self, original: str, rewrites: list[str], limit: int
    ) -> list[Passage]:
        scores = Counter()
        passages = {}
        original_hits = self.search(original, limit)
        for query in dict.fromkeys([original] + rewrites):
            for rank, passage in enumerate(self.search(query, limit), 1):
                scores[passage.id] += 1 / (60 + rank)
                passages[passage.id] = passage
        ordered = sorted(
            passages, key=lambda key: (-scores[key], passages[key].page, key)
        )
        chosen = [passages[key] for key in ordered[:limit]]
        if original_hits and original_hits[0].id not in {p.id for p in chosen}:
            chosen[-1] = original_hits[0]
        return chosen

    def contextual_candidates(self, ranked: list[Passage], limit: int) -> list[Passage]:
        """Keep nearby prose around the two strongest hits within the same section."""
        positions = {p.id: i for i, p in enumerate(self.passages)}
        chosen = list(ranked[:2])
        for anchor in ranked[:2]:
            position = positions[anchor.id]
            for direction in (-1, 1):
                if direction == -1 and anchor.section_start:
                    continue
                for step in (1, 2):
                    offset = position + direction * step
                    if not 0 <= offset < len(self.passages):
                        break
                    candidate = self.passages[offset]
                    if abs(candidate.page - anchor.page) > 1:
                        break
                    if direction == 1 and candidate.section_start:
                        break
                    if candidate.kind == "paragraph" and not candidate.text.startswith(
                        "Source:"
                    ):
                        chosen.append(candidate)
                    if direction == -1 and candidate.section_start:
                        break
        chosen.extend(ranked)
        return list({p.id: p for p in chosen}.values())[:limit]

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
        values = [
            page.extract_text(extraction_mode="layout") or "" for page in reader.pages
        ]
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
