"""Audit tracked source or reachable Git history without emitting sensitive text.

Private document-specific exclusions belong in an external JSON string array,
never in this public source file. This is a review aid, not a complete DLP system.
"""

import argparse
import json
import re
import subprocess
from pathlib import Path

SECRET = re.compile(
    rb"gsk_[A-Za-z0-9_-]{20,}|github_pat_[A-Za-z0-9_]{20,}|"
    rb"gh[pousr]_[A-Za-z0-9]{20,}|AKIA[A-Z0-9]{16}|"
    rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"
)
HOST_PATH = re.compile(rb"[A-Za-z]:[/\\]Users[/\\]|/(?:Users|home)/[^\s/]+/")
PRIVATE_SUFFIXES = {".pdf", ".docx", ".xlsx", ".gguf", ".sqlite", ".sqlite3", ".jsonl"}
PRIVATE_PARTS = {
    ".secrets",
    ".venv",
    "__pycache__",
    "query-records",
    "extractive-calls",
    "groq-verifier",
    "model-weights",
}


def git(*args):
    return subprocess.check_output(["git", *args])


def issues(path, raw, deny):
    reasons = []
    value = Path(path)
    if value.suffix.lower() in PRIVATE_SUFFIXES or set(value.parts) & PRIVATE_PARTS:
        reasons.append("private_artifact_path")
    if value.name.startswith(".env") and value.name not in {".env.example"}:
        reasons.append("environment_file")
    if SECRET.search(raw):
        reasons.append("credential_literal")
    if HOST_PATH.search(raw):
        reasons.append("user_specific_absolute_path")
    folded = raw.lower()
    if any(term in folded for term in deny):
        reasons.append("external_private_exclusion")
    if len(raw) > 1_000_000:
        reasons.append("large_file_requires_review")
    return reasons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", help="Audit all objects reachable from this ref")
    parser.add_argument("--deny-file", type=Path, help="External JSON array of strings")
    args = parser.parse_args()
    deny = []
    if args.deny_file:
        from evidence_guard.storage import outside_repository

        terms = json.loads(outside_repository(args.deny_file).read_text(encoding="utf-8"))
        if not isinstance(terms, list) or not all(
            isinstance(t, str) and t for t in terms
        ):
            parser.error("Deny file must contain a JSON array of nonempty strings.")
        deny = [t.lower().encode() for t in terms]
    findings = []
    checked = 0
    if args.revision:
        # --batch checks only reachable objects; unrelated private local refs are
        # intentionally excluded from the public publication candidate.
        objects = git("rev-list", "--objects", args.revision).decode().splitlines()
        names = {}
        for row in objects:
            oid, _, path = row.partition(" ")
            names.setdefault(oid, path)
        batch = subprocess.run(
            ["git", "cat-file", "--batch"],
            input=("\n".join(names) + "\n").encode(),
            capture_output=True,
            check=True,
        ).stdout
        offset = 0
        while offset < len(batch):
            end = batch.index(b"\n", offset)
            oid, kind, size = batch[offset:end].decode().split()
            start = end + 1
            raw = batch[start : start + int(size)]
            offset = start + int(size) + 1
            if kind not in {"blob", "commit", "tag"}:
                continue
            checked += 1
            path = names[oid] if kind == "blob" else "<" + kind + ">"
            # Git author emails/paths are not source files; private deny strings
            # and credential checks still apply to commit/tag metadata.
            found = issues(path, raw, deny)
            if found:
                findings.append({"path": path, "object": oid, "categories": found})
    else:
        root = Path(git("rev-parse", "--show-toplevel").decode().strip())
        for name in git("ls-files", "-z").decode().split("\0"):
            if not name:
                continue
            path = root / name
            if not path.is_file():
                continue
            checked += 1
            found = issues(name, path.read_bytes(), deny)
            if found:
                findings.append({"path": name, "categories": found})
    print(json.dumps({"checked": checked, "findings": findings}, indent=2))
    raise SystemExit(bool(findings))


if __name__ == "__main__":
    main()
