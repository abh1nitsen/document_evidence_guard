"""Public source checks must inspect historical objects without leaking content."""

import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_public_tree.py"


def git(directory, *args):
    return subprocess.run(
        ["git", *args],
        cwd=directory,
        capture_output=True,
        check=True,
    )


def repository(tmp_path):
    directory = tmp_path / "source"
    directory.mkdir()
    git(directory, "init", "-q")
    git(directory, "config", "user.name", "Synthetic Test")
    git(directory, "config", "user.email", "synthetic@example.invalid")
    return directory


def scan(directory, *args):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=directory,
        capture_output=True,
        text=True,
        check=False,
    )


def test_clean_source_and_removed_private_history_are_distinguished(tmp_path):
    directory = repository(tmp_path)
    (directory / "README.md").write_text("Synthetic source only.")
    git(directory, "add", ".")
    git(directory, "commit", "-qm", "Synthetic clean source")
    assert scan(directory, "--revision", "HEAD").returncode == 0
    (directory / "private.pdf").write_bytes(b"synthetic file, not an actual document")
    git(directory, "add", ".")
    git(directory, "commit", "-qm", "Synthetic private artifact")
    git(directory, "rm", "private.pdf")
    git(directory, "commit", "-qm", "Remove synthetic artifact")
    assert scan(directory).returncode == 0
    history = scan(directory, "--revision", "HEAD")
    assert history.returncode == 1
    assert any(
        "private_artifact_path" in f["categories"]
        for f in json.loads(history.stdout)["findings"]
    )


def test_secret_and_external_exclusion_content_is_never_printed(tmp_path):
    directory = repository(tmp_path)
    marker = "synthetic-customer-internal-marker"
    secret = "gsk_" + "x" * 28
    (directory / "config.txt").write_text(secret + "\n" + marker)
    git(directory, "add", ".")
    deny = tmp_path / "external-deny.json"
    deny.write_text(json.dumps([marker]))
    result = scan(directory, "--deny-file", str(deny))
    assert result.returncode == 1
    findings = json.loads(result.stdout)["findings"]
    assert {"credential_literal", "external_private_exclusion"} <= set(
        findings[0]["categories"]
    )
    assert secret not in result.stdout and marker not in result.stdout
