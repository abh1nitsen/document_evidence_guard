"""PDF ingestion boundaries and document identity, using unrelated synthetic files."""

from io import BytesIO

import pytest
from pypdf import PdfReader, PdfWriter
from pypdf.generic import (
    DecodedStreamObject,
    DictionaryObject,
    NameObject,
)

from evidence_guard.documents import DocumentStore


def pdf_bytes(text=None, encrypted=False):
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    if text is not None:
        font = DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )
        page[NameObject("/Resources")] = DictionaryObject(
            {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})}
        )
        escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        commands = (
            "BT /F1 12 Tf 50 740 Td "
            + " ".join(
                ("0 -18 Td " if n else "") + f"({line}) Tj"
                for n, line in enumerate(escaped.splitlines())
            )
            + " ET"
        )
        stream = DecodedStreamObject()
        stream.set_data(commands.encode("ascii"))
        page[NameObject("/Contents")] = writer._add_object(stream)
    if encrypted:
        writer.encrypt("synthetic-password")
    output = BytesIO()
    writer.write(output)
    return output.getvalue()


def test_upload_is_content_addressed_and_separate_from_filename(tmp_path):
    store = DocumentStore(tmp_path)
    raw = pdf_bytes("1.1 Storage\nThe retention period is three days.")
    first = store.upload(raw, "../../Policy.pdf")
    doc = first["document"]
    assert doc["name"] == "Policy.pdf"
    assert store.path(doc["id"]).parent == tmp_path / "documents"
    assert store.read_pdf(doc["id"]) == raw
    duplicate = store.upload(raw, "Another name.pdf")
    assert duplicate["duplicate"]
    assert len(store.list()) == 1


def test_distinct_pdfs_cannot_share_cached_evidence(tmp_path):
    store = DocumentStore(tmp_path)
    a = store.upload(pdf_bytes("Reserve is three days."), "A.pdf")["document"]
    b = store.upload(pdf_bytes("Reserve is nine days."), "B.pdf")["document"]
    ia, _ = store.get_index(a["id"])
    ib, _ = store.get_index(b["id"])
    assert a["id"] != b["id"]
    assert "three days" in ia.pages[0].text
    assert "nine days" not in ia.pages[0].text
    assert "nine days" in ib.pages[0].text
    with pytest.raises(ValueError):
        store.path("../A.pdf")


def test_scans_corruption_and_passwords_leave_library_unchanged(tmp_path):
    store = DocumentStore(tmp_path)
    store.upload(pdf_bytes("A readable document."), "Valid.pdf")
    with pytest.raises(ValueError, match="OCR"):
        store.upload(pdf_bytes(), "Scan.pdf")
    with pytest.raises(ValueError, match="Password"):
        store.upload(pdf_bytes("Secret", encrypted=True), "Locked.pdf")
    with pytest.raises(ValueError):
        store.upload(b"%PDF-1.7 broken file", "Broken.pdf")
    with pytest.raises(ValueError, match="PDF"):
        store.upload(b"some text", "Notes.txt")
    assert len(store.list()) == 1


def test_modified_stored_pdf_is_rejected(tmp_path):
    store = DocumentStore(tmp_path)
    doc = store.upload(pdf_bytes("Original content."), "A.pdf")["document"]
    store.path(doc["id"]).write_bytes(pdf_bytes("Changed content of a different length."))
    with pytest.raises(ValueError, match="changed"):
        store.get_index(doc["id"])
    with pytest.raises(ValueError, match="changed"):
        store.read_pdf(doc["id"])


def test_runtime_library_rejects_git_checkout(tmp_path):
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "HEAD").touch()
    with pytest.raises(ValueError, match="outside"):
        DocumentStore(tmp_path)


def test_nested_index_checkout_is_also_rejected(tmp_path):
    index = tmp_path / "documents" / "indexes"
    (index / ".git").mkdir(parents=True)
    (index / ".git" / "HEAD").touch()
    with pytest.raises(ValueError, match="outside"):
        DocumentStore(tmp_path)


def test_partial_scan_warns_and_preserves_actual_pdf_page_numbers(tmp_path):
    reader = PdfReader(BytesIO(pdf_bytes("Readable content on PDF page two.")))
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    writer.add_page(reader.pages[0])
    output = BytesIO()
    writer.write(output)
    store = DocumentStore(tmp_path)
    doc = store.upload(output.getvalue(), "Partial.pdf")["document"]
    index, _ = store.get_index(doc["id"])
    assert doc["pages"] == 2
    assert doc["searchable_pages"] == 1
    assert "OCR" in doc["warning"]
    assert index.pages[0].number == 2
