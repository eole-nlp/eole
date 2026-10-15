"""Bounded, local text extraction for translation imports (not layout conversion)."""

import base64
import binascii
from io import BytesIO
from pathlib import Path
import zipfile
from xml.etree import ElementTree

from fastapi import HTTPException
from pydantic import BaseModel, Field


class ImportDocument(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    data: str = Field(max_length=5600000)


def extract_document(name, data):
    try:
        raw = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("Invalid file encoding") from exc
    if len(raw) > 4000000:
        raise ValueError("Document limit: 4 MB")
    extension = Path(name).suffix.lower()
    if extension == ".docx":
        with zipfile.ZipFile(BytesIO(raw)) as archive:
            infos = archive.infolist()
            if len(infos) > 2000 or sum(i.file_size for i in infos) > 20000000:
                raise ValueError("DOCX archive exceeds the extraction limit")
            source = archive.read("word/document.xml")
        # No entity declarations or external references are needed in OOXML.
        if b"<!DOCTYPE" in source or b"<!ENTITY" in source:
            raise ValueError("XML entity declarations are unsupported")
        root = ElementTree.fromstring(source)
        ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
        paragraphs = []
        for paragraph in root.iter(ns + "p"):
            text = "".join(
                node.text or "" if node.tag == ns + "t" else "\t" if node.tag == ns + "tab" else "\n"
                for node in paragraph.iter()
                if node.tag in (ns + "t", ns + "tab", ns + "br")
            )
            if text.strip():
                paragraphs.append(text)
        text = "\n\n".join(paragraphs)
    elif extension == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise ValueError("PDF import requires the optional pypdf package: pip install 'eole[translation]'") from exc
        reader = PdfReader(BytesIO(raw))
        if reader.is_encrypted:
            raise ValueError("Unlock encrypted PDFs before importing")
        if len(reader.pages) > 50:
            raise ValueError("PDF import limit: 50 pages")
        text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
    else:
        raise ValueError("Choose a DOCX or text-based PDF document")
    if not text.strip():
        raise ValueError("No text found. Scanned documents need OCR before translation.")
    if len(text) > 80000:
        raise ValueError("Extracted text exceeds 80,000 characters; split the document")
    return text


def register_document_import(app):
    @app.post("/ui/translation/import")
    def import_document(request: ImportDocument):
        try:
            text = extract_document(request.name, request.data)
        except Exception as exc:
            raise HTTPException(422, str(exc)) from exc
        return {"text": text, "format": "text", "name": Path(request.name).stem + ".txt"}
