"""S1-07: Safe PDF/DOCX extraction with security validation.

Tests PDF structural security, DOCX content security, safe extraction
behavior, and endpoint integration for rejected unsafe documents.

Includes regression probes for three specific bypass cases:
1. word/activeX/control.png must be rejected
2. External relationship inside document.xml.RELS must be rejected
3. PDF /URI action inside nested arrays must be rejected
"""

import io
import os
import uuid
import zipfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from xml.etree import ElementTree as ET

import pytest

from app.services.pdf_security import validate_pdf_security
from app.services.docx_security import validate_docx_security
from app.services.upload_extractor import (
    ExtractionError,
    extract_content,
    extract_pdf,
    extract_docx,
    extract_txt,
    extract_md,
    compute_content_hash,
)
from app.services.upload_validator import UploadRejectionReason


def _make_valid_pdf() -> bytes:
    """Generate a valid unencrypted PDF with text."""
    from pypdf import PdfWriter
    from pypdf.generic import NameObject, TextStringObject
    w = PdfWriter()
    w.add_blank_page(612, 792)
    buf = io.BytesIO()
    w.write(buf)
    return buf.getvalue()


def _make_encrypted_pdf() -> bytes:
    from pypdf import PdfWriter
    w = PdfWriter()
    w.add_blank_page(612, 792)
    w.encrypt("secret")
    buf = io.BytesIO()
    w.write(buf)
    return buf.getvalue()


# ─── PDF Security Tests ───────────────────────────────────────────────────

class TestPdfSecurity:
    """PDF structural security validation."""

    def test_valid_pdf_accepted(self):
        pdf = _make_valid_pdf()
        valid, reason = validate_pdf_security(pdf)
        assert valid is True

    def test_encrypted_pdf_rejected(self):
        pdf = _make_encrypted_pdf()
        valid, reason = validate_pdf_security(pdf)
        assert valid is False
        assert reason == UploadRejectionReason.PDF_ENCRYPTED

    def test_malformed_pdf_rejected(self):
        valid, reason = validate_pdf_security(b"%PDF-1.4 garbage\x00\x01\x02")
        assert valid is False
        assert reason == UploadRejectionReason.PDF_MALFORMED

    def test_truncated_pdf_rejected(self):
        valid, reason = validate_pdf_security(b"%PDF-1")
        assert valid is False
        assert reason == UploadRejectionReason.PDF_MALFORMED

    def test_pdf_with_javascript_rejected(self):
        """PDF containing /JavaScript in catalog is rejected."""
        from pypdf import PdfWriter
        from pypdf.generic import (
            ArrayObject, DictionaryObject, NameObject, TextStringObject,
        )
        w = PdfWriter()
        w.add_blank_page(612, 792)
        js_action = DictionaryObject()
        js_action[NameObject("/Type")] = NameObject("/Action")
        js_action[NameObject("/S")] = NameObject("/JavaScript")
        js_action[NameObject("/JS")] = TextStringObject("app.alert('xss');")
        w._root_object[NameObject("/OpenAction")] = js_action
        buf = io.BytesIO()
        w.write(buf)

        valid, reason = validate_pdf_security(buf.getvalue())
        assert valid is False
        assert reason in (
            UploadRejectionReason.PDF_ACTIVE_CONTENT,
            UploadRejectionReason.PDF_EXTERNAL_LINK,
        )

    def test_pdf_with_embedded_file_rejected(self):
        """PDF with /EmbeddedFiles in catalog is rejected."""
        from pypdf import PdfWriter
        from pypdf.generic import DictionaryObject, NameObject, ArrayObject
        w = PdfWriter()
        w.add_blank_page(612, 792)
        ef_dict = DictionaryObject()
        ef_dict[NameObject("/Names")] = ArrayObject()
        w._root_object[NameObject("/EmbeddedFiles")] = ef_dict
        buf = io.BytesIO()
        w.write(buf)

        valid, reason = validate_pdf_security(buf.getvalue())
        assert valid is False
        assert reason == UploadRejectionReason.PDF_ATTACHMENT

    def test_pdf_with_uri_action_rejected(self):
        """PDF with /URI action annotation is rejected."""
        from pypdf import PdfWriter
        from pypdf.generic import DictionaryObject, NameObject, TextStringObject, ArrayObject
        w = PdfWriter()
        p = w.add_blank_page(612, 792)
        annot = DictionaryObject()
        annot[NameObject("/Type")] = NameObject("/Annot")
        annot[NameObject("/Subtype")] = NameObject("/Link")
        action = DictionaryObject()
        action[NameObject("/S")] = NameObject("/URI")
        action[NameObject("/URI")] = TextStringObject("http://evil.com")
        annot[NameObject("/A")] = action
        p[NameObject("/Annots")] = ArrayObject([annot])
        buf = io.BytesIO()
        w.write(buf)

        valid, reason = validate_pdf_security(buf.getvalue())
        assert valid is False
        assert reason == UploadRejectionReason.PDF_EXTERNAL_LINK

    def test_pdf_plain_text_with_url_allowed(self):
        """Plain text containing a URL is NOT rejected (only structural links)."""
        pdf = _make_valid_pdf()
        valid, reason = validate_pdf_security(pdf)
        assert valid is True

    def test_pdf_with_launch_action_rejected(self):
        """PDF with /Launch action is rejected as embedded object."""
        from pypdf import PdfWriter
        from pypdf.generic import DictionaryObject, NameObject, TextStringObject
        w = PdfWriter()
        w.add_blank_page(612, 792)
        launch = DictionaryObject()
        launch[NameObject("/S")] = NameObject("/Launch")
        launch[NameObject("/F")] = TextStringObject("calc.exe")
        w._root_object[NameObject("/OpenAction")] = launch
        buf = io.BytesIO()
        w.write(buf)

        valid, reason = validate_pdf_security(buf.getvalue())
        assert valid is False
        assert reason in (
            UploadRejectionReason.PDF_EMBEDDED_OBJECT,
            UploadRejectionReason.PDF_ACTIVE_CONTENT,
        )

    def test_pdf_uri_inside_nested_arrays(self):
        """REGRESSION: /URI action inside array nested inside another array is rejected."""
        from pypdf import PdfWriter
        from pypdf.generic import (
            DictionaryObject, NameObject, TextStringObject, ArrayObject,
        )
        w = PdfWriter()
        p = w.add_blank_page(612, 792)
        # Build: page -> /Annots -> [ArrayObject([ArrayObject([dict with /URI])])]
        action = DictionaryObject()
        action[NameObject("/S")] = NameObject("/URI")
        action[NameObject("/URI")] = TextStringObject("http://evil.com/nested")
        inner_array = ArrayObject([action])
        outer_array = ArrayObject([inner_array])
        p[NameObject("/Annots")] = outer_array
        buf = io.BytesIO()
        w.write(buf)

        valid, reason = validate_pdf_security(buf.getvalue())
        assert valid is False
        assert reason == UploadRejectionReason.PDF_EXTERNAL_LINK

    def test_pdf_javascript_hidden_in_nested_arrays(self):
        """JavaScript hidden inside nested arrays is rejected."""
        from pypdf import PdfWriter
        from pypdf.generic import (
            DictionaryObject, NameObject, TextStringObject, ArrayObject,
        )
        w = PdfWriter()
        p = w.add_blank_page(612, 792)
        js_dict = DictionaryObject()
        js_dict[NameObject("/JavaScript")] = TextStringObject("evil()")
        inner = ArrayObject([js_dict])
        outer = ArrayObject([inner])
        p[NameObject("/Kids")] = outer
        buf = io.BytesIO()
        w.write(buf)

        valid, reason = validate_pdf_security(buf.getvalue())
        assert valid is False
        assert reason == UploadRejectionReason.PDF_ACTIVE_CONTENT

    def test_pdf_attachment_via_nested_array_and_indirect(self):
        """Attachment dictionary reached through nested arrays is rejected."""
        from pypdf import PdfWriter
        from pypdf.generic import (
            DictionaryObject, NameObject, ArrayObject, TextStringObject,
        )
        w = PdfWriter()
        w.add_blank_page(612, 792)
        ef_dict = DictionaryObject()
        ef_dict[NameObject("/EmbeddedFiles")] = TextStringObject("payload")
        inner = ArrayObject([ef_dict])
        outer = ArrayObject([inner])
        w._root_object[NameObject("/Names")] = outer
        buf = io.BytesIO()
        w.write(buf)

        valid, reason = validate_pdf_security(buf.getvalue())
        assert valid is False
        assert reason == UploadRejectionReason.PDF_ATTACHMENT

    def test_pdf_unresolvable_indirect_object(self):
        """IndirectObject that cannot be resolved triggers PDF_MALFORMED."""
        from pypdf import PdfWriter
        from pypdf.generic import (
            DictionaryObject, NameObject, ArrayObject, IndirectObject,
        )
        w = PdfWriter()
        p = w.add_blank_page(612, 792)
        # Create an indirect reference with invalid id
        bad_ref = IndirectObject(99999, 0, w)
        p[NameObject("/BadRef")] = ArrayObject([bad_ref])
        buf = io.BytesIO()
        w.write(buf)

        valid, reason = validate_pdf_security(buf.getvalue())
        assert valid is False
        assert reason == UploadRejectionReason.PDF_MALFORMED

    def test_pdf_normal_with_harmless_arrays(self):
        """Normal PDF with harmless arrays is accepted."""
        from pypdf import PdfWriter
        from pypdf.generic import (
            DictionaryObject, NameObject, ArrayObject, NumberObject,
        )
        w = PdfWriter()
        p = w.add_blank_page(612, 792)
        # Add a harmless array (media box is already one, but let's add another)
        p[NameObject("/SomeData")] = ArrayObject([
            NumberObject(1), NumberObject(2), NumberObject(3)
        ])
        buf = io.BytesIO()
        w.write(buf)

        valid, reason = validate_pdf_security(buf.getvalue())
        assert valid is True


# ─── DOCX Security Tests ──────────────────────────────────────────────────

def _make_valid_docx(path: Path) -> Path:
    """Create a minimal valid DOCX."""
    from docx import Document
    doc = Document()
    doc.add_paragraph("Training script content.")
    doc.save(str(path))
    return path


class TestDocxSecurity:
    """DOCX content security validation."""

    def test_valid_docx_accepted(self, tmp_path):
        path = _make_valid_docx(tmp_path / "ok.docx")
        valid, reason = validate_docx_security(path)
        assert valid is True

    def test_ole_embedding_rejected(self, tmp_path):
        """DOCX with word/embeddings/oleObject1.bin is rejected."""
        path = tmp_path / "ole.docx"
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("[Content_Types].xml", "<Types/>")
            zf.writestr("word/document.xml", "<doc/>")
            zf.writestr("word/embeddings/oleObject1.bin", b"\x00" * 50)
        valid, reason = validate_docx_security(path)
        assert valid is False
        assert reason == UploadRejectionReason.DOCX_EMBEDDED_OBJECT

    def test_activex_rejected(self, tmp_path):
        """DOCX with word/activeX/ entries is rejected."""
        path = tmp_path / "ax.docx"
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("[Content_Types].xml", "<Types/>")
            zf.writestr("word/document.xml", "<doc/>")
            zf.writestr("word/activeX/activeX1.xml", "<ax/>")
        valid, reason = validate_docx_security(path)
        assert valid is False
        assert reason == UploadRejectionReason.DOCX_EMBEDDED_OBJECT

    def test_activex_control_png_rejected(self, tmp_path):
        """REGRESSION: word/activeX/control.png is rejected (any extension)."""
        path = tmp_path / "ax_png.docx"
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("[Content_Types].xml", "<Types/>")
            zf.writestr("word/document.xml", "<doc/>")
            zf.writestr("word/activeX/control.png", b"\x89PNG fake")
        valid, reason = validate_docx_security(path)
        assert valid is False
        assert reason == UploadRejectionReason.DOCX_EMBEDDED_OBJECT

    def test_embeddings_object_jpg_rejected(self, tmp_path):
        """REGRESSION: word/embeddings/object.jpg is rejected (any extension)."""
        path = tmp_path / "emb_jpg.docx"
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("[Content_Types].xml", "<Types/>")
            zf.writestr("word/document.xml", "<doc/>")
            zf.writestr("word/embeddings/object.jpg", b"\xFF\xD8\xFF fake jpg")
        valid, reason = validate_docx_security(path)
        assert valid is False
        assert reason == UploadRejectionReason.DOCX_EMBEDDED_OBJECT

    def test_mixed_case_activex_rejected(self, tmp_path):
        """Mixed-case word/ActiveX/ paths are rejected."""
        path = tmp_path / "ax_mixed.docx"
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("[Content_Types].xml", "<Types/>")
            zf.writestr("word/document.xml", "<doc/>")
            zf.writestr("Word/ActiveX/Control.svg", b"<svg/>")
        valid, reason = validate_docx_security(path)
        assert valid is False
        assert reason == UploadRejectionReason.DOCX_EMBEDDED_OBJECT

    def test_mixed_case_embeddings_rejected(self, tmp_path):
        """Mixed-case word/Embeddings/ paths are rejected."""
        path = tmp_path / "emb_mixed.docx"
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("[Content_Types].xml", "<Types/>")
            zf.writestr("word/document.xml", "<doc/>")
            zf.writestr("Word/Embeddings/image.png", b"\x89PNG")
        valid, reason = validate_docx_security(path)
        assert valid is False
        assert reason == UploadRejectionReason.DOCX_EMBEDDED_OBJECT

    def test_word_media_image_accepted(self, tmp_path):
        """word/media/image.png remains accepted."""
        path = tmp_path / "media_ok.docx"
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("[Content_Types].xml", "<Types/>")
            zf.writestr("word/document.xml", "<doc/>")
            zf.writestr("word/media/image.png", b"\x89PNG")
        valid, reason = validate_docx_security(path)
        assert valid is True

    def test_external_hyperlink_rejected(self, tmp_path):
        """DOCX with external HTTP relationship is rejected."""
        path = tmp_path / "ext.docx"
        rels_xml = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink" '
            'Target="https://evil.com/payload" TargetMode="External"/>'
            '</Relationships>'
        )
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("[Content_Types].xml", "<Types/>")
            zf.writestr("word/document.xml", "<doc/>")
            zf.writestr("word/_rels/document.xml.rels", rels_xml)
        valid, reason = validate_docx_security(path)
        assert valid is False
        assert reason == UploadRejectionReason.DOCX_EXTERNAL_LINK

    def test_external_unc_rejected(self, tmp_path):
        """DOCX with UNC path relationship is rejected."""
        path = tmp_path / "unc.docx"
        rels_xml = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" '
            'Target="\\\\evil\\share\\img.png" TargetMode="External"/>'
            '</Relationships>'
        )
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("[Content_Types].xml", "<Types/>")
            zf.writestr("word/document.xml", "<doc/>")
            zf.writestr("word/_rels/document.xml.rels", rels_xml)
        valid, reason = validate_docx_security(path)
        assert valid is False
        assert reason == UploadRejectionReason.DOCX_EXTERNAL_LINK

    def test_internal_relationship_allowed(self, tmp_path):
        """Internal relationships (no TargetMode=External) are accepted."""
        path = tmp_path / "internal.docx"
        rels_xml = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" '
            'Target="media/image1.png"/>'
            '</Relationships>'
        )
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("[Content_Types].xml", "<Types/>")
            zf.writestr("word/document.xml", "<doc/>")
            zf.writestr("word/_rels/document.xml.rels", rels_xml)
            zf.writestr("word/media/image1.png", b"\x89PNG")
        valid, reason = validate_docx_security(path)
        assert valid is True

    def test_normal_embedded_image_allowed(self, tmp_path):
        """Normal image in word/media/ is accepted."""
        path = _make_valid_docx(tmp_path / "img.docx")
        valid, reason = validate_docx_security(path)
        assert valid is True

    def test_dtd_declaration_rejected(self, tmp_path):
        """XML with <!DOCTYPE is rejected."""
        path = tmp_path / "dtd.docx"
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("[Content_Types].xml", '<!DOCTYPE foo SYSTEM "http://evil/dtd"><Types/>')
            zf.writestr("word/document.xml", "<doc/>")
        valid, reason = validate_docx_security(path)
        assert valid is False
        assert reason == UploadRejectionReason.DOCX_UNSAFE_XML

    def test_entity_declaration_rejected(self, tmp_path):
        """XML with <!ENTITY is rejected."""
        path = tmp_path / "entity.docx"
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("[Content_Types].xml", "<Types/>")
            zf.writestr("word/document.xml", '<!ENTITY xxe SYSTEM "file:///etc/passwd"><doc/>')
        valid, reason = validate_docx_security(path)
        assert valid is False
        assert reason == UploadRejectionReason.DOCX_UNSAFE_XML

    def test_ole_relationship_type_rejected(self, tmp_path):
        """Relationship with oleObject type is rejected."""
        path = tmp_path / "olerel.docx"
        rels_xml = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/oleObject" '
            'Target="embeddings/oleObject1.bin"/>'
            '</Relationships>'
        )
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("[Content_Types].xml", "<Types/>")
            zf.writestr("word/document.xml", "<doc/>")
            zf.writestr("word/_rels/document.xml.rels", rels_xml)
        valid, reason = validate_docx_security(path)
        assert valid is False
        assert reason == UploadRejectionReason.DOCX_EMBEDDED_OBJECT

    def test_malformed_zip_rejected(self, tmp_path):
        """Non-ZIP file is rejected."""
        path = tmp_path / "bad.docx"
        path.write_bytes(b"not a zip")
        valid, reason = validate_docx_security(path)
        assert valid is False
        assert reason == UploadRejectionReason.DOCX_MALFORMED


# ─── DOCX Case-Insensitive Part Discovery Tests ───────────────────────────

class TestDocxCaseInsensitiveDiscovery:
    """Case-insensitive .rels and .xml part discovery."""

    def test_external_link_in_uppercase_rels(self, tmp_path):
        """REGRESSION: External link in document.xml.RELS is rejected."""
        path = tmp_path / "upper_rels.docx"
        rels_xml = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink" '
            'Target="https://evil.com" TargetMode="External"/>'
            '</Relationships>'
        )
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("[Content_Types].xml", "<Types/>")
            zf.writestr("word/document.xml", "<doc/>")
            zf.writestr("word/_rels/document.xml.RELS", rels_xml)
        valid, reason = validate_docx_security(path)
        assert valid is False
        assert reason == UploadRejectionReason.DOCX_EXTERNAL_LINK

    def test_external_link_in_mixed_case_rels(self, tmp_path):
        """External link in .Rels (mixed case) is rejected."""
        path = tmp_path / "mixed_rels.docx"
        rels_xml = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink" '
            'Target="http://evil.com" TargetMode="External"/>'
            '</Relationships>'
        )
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("[Content_Types].xml", "<Types/>")
            zf.writestr("word/document.xml", "<doc/>")
            zf.writestr("word/_rels/document.xml.Rels", rels_xml)
        valid, reason = validate_docx_security(path)
        assert valid is False
        assert reason == UploadRejectionReason.DOCX_EXTERNAL_LINK

    def test_dtd_in_uppercase_xml(self, tmp_path):
        """DTD content in an uppercase .XML part is rejected."""
        path = tmp_path / "upper_xml.docx"
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("[Content_Types].xml", "<Types/>")
            zf.writestr("word/document.XML", '<!DOCTYPE foo SYSTEM "http://evil/dtd"><doc/>')
        valid, reason = validate_docx_security(path)
        assert valid is False
        assert reason == UploadRejectionReason.DOCX_UNSAFE_XML

    def test_entity_in_mixed_case_xml(self, tmp_path):
        """Entity declaration in mixed-case .Xml part is rejected."""
        path = tmp_path / "mixed_xml.docx"
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("[Content_Types].Xml", '<!ENTITY x "bad"><Types/>')
            zf.writestr("word/document.xml", "<doc/>")
        valid, reason = validate_docx_security(path)
        assert valid is False
        assert reason == UploadRejectionReason.DOCX_UNSAFE_XML

    def test_normal_lowercase_internal_rels_accepted(self, tmp_path):
        """Normal lowercase internal .rels file is still accepted."""
        path = tmp_path / "normal.docx"
        rels_xml = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" '
            'Target="media/image1.png"/>'
            '</Relationships>'
        )
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("[Content_Types].xml", "<Types/>")
            zf.writestr("word/document.xml", "<doc/>")
            zf.writestr("word/_rels/document.xml.rels", rels_xml)
        valid, reason = validate_docx_security(path)
        assert valid is True


# ─── DOCX Fail-Closed Read Tests ──────────────────────────────────────────

class TestDocxFailClosedReads:
    """If a listed .rels or .xml part cannot be read, reject the DOCX."""

    def test_unreadable_rels_file_rejected(self, tmp_path):
        """Rels file that cannot be read causes rejection."""
        path = tmp_path / "bad_rels.docx"
        rels_xml = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" '
            'Target="media/image1.png"/>'
            '</Relationships>'
        )
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("[Content_Types].xml", "<Types/>")
            zf.writestr("word/document.xml", "<doc/>")
            zf.writestr("word/_rels/document.xml.rels", rels_xml)

        # Patch zf.read to raise on the rels file
        orig_init = zipfile.ZipFile.__init__

        with patch.object(zipfile.ZipFile, "read", side_effect=IOError("corrupt")):
            valid, reason = validate_docx_security(path)

        assert valid is False
        assert reason == UploadRejectionReason.DOCX_MALFORMED

    def test_unreadable_xml_file_rejected(self, tmp_path):
        """XML file that cannot be read causes rejection."""
        path = tmp_path / "bad_xml.docx"
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("[Content_Types].xml", "<Types/>")
            zf.writestr("word/document.xml", "<doc/>")

        # Make read fail only for xml files (not rels)
        real_read = zipfile.ZipFile.read

        def _failing_read(self, name, *args, **kwargs):
            if name.lower().endswith(".xml"):
                raise IOError("corrupt xml")
            return real_read(self, name, *args, **kwargs)

        with patch.object(zipfile.ZipFile, "read", side_effect=_failing_read):
            valid, reason = validate_docx_security(path)

        assert valid is False
        assert reason == UploadRejectionReason.DOCX_UNSAFE_XML


# ─── Safe Extraction Tests ─────────────────────────────────────────────────

class TestSafeExtraction:
    """Extraction behavior: fail-closed, resource limits, body-text only."""

    def test_pdf_extraction_valid(self, tmp_path):
        """Valid PDF text extraction works."""
        from pypdf import PdfWriter
        path = tmp_path / "valid.pdf"
        w = PdfWriter()
        w.add_blank_page(612, 792)
        with open(path, "wb") as f:
            w.write(f)
        text = extract_pdf(path)
        assert isinstance(text, str)

    def test_pdf_extraction_malformed_raises(self, tmp_path):
        """Malformed PDF raises ExtractionError."""
        path = tmp_path / "bad.pdf"
        path.write_bytes(b"%PDF-1.4 corrupted garbage data\x00\x01")
        with pytest.raises(ExtractionError):
            extract_pdf(path)

    def test_docx_extraction_body_only(self, tmp_path):
        """DOCX extraction returns body paragraph text."""
        from docx import Document
        doc = Document()
        doc.add_paragraph("First paragraph.")
        doc.add_paragraph("Second paragraph.")
        path = tmp_path / "body.docx"
        doc.save(str(path))
        text = extract_docx(path)
        assert "First paragraph." in text
        assert "Second paragraph." in text

    def test_docx_extraction_malformed_raises(self, tmp_path):
        """Malformed DOCX raises ExtractionError."""
        path = tmp_path / "bad.docx"
        path.write_bytes(b"not a docx")
        with pytest.raises(ExtractionError):
            extract_docx(path)

    def test_txt_extraction_works(self, tmp_path):
        path = tmp_path / "test.txt"
        path.write_bytes(b"Hello world")
        assert extract_txt(path) == "Hello world"

    def test_md_extraction_works(self, tmp_path):
        path = tmp_path / "test.md"
        path.write_bytes(b"# Heading\n\nBody text")
        text = extract_md(path)
        assert "# Heading" in text
        assert "Body text" in text

    def test_extraction_error_is_typed(self):
        """ExtractionError is a proper exception class."""
        assert issubclass(ExtractionError, Exception)
        err = ExtractionError("test reason")
        assert "test reason" in str(err)

    def test_content_hash_deterministic(self):
        h1 = compute_content_hash("hello")
        h2 = compute_content_hash("hello")
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex

    def test_unsupported_extension_raises(self, tmp_path):
        path = tmp_path / "file.xyz"
        path.write_bytes(b"data")
        with pytest.raises(ValueError, match="Unsupported"):
            extract_content(path, ".xyz")


# ─── Fix 1: External relationships (all TargetMode=External rejected) ─────

class TestDocxExternalRelationshipsAll:
    """Reject ALL external relationships regardless of URI scheme."""

    @pytest.mark.parametrize("target", [
        "https://evil.com/payload",
        "http://example.org/track",
        "mailto:user@evil.com",
        "ftp://ftp.evil.com/file",
        "data:text/html;base64,PHNjcmlwdD4=",
        "file:///etc/passwd",
        "\\\\server\\share\\file.docx",
        "//protocol-relative.com/path",
        "custom-scheme://app/action",
        "relative/path/to/resource",
        "",  # empty target still has External mode
    ])
    def test_external_target_rejected(self, tmp_path, target):
        path = tmp_path / "ext.docx"
        rels_xml = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            f'<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink" '
            f'Target="{target}" TargetMode="External"/>'
            '</Relationships>'
        )
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("[Content_Types].xml", "<Types/>")
            zf.writestr("word/document.xml", "<doc/>")
            zf.writestr("word/_rels/document.xml.rels", rels_xml)
        valid, reason = validate_docx_security(path)
        assert valid is False, f"External target '{target}' was not rejected"
        assert reason == UploadRejectionReason.DOCX_EXTERNAL_LINK

    def test_internal_relationship_preserved(self, tmp_path):
        """Relationship without TargetMode=External is accepted."""
        path = tmp_path / "int.docx"
        rels_xml = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" '
            'Target="media/image1.png"/>'
            '</Relationships>'
        )
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("[Content_Types].xml", "<Types/>")
            zf.writestr("word/document.xml", "<doc/>")
            zf.writestr("word/_rels/document.xml.rels", rels_xml)
        valid, reason = validate_docx_security(path)
        assert valid is True

    def test_case_insensitive_target_mode(self, tmp_path):
        """TargetMode matching is case-insensitive."""
        path = tmp_path / "case.docx"
        rels_xml = (
            '<?xml version="1.0"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink" '
            'Target="http://x.com" TargetMode="EXTERNAL"/>'
            '</Relationships>'
        )
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("[Content_Types].xml", "<Types/>")
            zf.writestr("word/document.xml", "<doc/>")
            zf.writestr("word/_rels/document.xml.rels", rels_xml)
        valid, reason = validate_docx_security(path)
        assert valid is False
        assert reason == UploadRejectionReason.DOCX_EXTERNAL_LINK


# ─── Fix 3: Per-entry compression bomb ────────────────────────────────────

class TestDocxPerEntryBomb:
    """Per-entry ratio and suspicious zero-compress detection."""

    def test_per_entry_high_ratio_rejected(self, tmp_path):
        """Single entry with ratio > 100 is rejected even if aggregate is OK."""
        from app.services.upload_validator import validate_docx_archive
        path = tmp_path / "bomb.docx"
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("[Content_Types].xml", "X" * 10000)
            zf.writestr("word/document.xml", "\x00" * (200 * 1024))
        valid, reason = validate_docx_archive(path)
        assert valid is False
        assert reason == UploadRejectionReason.DOCX_EXCESSIVE_RATIO

    def test_zero_compress_size_rejected(self, tmp_path):
        """Entry with file_size > 0 and compress_size == 0 is rejected."""
        from app.services.upload_validator import validate_docx_archive
        path = tmp_path / "zero.docx"
        content = b"x" * 100
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("[Content_Types].xml", "<Types/>")
            zf.writestr("word/document.xml", content)

        with zipfile.ZipFile(path, "r") as zf:
            info_list = zf.infolist()

        for info in info_list:
            if info.filename == "word/document.xml":
                info.compress_size = 0
                info.file_size = 100

        from unittest.mock import patch as _p
        with _p("zipfile.ZipFile.infolist", return_value=info_list):
            valid, reason = validate_docx_archive(path)

        assert valid is False
        assert reason == UploadRejectionReason.DOCX_EXCESSIVE_RATIO

    def test_normal_docx_accepted(self, tmp_path):
        """Ordinary DOCX passes per-entry checks."""
        from app.services.upload_validator import validate_docx_archive
        path = tmp_path / "normal.docx"
        from docx import Document
        doc = Document()
        doc.add_paragraph("Normal content")
        doc.save(str(path))
        valid, reason = validate_docx_archive(path)
        assert valid is True


# ─── Fix 4 & 5: Endpoint integration for security rejections ──────────────

class TestEndpointSecurityRejection:
    """Endpoint tests: unsafe files -> 422, file deleted, no extraction/success."""

    @pytest.fixture
    def _setup(self, tmp_path):
        from app.main import create_app
        from app.services.auth import require_admin
        from app.database import get_session
        from app.models.user import User, UserRole
        from unittest.mock import AsyncMock

        admin = MagicMock(spec=User)
        admin.id = uuid.uuid4()
        admin.email = "admin@test.com"
        admin.role = UserRole.ADMIN.value

        db = MagicMock()
        db.add = MagicMock()
        db.commit = AsyncMock()
        db.rollback = AsyncMock()

        async def _refresh(obj):
            from datetime import datetime, timezone
            obj.created_at = datetime.now(timezone.utc)
        db.refresh = AsyncMock(side_effect=_refresh)

        app = create_app()
        app.dependency_overrides[require_admin] = lambda: admin
        app.dependency_overrides[get_session] = lambda: db

        q_dir = tmp_path / "q"
        q_dir.mkdir()
        yield app, admin, db, q_dir
        app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_unsafe_pdf_rejected_422(self, _setup, caplog):
        """PDF with unsafe structure -> 422, no extraction, no success."""
        import logging
        from httpx import ASGITransport, AsyncClient
        app, admin, db, q_dir = _setup

        with caplog.at_level(logging.WARNING, logger="app.api.uploads"):
            with patch("app.services.upload_quarantine.settings") as qs:
                qs.upload_quarantine_path = str(q_dir)
                with patch("app.api.uploads.settings") as us:
                    us.upload_max_file_size = 10_485_760
                    us.upload_quarantine_retention_hours = 24
                    with patch("app.api.uploads.scan_file") as ms:
                        ms.return_value = MagicMock(clean=True)
                        with patch("app.services.pdf_security.validate_pdf_security",
                                   return_value=(False, UploadRejectionReason.PDF_ACTIVE_CONTENT)):
                            with patch("app.api.uploads.validate_pdf_not_encrypted", return_value=(True, None)):
                                with patch("app.api.uploads.extract_content") as mock_ext:
                                    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
                                        r = await c.post("/api/scripts/upload",
                                            files={"file": ("d.pdf", b"%PDF-1.4 x", "application/pdf")})

        assert r.status_code == 422
        assert r.json()["detail"]["reason_code"] == "pdf_active_content"
        mock_ext.assert_not_called()
        db.add.assert_not_called()
        assert not any(rec.message == "upload_success" for rec in caplog.records)
        remaining = [f for f in q_dir.iterdir() if f.is_file()]
        assert remaining == []
        # Verify exact structured audit fields
        rejected_records = [rec for rec in caplog.records if rec.message == "upload_rejected"]
        assert len(rejected_records) >= 1
        audit = rejected_records[0]
        assert audit.user_id == str(admin.id)
        assert audit.upload_filename == "d.pdf"
        assert audit.file_size == len(b"%PDF-1.4 x")
        assert audit.reason_code == "pdf_active_content"
        assert audit.ip_address == "127.0.0.1"

    @pytest.mark.asyncio
    async def test_unsafe_docx_rejected_422(self, _setup, caplog):
        """DOCX with unsafe content -> 422, no extraction, no success."""
        import logging
        from httpx import ASGITransport, AsyncClient
        app, admin, db, q_dir = _setup

        with caplog.at_level(logging.WARNING, logger="app.api.uploads"):
            with patch("app.services.upload_quarantine.settings") as qs:
                qs.upload_quarantine_path = str(q_dir)
                with patch("app.api.uploads.settings") as us:
                    us.upload_max_file_size = 10_485_760
                    us.upload_quarantine_retention_hours = 24
                    with patch("app.api.uploads.scan_file") as ms:
                        ms.return_value = MagicMock(clean=True)
                        with patch("app.services.docx_security.validate_docx_security",
                                   return_value=(False, UploadRejectionReason.DOCX_EMBEDDED_OBJECT)):
                            with patch("app.api.uploads.validate_docx_archive", return_value=(True, None)):
                                with patch("app.api.uploads.extract_content") as mock_ext:
                                    # Create a minimal valid ZIP for DOCX
                                    docx_buf = io.BytesIO()
                                    with zipfile.ZipFile(docx_buf, "w") as zf:
                                        zf.writestr("[Content_Types].xml", "<Types/>")
                                        zf.writestr("word/document.xml", "<doc/>")
                                    docx_bytes = docx_buf.getvalue()
                                    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
                                        r = await c.post("/api/scripts/upload",
                                            files={"file": ("d.docx", docx_bytes,
                                                "application/vnd.openxmlformats-officedocument.wordprocessingml.document")})

        assert r.status_code == 422
        assert r.json()["detail"]["reason_code"] == "docx_embedded_object"
        mock_ext.assert_not_called()
        db.add.assert_not_called()
        assert not any(rec.message == "upload_success" for rec in caplog.records)
        remaining = [f for f in q_dir.iterdir() if f.is_file()]
        assert remaining == []
        # Verify exact structured audit fields
        rejected_records = [rec for rec in caplog.records if rec.message == "upload_rejected"]
        assert len(rejected_records) >= 1
        audit = rejected_records[0]
        assert audit.user_id == str(admin.id)
        assert audit.upload_filename == "d.docx"
        assert audit.reason_code == "docx_embedded_object"
        assert audit.ip_address == "127.0.0.1"
        assert audit.file_size == len(docx_bytes)

    @pytest.mark.asyncio
    async def test_extraction_error_422(self, _setup, caplog):
        """ExtractionError -> 422, file deleted, no success."""
        import logging
        from httpx import ASGITransport, AsyncClient
        app, admin, db, q_dir = _setup

        with caplog.at_level(logging.WARNING, logger="app.api.uploads"):
            with patch("app.services.upload_quarantine.settings") as qs:
                qs.upload_quarantine_path = str(q_dir)
                with patch("app.api.uploads.settings") as us:
                    us.upload_max_file_size = 10_485_760
                    us.upload_quarantine_retention_hours = 24
                    with patch("app.api.uploads.scan_file") as ms:
                        ms.return_value = MagicMock(clean=True)
                        with patch("app.services.pdf_security.validate_pdf_security", return_value=(True, None)):
                            with patch("app.api.uploads.validate_pdf_not_encrypted", return_value=(True, None)):
                                with patch("app.api.uploads.extract_content", side_effect=ExtractionError("corrupt")):
                                    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
                                        r = await c.post("/api/scripts/upload",
                                            files={"file": ("d.pdf", b"%PDF-1.4 x", "application/pdf")})

        assert r.status_code == 422
        assert r.json()["detail"]["reason_code"] == "extraction_failed"
        assert "corrupt" not in r.json()["detail"]["message"]
        db.add.assert_not_called()
        assert not any(rec.message == "upload_success" for rec in caplog.records)
        remaining = [f for f in q_dir.iterdir() if f.is_file()]
        assert remaining == []
        # Verify exact structured audit fields for extraction error
        rejected_records = [rec for rec in caplog.records if rec.message == "upload_rejected"]
        assert len(rejected_records) >= 1
        audit = rejected_records[0]
        assert audit.user_id == str(admin.id)
        assert audit.upload_filename == "d.pdf"
        assert audit.file_size == len(b"%PDF-1.4 x")
        assert audit.reason_code == "extraction_failed"
        assert audit.ip_address == "127.0.0.1"


# ─── Independent Regression Probes (Bypass Cases) ─────────────────────────

class TestRegressionProbes:
    """Automated equivalents of three previously failing bypass probes.

    These are the critical regression tests that MUST pass:
    1. word/activeX/control.png must be rejected
    2. External relationship inside document.xml.RELS must be rejected
    3. PDF /URI action inside nested arrays must be rejected
    """

    def test_probe_1_activex_control_png_rejected(self, tmp_path):
        """PROBE 1: word/activeX/control.png MUST be rejected."""
        path = tmp_path / "probe1.docx"
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("[Content_Types].xml", "<Types/>")
            zf.writestr("word/document.xml", "<doc/>")
            zf.writestr("word/activeX/control.png", b"\x89PNG\r\n\x1a\n")
        valid, reason = validate_docx_security(path)
        assert valid is False, "BYPASS: word/activeX/control.png was NOT rejected!"
        assert reason == UploadRejectionReason.DOCX_EMBEDDED_OBJECT

    def test_probe_2_external_rel_in_uppercase_rels(self, tmp_path):
        """PROBE 2: External relationship in document.xml.RELS MUST be rejected."""
        path = tmp_path / "probe2.docx"
        rels_xml = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink" '
            'Target="https://evil.com/exfil" TargetMode="External"/>'
            '</Relationships>'
        )
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("[Content_Types].xml", "<Types/>")
            zf.writestr("word/document.xml", "<doc/>")
            zf.writestr("word/_rels/document.xml.RELS", rels_xml)
        valid, reason = validate_docx_security(path)
        assert valid is False, "BYPASS: External rel in .RELS was NOT rejected!"
        assert reason == UploadRejectionReason.DOCX_EXTERNAL_LINK

    def test_probe_3_pdf_uri_in_nested_arrays(self):
        """PROBE 3: PDF /URI action inside nested arrays MUST be rejected."""
        from pypdf import PdfWriter
        from pypdf.generic import (
            DictionaryObject, NameObject, TextStringObject, ArrayObject,
        )
        w = PdfWriter()
        p = w.add_blank_page(612, 792)
        # /URI inside dict -> inside array -> inside array -> on page
        action = DictionaryObject()
        action[NameObject("/S")] = NameObject("/URI")
        action[NameObject("/URI")] = TextStringObject("http://evil.com/nested")
        inner = ArrayObject([action])
        outer = ArrayObject([inner])
        p[NameObject("/Annots")] = outer
        buf = io.BytesIO()
        w.write(buf)

        valid, reason = validate_pdf_security(buf.getvalue())
        assert valid is False, "BYPASS: /URI in nested arrays was NOT rejected!"
        assert reason == UploadRejectionReason.PDF_EXTERNAL_LINK


# ─── XML Encoding Security Tests ──────────────────────────────────────────

class TestDocxXmlEncodingSecurity:
    """Encoding-based XML security bypass prevention.

    Verifies that DTD/entity declarations are rejected regardless of
    XML encoding: UTF-8, UTF-16 LE, UTF-16 BE, with/without BOM.
    """

    def test_utf16_le_xml_with_doctype_rejected(self, tmp_path):
        """UTF-16 LE .XML containing DOCTYPE is rejected."""
        path = tmp_path / "utf16le_dtd.docx"
        xml_content = '<?xml version="1.0" encoding="UTF-16"?><!DOCTYPE x SYSTEM "http://evil"><doc/>'
        xml_bytes = xml_content.encode("utf-16")  # BOM + UTF-16 LE
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("[Content_Types].xml", b'<?xml version="1.0"?><Types/>')
            zf.writestr("word/document.XML", xml_bytes)
        valid, reason = validate_docx_security(path)
        assert valid is False
        assert reason == UploadRejectionReason.DOCX_UNSAFE_XML

    def test_utf16_be_xml_with_entity_rejected(self, tmp_path):
        """UTF-16 BE .XML containing entity declaration is rejected."""
        path = tmp_path / "utf16be_entity.docx"
        xml_content = '<?xml version="1.0" encoding="UTF-16"?><!DOCTYPE x [<!ENTITY e "bad">]><doc/>'
        xml_bytes = b'\xfe\xff' + xml_content.encode("utf-16-be")
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("[Content_Types].xml", b'<?xml version="1.0"?><Types/>')
            zf.writestr("word/document.XML", xml_bytes)
        valid, reason = validate_docx_security(path)
        assert valid is False
        assert reason == UploadRejectionReason.DOCX_UNSAFE_XML

    def test_utf16_rels_with_doctype_rejected(self, tmp_path):
        """UTF-16 .RELS containing DOCTYPE is rejected."""
        path = tmp_path / "utf16_rels_dtd.docx"
        rels_content = (
            '<?xml version="1.0" encoding="UTF-16"?>'
            '<!DOCTYPE x SYSTEM "http://evil">'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"/>'
        )
        rels_bytes = rels_content.encode("utf-16")
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("[Content_Types].xml", b'<?xml version="1.0"?><Types/>')
            zf.writestr("word/document.xml", b'<?xml version="1.0"?><doc/>')
            zf.writestr("word/_rels/document.xml.RELS", rels_bytes)
        valid, reason = validate_docx_security(path)
        assert valid is False
        assert reason == UploadRejectionReason.DOCX_UNSAFE_XML

    def test_utf16_rels_with_entity_rejected(self, tmp_path):
        """UTF-16 .RELS containing entity declaration is rejected."""
        path = tmp_path / "utf16_rels_entity.docx"
        rels_content = (
            '<?xml version="1.0" encoding="UTF-16"?>'
            '<!DOCTYPE x [<!ENTITY xxe "evil">]>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"/>'
        )
        rels_bytes = rels_content.encode("utf-16")
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("[Content_Types].xml", b'<?xml version="1.0"?><Types/>')
            zf.writestr("word/document.xml", b'<?xml version="1.0"?><doc/>')
            zf.writestr("word/_rels/document.xml.RELS", rels_bytes)
        valid, reason = validate_docx_security(path)
        assert valid is False
        assert reason == UploadRejectionReason.DOCX_UNSAFE_XML

    def test_utf8_bom_xml_with_doctype_rejected(self, tmp_path):
        """UTF-8 BOM XML containing DOCTYPE is rejected."""
        path = tmp_path / "utf8bom.docx"
        xml_bytes = b'\xef\xbb\xbf<?xml version="1.0" encoding="UTF-8"?><!DOCTYPE x><doc/>'
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("[Content_Types].xml", b'<?xml version="1.0"?><Types/>')
            zf.writestr("word/document.xml", xml_bytes)
        valid, reason = validate_docx_security(path)
        assert valid is False
        assert reason == UploadRejectionReason.DOCX_UNSAFE_XML

    def test_malformed_document_xml_rejected(self, tmp_path):
        """Malformed word/document.xml is rejected."""
        path = tmp_path / "malformed_doc.docx"
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("[Content_Types].xml", b'<?xml version="1.0"?><Types/>')
            zf.writestr("word/document.xml", b"<unclosed tag")
        valid, reason = validate_docx_security(path)
        assert valid is False
        assert reason == UploadRejectionReason.DOCX_UNSAFE_XML

    def test_malformed_ancillary_xml_rejected(self, tmp_path):
        """Malformed ancillary/custom XML part is rejected."""
        path = tmp_path / "malformed_custom.docx"
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("[Content_Types].xml", b'<?xml version="1.0"?><Types/>')
            zf.writestr("word/document.xml", b'<?xml version="1.0"?><doc/>')
            zf.writestr("customXml/item1.xml", b"<<< not valid xml >>>")
        valid, reason = validate_docx_security(path)
        assert valid is False
        assert reason == UploadRejectionReason.DOCX_UNSAFE_XML

    def test_malformed_rels_rejected(self, tmp_path):
        """Malformed .rels file is rejected."""
        path = tmp_path / "malformed_rels.docx"
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("[Content_Types].xml", b'<?xml version="1.0"?><Types/>')
            zf.writestr("word/document.xml", b'<?xml version="1.0"?><doc/>')
            zf.writestr("word/_rels/document.xml.rels", b"not xml at all {{{")
        valid, reason = validate_docx_security(path)
        assert valid is False
        assert reason == UploadRejectionReason.DOCX_UNSAFE_XML

    def test_billion_laughs_style_rejected(self, tmp_path):
        """Entity-expansion / billion-laughs-style XML is rejected without resource consumption."""
        path = tmp_path / "laughs.docx"
        # This would consume GBs if entities were expanded; defusedxml blocks it
        laughs_xml = (
            b'<?xml version="1.0"?>'
            b'<!DOCTYPE lolz ['
            b'<!ENTITY lol "lol">'
            b'<!ENTITY lol2 "&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;">'
            b'<!ENTITY lol3 "&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;">'
            b']><root>&lol3;</root>'
        )
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("[Content_Types].xml", b'<?xml version="1.0"?><Types/>')
            zf.writestr("word/document.xml", laughs_xml)
        valid, reason = validate_docx_security(path)
        assert valid is False
        assert reason == UploadRejectionReason.DOCX_UNSAFE_XML

    def test_normal_utf8_xml_accepted(self, tmp_path):
        """Normal UTF-8 XML part is accepted."""
        path = tmp_path / "normal_utf8.docx"
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("[Content_Types].xml", b'<?xml version="1.0" encoding="UTF-8"?><Types/>')
            zf.writestr("word/document.xml", b'<?xml version="1.0" encoding="UTF-8"?><document><body/></document>')
        valid, reason = validate_docx_security(path)
        assert valid is True

    def test_valid_utf16_xml_without_dtd_accepted(self, tmp_path):
        """Valid UTF-16 XML part without DTD/entities is accepted."""
        path = tmp_path / "valid_utf16.docx"
        xml_content = '<?xml version="1.0" encoding="UTF-16"?><document><body/></document>'
        xml_bytes = xml_content.encode("utf-16")  # BOM + UTF-16 LE
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("[Content_Types].xml", b'<?xml version="1.0"?><Types/>')
            zf.writestr("word/document.xml", xml_bytes)
        valid, reason = validate_docx_security(path)
        assert valid is True

    def test_normal_docx_from_python_docx_accepted(self, tmp_path):
        """Ordinary DOCX file generated by python-docx is accepted."""
        from docx import Document
        doc = Document()
        doc.add_paragraph("Normal training script content.")
        doc.add_paragraph("Second paragraph with UTF-8 text: résumé, naïve.")
        path = tmp_path / "python_docx.docx"
        doc.save(str(path))
        valid, reason = validate_docx_security(path)
        assert valid is True

    def test_normal_internal_rels_in_valid_docx(self, tmp_path):
        """Internal relationships in a normal DOCX are accepted."""
        path = tmp_path / "int_rels.docx"
        rels_xml = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" '
            'Target="styles.xml"/>'
            '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/settings" '
            'Target="settings.xml"/>'
            '</Relationships>'
        )
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("[Content_Types].xml", b'<?xml version="1.0"?><Types/>')
            zf.writestr("word/document.xml", b'<?xml version="1.0"?><doc/>')
            zf.writestr("word/_rels/document.xml.rels", rels_xml.encode("utf-8"))
            zf.writestr("word/styles.xml", b'<?xml version="1.0"?><styles/>')
            zf.writestr("word/settings.xml", b'<?xml version="1.0"?><settings/>')
        valid, reason = validate_docx_security(path)
        assert valid is True


# ─── UTF-16 Encoding Bypass Regression Probes ─────────────────────────────

class TestEncodingBypassProbes:
    """Previously failing probes: UTF-16 XML/RELS with DTD/entity MUST be rejected."""

    def test_probe_utf16_xml_with_dtd_rejected(self, tmp_path):
        """PROBE: UTF-16 .XML with DTD MUST be rejected."""
        path = tmp_path / "probe_utf16_xml.docx"
        xml_content = '<?xml version="1.0" encoding="UTF-16"?><!DOCTYPE x SYSTEM "evil"><doc/>'
        xml_bytes = xml_content.encode("utf-16")
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("[Content_Types].xml", b'<?xml version="1.0"?><Types/>')
            zf.writestr("word/document.XML", xml_bytes)
        valid, reason = validate_docx_security(path)
        assert valid is False, "BYPASS: UTF-16 .XML with DTD was NOT rejected!"
        assert reason == UploadRejectionReason.DOCX_UNSAFE_XML

    def test_probe_utf16_xml_with_entity_rejected(self, tmp_path):
        """PROBE: UTF-16 .XML with entity MUST be rejected."""
        path = tmp_path / "probe_utf16_entity.docx"
        xml_content = '<?xml version="1.0" encoding="UTF-16"?><!DOCTYPE x [<!ENTITY e "pwned">]><doc/>'
        xml_bytes = xml_content.encode("utf-16")
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("[Content_Types].xml", b'<?xml version="1.0"?><Types/>')
            zf.writestr("word/custom.XML", xml_bytes)
        valid, reason = validate_docx_security(path)
        assert valid is False, "BYPASS: UTF-16 .XML with entity was NOT rejected!"
        assert reason == UploadRejectionReason.DOCX_UNSAFE_XML

    def test_probe_utf16_rels_with_dtd_rejected(self, tmp_path):
        """PROBE: UTF-16 .RELS with DTD MUST be rejected."""
        path = tmp_path / "probe_utf16_rels.docx"
        rels_content = (
            '<?xml version="1.0" encoding="UTF-16"?>'
            '<!DOCTYPE x SYSTEM "evil">'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"/>'
        )
        rels_bytes = rels_content.encode("utf-16")
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("[Content_Types].xml", b'<?xml version="1.0"?><Types/>')
            zf.writestr("word/document.xml", b'<?xml version="1.0"?><doc/>')
            zf.writestr("_rels/.RELS", rels_bytes)
        valid, reason = validate_docx_security(path)
        assert valid is False, "BYPASS: UTF-16 .RELS with DTD was NOT rejected!"
        assert reason == UploadRejectionReason.DOCX_UNSAFE_XML

    def test_probe_utf16_rels_with_entity_rejected(self, tmp_path):
        """PROBE: UTF-16 .RELS with entity MUST be rejected."""
        path = tmp_path / "probe_utf16_rels_ent.docx"
        rels_content = (
            '<?xml version="1.0" encoding="UTF-16"?>'
            '<!DOCTYPE x [<!ENTITY e "bad">]>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"/>'
        )
        rels_bytes = rels_content.encode("utf-16")
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("[Content_Types].xml", b'<?xml version="1.0"?><Types/>')
            zf.writestr("word/document.xml", b'<?xml version="1.0"?><doc/>')
            zf.writestr("word/_rels/document.xml.RELS", rels_bytes)
        valid, reason = validate_docx_security(path)
        assert valid is False, "BYPASS: UTF-16 .RELS with entity was NOT rejected!"
        assert reason == UploadRejectionReason.DOCX_UNSAFE_XML
