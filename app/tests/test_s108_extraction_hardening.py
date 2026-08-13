"""S1-08: Extraction hardening tests — encoding, CSV, Markdown, HTML."""

import csv
import io
from unittest.mock import patch

import pytest

from app.services.upload_extractor import (
    _MAX_CSV_CELL_LEN,
    _MAX_CSV_ROWS,
    _MAX_EXTRACTED_BYTES,
    ExtractionError,
    _enforce_size_limit,
    _is_chardet_trustworthy,
    _is_dangerous_uri,
    _neutralize_csv_cell,
    _normalize_encoding,
    _strip_control_chars,
    compute_content_hash,
    extract_content,
    extract_csv,
    extract_md,
    extract_txt,
)


# ═══ BOM Handling ═════════════════════════════════════════════════════════


class TestBomHandling:
    """BOMs are consumed; decoded text never starts with U+FEFF from BOM."""

    def test_utf8_bom_removed(self):
        raw = b"\xef\xbb\xbfHello"
        assert _normalize_encoding(raw) == "Hello"

    def test_utf16_le_bom_removed(self):
        raw = "Hello".encode("utf-16-le")
        result = _normalize_encoding(b"\xff\xfe" + raw)
        assert result == "Hello"
        assert not result.startswith("\ufeff")

    def test_utf16_be_bom_removed(self):
        raw = "Hello".encode("utf-16-be")
        result = _normalize_encoding(b"\xfe\xff" + raw)
        assert result == "Hello"

    def test_utf32_le_bom_removed(self):
        raw = "Hi".encode("utf-32-le")
        result = _normalize_encoding(b"\xff\xfe\x00\x00" + raw)
        assert result == "Hi"

    def test_utf32_be_bom_removed(self):
        raw = "Hi".encode("utf-32-be")
        result = _normalize_encoding(b"\x00\x00\xfe\xff" + raw)
        assert result == "Hi"

    def test_non_ascii_after_bom(self):
        raw = b"\xef\xbb\xbf" + "café".encode()
        assert _normalize_encoding(raw) == "café"

    def test_empty_bom_only(self):
        result = _normalize_encoding(b"\xef\xbb\xbf")
        assert result == ""

    def test_embedded_feff_not_stripped(self):
        """U+FEFF in non-BOM position is preserved."""
        raw = "A\ufeffB".encode()
        result = _normalize_encoding(raw)
        assert "\ufeff" in result


class TestBomCsvFormulaProbes:
    """BOM-prefixed formulas must be neutralized after extraction."""

    def test_utf16_le_bom_formula(self, tmp_path):
        path = tmp_path / "u16le.csv"
        path.write_bytes("=CMD(),ok".encode("utf-16"))
        result = extract_csv(path)
        reader = csv.reader(io.StringIO(result))
        rows = list(reader)
        assert rows[0][0].startswith("'"), f"Not neutralized: {rows[0][0]!r}"

    def test_utf16_be_bom_plus(self, tmp_path):
        path = tmp_path / "u16be.csv"
        content = "+100,ok".encode("utf-16-be")
        path.write_bytes(b"\xfe\xff" + content)
        result = extract_csv(path)
        reader = csv.reader(io.StringIO(result))
        rows = list(reader)
        assert rows[0][0].startswith("'")

    def test_utf32_bom_at_sum(self, tmp_path):
        path = tmp_path / "u32.csv"
        path.write_bytes("@SUM(1,2),x".encode("utf-32"))
        result = extract_csv(path)
        reader = csv.reader(io.StringIO(result))
        rows = list(reader)
        assert rows[0][0].startswith("'")

    def test_utf8_bom_pipe(self, tmp_path):
        path = tmp_path / "u8bom.csv"
        path.write_bytes(b"\xef\xbb\xbf|cmd,safe")
        result = extract_csv(path)
        reader = csv.reader(io.StringIO(result))
        rows = list(reader)
        assert rows[0][0].startswith("'")


# ═══ Encoding Detection ═══════════════════════════════════════════════════


class TestEncodingDetection:
    """Encoding normalization: short inputs, chardet, fallback."""

    def test_short_ascii(self):
        assert _normalize_encoding(b"Plaintext") == "Plaintext"
        assert _normalize_encoding(b"hello") == "hello"
        assert _normalize_encoding(b"+100") == "+100"
        assert _normalize_encoding(b"name,value\nfoo,bar") == "name,value\nfoo,bar"

    def test_short_utf8_non_ascii(self):
        assert _normalize_encoding("café".encode()) == "café"

    def test_empty(self):
        assert _normalize_encoding(b"") == ""

    def test_latin1(self):
        raw = "café".encode("latin-1")
        result = _normalize_encoding(raw)
        assert isinstance(result, str) and len(result) > 0


class TestChardetConfidence:
    """Chardet confidence threshold enforcement."""

    def test_high_confidence_trusted(self):
        assert _is_chardet_trustworthy("latin-1", 0.9) is True

    def test_at_threshold_trusted(self):
        assert _is_chardet_trustworthy("windows-1252", 0.5) is True

    def test_below_threshold_rejected(self):
        assert _is_chardet_trustworthy("windows-1252", 0.49) is False

    def test_missing_confidence(self):
        assert _is_chardet_trustworthy("utf-8", None) is False

    def test_none_confidence(self):
        assert _is_chardet_trustworthy("utf-8", None) is False

    def test_invalid_confidence_type(self):
        assert _is_chardet_trustworthy("utf-8", "high") is False

    def test_unknown_encoding_but_valid_confidence(self):
        assert _is_chardet_trustworthy("fake-enc", 0.9) is True

    def test_utf7_rejected_high_confidence(self):
        assert _is_chardet_trustworthy("utf-7", 0.99) is False
        assert _is_chardet_trustworthy("UTF-7", 0.99) is False

    def test_low_confidence_fallback(self, tmp_path):
        """Low-confidence chardet => falls back to UTF-8."""
        path = tmp_path / "low.txt"
        path.write_bytes(b"simple ascii")
        with patch("chardet.detect", return_value={"encoding": "EUC-JP", "confidence": 0.3}):
            result = extract_txt(path)
        assert result == "simple ascii"

    def test_chardet_unavailable(self, tmp_path):
        path = tmp_path / "nochardet.txt"
        path.write_bytes(b"hello")
        with patch.dict("sys.modules", {"chardet": None}):
            with patch("builtins.__import__", side_effect=ImportError):
                # Falls through to UTF-8 fallback
                result = _normalize_encoding(b"hello")
        assert result == "hello"


# ═══ Control Chars ════════════════════════════════════════════════════════


class TestControlChars:
    def test_null_removed(self):
        assert _strip_control_chars("A\x00B") == "AB"

    def test_tab_preserved(self):
        assert _strip_control_chars("a\tb") == "a\tb"

    def test_newline_preserved(self):
        assert _strip_control_chars("a\nb") == "a\nb"

    def test_cr_preserved(self):
        assert _strip_control_chars("a\r\nb") == "a\r\nb"

    def test_c1_removed(self):
        assert _strip_control_chars("A\x80\x9fB") == "AB"

    def test_del_removed(self):
        assert _strip_control_chars("a\x7fb") == "ab"


# ═══ CSV Serialization ════════════════════════════════════════════════════


class TestCsvSerialization:
    """csv.writer preserves cell boundaries through round-trip."""

    def test_quoted_cell_with_comma_formula(self, tmp_path):
        """'safe,=CMD()' must remain one cell."""
        path = tmp_path / "q.csv"
        buf = io.StringIO()
        csv.writer(buf).writerow(["value"])
        csv.writer(buf).writerow(["safe,=CMD()"])
        path.write_text(buf.getvalue())
        result = extract_csv(path)
        rows = list(csv.reader(io.StringIO(result)))
        assert len(rows[1]) == 1
        assert "safe" in rows[1][0] and "=CMD()" in rows[1][0]

    def test_multiple_commas(self, tmp_path):
        path = tmp_path / "c.csv"
        buf = io.StringIO()
        csv.writer(buf).writerow(["a,b,c,d"])
        path.write_text(buf.getvalue())
        rows = list(csv.reader(io.StringIO(extract_csv(path))))
        assert rows[0] == ["a,b,c,d"]

    def test_double_quotes(self, tmp_path):
        path = tmp_path / "dq.csv"
        buf = io.StringIO()
        csv.writer(buf).writerow(['He said "hi"'])
        path.write_text(buf.getvalue())
        rows = list(csv.reader(io.StringIO(extract_csv(path))))
        assert rows[0][0] == 'He said "hi"'

    def test_embedded_newline(self, tmp_path):
        path = tmp_path / "nl.csv"
        buf = io.StringIO()
        csv.writer(buf).writerow(["line1\nline2"])
        path.write_text(buf.getvalue())
        rows = list(csv.reader(io.StringIO(extract_csv(path))))
        assert len(rows) == 1
        assert "line1" in rows[0][0] and "line2" in rows[0][0]

    def test_empty_cells(self, tmp_path):
        path = tmp_path / "e.csv"
        buf = io.StringIO()
        csv.writer(buf).writerow(["a", "", "b", ""])
        path.write_text(buf.getvalue())
        rows = list(csv.reader(io.StringIO(extract_csv(path))))
        assert rows[0] == ["a", "", "b", ""]

    def test_round_trip(self, tmp_path):
        path = tmp_path / "rt.csv"
        original = [["name", "note"], ["Alice", "has, comma"], ["Bob", 'says "hi"']]
        buf = io.StringIO()
        csv.writer(buf).writerows(original)
        path.write_text(buf.getvalue())
        rows = list(csv.reader(io.StringIO(extract_csv(path))))
        assert rows == original


# ═══ CSV Size Limits (UTF-8 Bytes) ════════════════════════════════════════


class TestCsvByteLimits:
    """Size limit enforced in UTF-8 bytes, not characters."""

    def test_ascii_at_limit(self, tmp_path):
        path = tmp_path / "at.csv"
        # 10-byte limit: "a,b\r\n" is 5 bytes, "c,d\r\n" is 5 bytes = 10 total
        path.write_text("a,b\nc,d\n")
        # With 10-byte limit, this should just fit (rows: "a,b\r\n"=5 + "c,d\r\n"=5)
        result = extract_csv(path, max_bytes=10)
        assert "a" in result

    def test_ascii_over_limit(self, tmp_path):
        path = tmp_path / "over.csv"
        path.write_text("a,b\nc,d\nX,Y\n")
        with pytest.raises(ExtractionError, match="size limit"):
            extract_csv(path, max_bytes=10)

    def test_multibyte_over_limit(self, tmp_path):
        """Multibyte chars: 'é' is 2 UTF-8 bytes."""
        path = tmp_path / "mb.csv"
        # "é\r\n" = 2+1+1 = 4 bytes per row
        path.write_text("é\né\né\n")
        with pytest.raises(ExtractionError, match="size limit"):
            extract_csv(path, max_bytes=10)

    def test_regression_probe_12_bytes(self, tmp_path):
        """6 two-byte chars = 12 UTF-8 bytes must be rejected at 10-byte limit."""
        path = tmp_path / "probe.csv"
        # "éééééé" = 6 chars * 2 bytes = 12 bytes + CRLF = 14
        path.write_text("éééééé\n")
        with pytest.raises(ExtractionError, match="size limit"):
            extract_csv(path, max_bytes=10)

    def test_formula_prefix_in_byte_count(self, tmp_path):
        path = tmp_path / "fp.csv"
        # "=x\r\n" without prefix: 4 bytes. With "'": "'=x\r\n" = 5 bytes
        path.write_text("=x\n")
        # At 4-byte limit the prefixed output (5 bytes) should fail
        with pytest.raises(ExtractionError, match="size limit"):
            extract_csv(path, max_bytes=4)

    def test_quoting_in_byte_count(self, tmp_path):
        path = tmp_path / "qt.csv"
        # Cell "a,b" quoted becomes '"a,b"\r\n' = 7 bytes
        buf = io.StringIO()
        csv.writer(buf).writerow(["a,b"])
        path.write_text(buf.getvalue())
        with pytest.raises(ExtractionError, match="size limit"):
            extract_csv(path, max_bytes=6)


# ═══ CSV Formula Neutralization ═══════════════════════════════════════════


class TestFormulaNeutralization:
    @pytest.mark.parametrize(
        "cell,expect_prefix",
        [
            ("=CMD()", True),
            ("+100", True),
            ("-50", True),
            ("@SUM(1,2)", True),
            ("|cmd", True),
            ("\t=CMD()", True),
            (" =CMD()", True),
            ("  @SUM(1,2)", True),
            ("\r=CMD()", True),
            ("\n=CMD()", True),
            (" \t =CMD()", True),
            ("\u00a0=CMD()", True),  # NBSP
            ("\ufeff=CMD()", True),  # BOM artifact
            ("Normal", False),
            ("123", False),
            ("", False),
            ("   ", False),
            ("'quoted", False),
        ],
    )
    def test_neutralization(self, cell, expect_prefix):
        result = _neutralize_csv_cell(cell)
        if expect_prefix:
            assert result.startswith("'"), f"{cell!r} not neutralized"
        else:
            assert not result.startswith("'") or cell.startswith("'")

    def test_cell_truncated(self):
        long = "A" * (_MAX_CSV_CELL_LEN + 100)
        assert len(_neutralize_csv_cell(long)) == _MAX_CSV_CELL_LEN

    def test_neutralized_truncated(self):
        long = "=" + "A" * _MAX_CSV_CELL_LEN
        result = _neutralize_csv_cell(long)
        assert len(result) <= _MAX_CSV_CELL_LEN and result.startswith("'")


# ═══ Markdown URI Normalization ═══════════════════════════════════════════


class TestMarkdownUri:
    @pytest.mark.parametrize(
        "uri,dangerous",
        [
            ("data:text/html,hello", True),
            ("data:text/html;base64,abc", True),
            ("DaTa:image/svg+xml,x", True),
            ("javascript:alert(1)", True),
            ("JaVaScRiPt:void(0)", True),
            ("vbscript:x", True),
            ("file:///etc/passwd", True),
            (" javascript:x", True),
            ("java%73cript:alert(1)", True),  # percent-encoded 's'
            ("javascript%3Aalert(1)", True),  # percent-encoded ':'
            ("javascript%253Aalert(1)", True),  # double-encoded ':'
            ("javascript%2525253Aalert(1)", True),  # 4-level encoded ':'
            ("javascript&#58;alert(1)", True),  # HTML entity ':'
            ("javascript&#00058;alert(1)", True),  # padded entity
            ("javascript&#x3a;alert(1)", True),  # hex entity
            # Embedded whitespace/control in scheme
            ("java\tscript:alert(1)", True),
            ("java\nscript:alert(1)", True),
            ("java\rscript:alert(1)", True),
            ("java%09script:alert(1)", True),
            ("java%0Ascript:alert(1)", True),
            ("java%0Dscript:alert(1)", True),
            ("java&#9;script:alert(1)", True),
            ("java&#10;script:alert(1)", True),
            ("java&#13;script:alert(1)", True),
            ("vb%0Ascript:x", True),
            ("da%0Dta:text/html,x", True),
            ("fi&#9;le:///etc/passwd", True),
            # Safe
            ("https://safe.com", False),
            ("http://example.org", False),
            ("mailto:u@x.com", False),
            ("/relative", False),
            ("#frag", False),
            ("./x.md", False),
            ("https://x.com/path%20here", False),
            ("./file%20name.md", False),
            ("http://example.org/a?q=%3D", False),
        ],
    )
    def test_is_dangerous(self, uri, dangerous):
        assert _is_dangerous_uri(uri) == dangerous, f"URI {uri!r}"

    def test_inline_data_html(self, tmp_path):
        path = tmp_path / "d.md"
        path.write_bytes(b"[click](data:text/html,hello)")
        result = extract_md(path)
        assert "data:" not in result and "click" in result

    def test_inline_javascript_space(self, tmp_path):
        path = tmp_path / "js.md"
        path.write_bytes(b"[x]( javascript:alert(1))")
        result = extract_md(path)
        assert "javascript:" not in result

    def test_safe_https_preserved(self, tmp_path):
        path = tmp_path / "s.md"
        path.write_bytes(b"[ok](https://safe.com)")
        result = extract_md(path)
        assert "[ok](https://safe.com)" in result

    def test_safe_relative_preserved(self, tmp_path):
        path = tmp_path / "r.md"
        path.write_bytes(b"[doc](./other.md)")
        assert "[doc](./other.md)" in extract_md(path)

    def test_safe_fragment_preserved(self, tmp_path):
        path = tmp_path / "f.md"
        path.write_bytes(b"[s](#top)")
        assert "[s](#top)" in extract_md(path)


# ═══ Reference-Style Markdown Links ══════════════════════════════════════


class TestMarkdownRefLinks:
    """Dangerous reference definitions are neutralized."""

    def test_dangerous_ref_definition(self, tmp_path):
        path = tmp_path / "ref.md"
        content = "[x][r]\n\n[r]: javascript:alert(1)\n"
        path.write_bytes(content.encode("utf-8"))
        result = extract_md(path)
        assert "javascript:" not in result
        # Definition should have # as destination
        assert "[r]: #" in result

    def test_dangerous_image_ref(self, tmp_path):
        path = tmp_path / "imgref.md"
        content = "![img][d]\n\n[d]: data:image/svg+xml,<svg/>\n"
        path.write_bytes(content.encode("utf-8"))
        result = extract_md(path)
        assert "data:" not in result

    def test_safe_ref_preserved(self, tmp_path):
        path = tmp_path / "saferef.md"
        content = "[link][s]\n\n[s]: https://safe.com\n"
        path.write_bytes(content.encode("utf-8"))
        result = extract_md(path)
        assert "[s]: https://safe.com" in result

    def test_safe_relative_ref(self, tmp_path):
        path = tmp_path / "relref.md"
        content = "[doc][r]\n\n[r]: ./page.md\n"
        path.write_bytes(content.encode("utf-8"))
        result = extract_md(path)
        assert "[r]: ./page.md" in result

    def test_encoded_dangerous_ref(self, tmp_path):
        path = tmp_path / "encref.md"
        content = "[x][e]\n\n[e]: java%73cript:alert(1)\n"
        path.write_bytes(content.encode("utf-8"))
        result = extract_md(path)
        assert "javascript" not in result.lower() or "[e]: #" in result

    def test_vbscript_ref(self, tmp_path):
        path = tmp_path / "vbs.md"
        content = "[x][v]\n\n[v]: vbscript:x\n"
        path.write_bytes(content.encode("utf-8"))
        result = extract_md(path)
        assert "vbscript:" not in result

    def test_file_ref(self, tmp_path):
        path = tmp_path / "file.md"
        content = "[x][f]\n\n[f]: file:///etc/passwd\n"
        path.write_bytes(content.encode("utf-8"))
        result = extract_md(path)
        assert "file:" not in result


# ═══ Raw HTML Sanitation ══════════════════════════════════════════════════


class TestHtmlSanitation:
    def test_mixed_case_script(self, tmp_path):
        path = tmp_path / "s.md"
        path.write_bytes(b"A <ScRiPt>evil()</ScRiPt> B")
        r = extract_md(path)
        assert "evil()" not in r and "A" in r and "B" in r

    def test_iframe(self, tmp_path):
        path = tmp_path / "i.md"
        path.write_bytes(b'X <iframe src="e"></iframe> Y')
        r = extract_md(path)
        assert "<iframe" not in r and "X" in r and "Y" in r

    def test_svg_event(self, tmp_path):
        path = tmp_path / "svg.md"
        path.write_bytes(b'<svg onload="x()"><circle/></svg> ok')
        r = extract_md(path)
        assert "<svg" not in r and "ok" in r

    def test_embed_self_closing(self, tmp_path):
        path = tmp_path / "emb.md"
        path.write_bytes(b'before <embed src="x"> after')
        r = extract_md(path)
        assert "<embed" not in r and "before" in r and "after" in r

    def test_unclosed_script(self, tmp_path):
        path = tmp_path / "uc.md"
        path.write_bytes(b"Normal <script>never closed")
        r = extract_md(path)
        assert "<script" not in r and "Normal" in r


# ═══ Content Limits ═══════════════════════════════════════════════════════


class TestLimits:
    def test_txt_over_limit(self, tmp_path):
        path = tmp_path / "big.txt"
        path.write_bytes(b"x" * (_MAX_EXTRACTED_BYTES + 1))
        with pytest.raises(ExtractionError):
            extract_txt(path)

    def test_csv_row_limit(self, tmp_path):
        path = tmp_path / "rows.csv"
        path.write_text("a\n" * (_MAX_CSV_ROWS + 1))
        with pytest.raises(ExtractionError, match="row count"):
            extract_csv(path)

    def test_enforce_size_truncates(self):
        big = "A" * (_MAX_EXTRACTED_BYTES + 100)
        assert len(_enforce_size_limit(big).encode("utf-8")) <= _MAX_EXTRACTED_BYTES


# ═══ Dispatch and Hash ════════════════════════════════════════════════════


class TestDispatch:
    def test_txt(self, tmp_path):
        p = tmp_path / "t.txt"
        p.write_bytes(b"hello")
        assert extract_content(p, ".txt") == "hello"

    def test_csv(self, tmp_path):
        p = tmp_path / "t.csv"
        p.write_bytes(b"a,b\n1,2\n")
        r = extract_content(p, ".csv")
        rows = list(csv.reader(io.StringIO(r)))
        assert rows == [["a", "b"], ["1", "2"]]

    def test_md(self, tmp_path):
        p = tmp_path / "t.md"
        p.write_bytes(b"# Hi")
        assert "# Hi" in extract_content(p, ".md")

    def test_unsupported(self, tmp_path):
        p = tmp_path / "x.xyz"
        p.write_bytes(b"x")
        with pytest.raises(ValueError):
            extract_content(p, ".xyz")

    def test_hash_deterministic(self):
        assert compute_content_hash("x") == compute_content_hash("x")
        assert len(compute_content_hash("x")) == 64


# ═══ Independent Regression Probes ════════════════════════════════════════


class TestRegressionProbes:
    """All previously demonstrated bypasses must be neutralized."""

    def test_utf16_bom_formula_bypass(self, tmp_path):
        path = tmp_path / "bom.csv"
        path.write_bytes("=CMD(),ok".encode("utf-16"))
        result = extract_csv(path)
        rows = list(csv.reader(io.StringIO(result)))
        assert rows[0][0].startswith("'"), "UTF-16 BOM formula bypass!"

    def test_utf32_bom_formula_bypass(self, tmp_path):
        path = tmp_path / "u32.csv"
        path.write_bytes("@SUM(1,2),x".encode("utf-32"))
        result = extract_csv(path)
        rows = list(csv.reader(io.StringIO(result)))
        assert rows[0][0].startswith("'"), "UTF-32 BOM formula bypass!"

    def test_utf8_byte_size_undercount(self, tmp_path):
        """Multibyte output must be counted in bytes, not chars."""
        path = tmp_path / "mb.csv"
        path.write_text("éééééé\n")  # 6 * 2 = 12 UTF-8 bytes + CRLF
        with pytest.raises(ExtractionError):
            extract_csv(path, max_bytes=10)

    def test_percent_encoded_javascript(self, tmp_path):
        path = tmp_path / "pct.md"
        path.write_bytes(b"[x](java%73cript:alert(1))")
        result = extract_md(path)
        assert "javascript" not in result.lower()
        assert "x" in result

    def test_entity_encoded_javascript(self, tmp_path):
        path = tmp_path / "ent.md"
        path.write_bytes(b"[x](javascript&#00058;alert(1))")
        result = extract_md(path)
        assert "javascript" not in result.lower()

    def test_reference_style_javascript(self, tmp_path):
        path = tmp_path / "ref.md"
        content = "[x][r]\n\n[r]: javascript:alert(1)\n"
        path.write_bytes(content.encode("utf-8"))
        result = extract_md(path)
        assert "javascript:" not in result

    def test_safe_reference_https(self, tmp_path):
        path = tmp_path / "safe.md"
        content = "[x][r]\n\n[r]: https://safe.com\n"
        path.write_bytes(content.encode("utf-8"))
        result = extract_md(path)
        assert "[r]: https://safe.com" in result

    def test_low_confidence_chardet_fallback(self, tmp_path):
        path = tmp_path / "low.txt"
        path.write_bytes(b"simple text")
        with patch("chardet.detect", return_value={"encoding": "EUC-JP", "confidence": 0.3}):
            result = extract_txt(path)
        assert result == "simple text"

    def test_csv_quoted_cell_roundtrip(self, tmp_path):
        """'safe,=CMD()' must stay one cell."""
        path = tmp_path / "rt.csv"
        buf = io.StringIO()
        csv.writer(buf).writerow(["value"])
        csv.writer(buf).writerow(["safe,=CMD()"])
        path.write_text(buf.getvalue())
        result = extract_csv(path)
        rows = list(csv.reader(io.StringIO(result)))
        assert len(rows[1]) == 1
        assert "safe" in rows[1][0] and "=CMD()" in rows[1][0]


# ═══ URI Scheme Normalization Bypass Regression ═══════════════════════════


class TestUriSchemeBypassRegression:
    """Embedded whitespace/control chars and deep encoding in schemes."""

    @pytest.mark.parametrize(
        "uri",
        [
            "java\tscript:alert(1)",
            "java%09script:alert(1)",
            "java&#9;script:alert(1)",
            "java\nscript:x",
            "java%0Ascript:x",
            "java&#10;script:x",
            "java\rscript:x",
            "java%0Dscript:x",
            "java&#13;script:x",
            "vb%0Ascript:x",
            "da%0Dta:text/html,x",
            "fi&#9;le:///etc/passwd",
        ],
    )
    def test_embedded_whitespace_dangerous(self, uri):
        assert _is_dangerous_uri(uri), f"NOT detected: {uri!r}"

    def test_four_level_encoding(self):
        assert _is_dangerous_uri("javascript%2525253Aalert(1)")

    def test_triple_encoded_colon(self):
        assert _is_dangerous_uri("javascript%25253Aalert(1)")

    def test_normalization_stable_before_limit(self):
        """Already-clean URI stabilizes in 1 iteration."""
        assert not _is_dangerous_uri("https://safe.com")

    def test_safe_https_encoded_path(self):
        assert not _is_dangerous_uri("https://x.com/path%20with%20spaces")

    def test_safe_relative_encoded(self):
        assert not _is_dangerous_uri("./file%20name.md")

    def test_safe_fragment_encoded(self):
        assert not _is_dangerous_uri("#section%20two")

    # Full extraction probes
    def test_inline_tab_javascript(self, tmp_path):
        path = tmp_path / "tab.md"
        path.write_bytes(b"[x](java\tscript:alert(1))")
        result = extract_md(path)
        assert "javascript" not in result.replace("\t", "").lower()
        assert "x" in result

    def test_inline_percent09(self, tmp_path):
        path = tmp_path / "pct09.md"
        path.write_bytes(b"[x](java%09script:alert(1))")
        result = extract_md(path)
        assert "alert" not in result

    def test_inline_entity9(self, tmp_path):
        path = tmp_path / "ent9.md"
        path.write_bytes(b"[x](java&#9;script:alert(1))")
        result = extract_md(path)
        assert "alert" not in result

    def test_ref_def_encoded_whitespace(self, tmp_path):
        path = tmp_path / "ref_ws.md"
        content = "[x][r]\n\n[r]: java%09script:alert(1)\n"
        path.write_bytes(content.encode("utf-8"))
        result = extract_md(path)
        assert "java" not in result.lower() or "[r]: #" in result

    def test_image_encoded_whitespace(self, tmp_path):
        path = tmp_path / "img_ws.md"
        path.write_bytes(b"![x](da%0Dta:text/html,evil)")
        result = extract_md(path)
        assert "data" not in result.lower()

    def test_four_level_inline(self, tmp_path):
        path = tmp_path / "deep.md"
        path.write_bytes(b"[x](javascript%2525253Aalert(1))")
        result = extract_md(path)
        assert "alert" not in result

    def test_safe_https_link_preserved(self, tmp_path):
        path = tmp_path / "safe.md"
        path.write_bytes(b"[ok](https://safe.com/path%20here)")
        result = extract_md(path)
        assert "[ok](https://safe.com/path%20here)" in result

    def test_safe_ref_encoded_query(self, tmp_path):
        path = tmp_path / "saferef.md"
        content = "[x][s]\n\n[s]: https://x.com/a?q=%3D\n"
        path.write_bytes(content.encode("utf-8"))
        result = extract_md(path)
        assert "[s]: https://x.com/a?q=%3D" in result


# ═══ Fully Encoded URI Scheme Bypass Tests ════════════════════════════════


def _full_percent_encode(s: str) -> str:
    """Percent-encode every byte of a string."""
    return "".join(f"%{b:02X}" for b in s.encode("ascii"))


def _encode_to_depth(s: str, depth: int) -> str:
    """Repeatedly percent-encode a string to the given depth."""
    result = s
    for _ in range(depth):
        result = _full_percent_encode(result)
    return result


# ═══ Fully Encoded and Boundary Tests ═════════════════════════════════════


class TestFullyEncodedSchemes:
    """Dangerous schemes encoded at depth 1 must be rejected."""

    @pytest.mark.parametrize(
        "uri",
        [
            _encode_to_depth("javascript:x", 1),
            _encode_to_depth("data:x", 1),
            _encode_to_depth("vbscript:x", 1),
            _encode_to_depth("file:x", 1),
        ],
        ids=["js-d1", "data-d1", "vbs-d1", "file-d1"],
    )
    def test_depth1(self, uri):
        assert _is_dangerous_uri(uri)

    @pytest.mark.parametrize(
        "uri",
        [
            "java\tscript:x",
            "java%09script:x",
            "java&#9;script:x",
            "javascript%2525253Aalert(1)",
        ],
        ids=["lit-tab", "pct-tab", "ent-tab", "mixed-colon"],
    )
    def test_controls(self, uri):
        assert _is_dangerous_uri(uri)

    def test_oversized_unresolved(self):
        assert _is_dangerous_uri("%25" * 100 + "javascript:x")

    def test_non_stabilizing(self):
        # Deeply encoded percent signs that don't stabilize in 3 iterations
        assert _is_dangerous_uri("%2525252525253Ajavascript%3Ax")


class TestCustomSchemes:
    """Encoded custom schemes with paths must remain safe."""

    def test_webcal_encoded_colon(self):
        assert not _is_dangerous_uri("webcal%3A//safe.example/path%252520x")

    def test_custom_with_encoded_path(self):
        assert not _is_dangerous_uri("myapp%3A//action/%2520data")

    def test_safe_https_double_encoded_path(self):
        assert not _is_dangerous_uri("https://safe.example/path%2520with%2520space")

    def test_safe_https_triple_encoded_path(self):
        assert not _is_dangerous_uri("https://safe.example/path%252520triple")

    def test_safe_http_encoded_query(self):
        assert not _is_dangerous_uri("https://x.com/?next=%252Fdashboard")

    def test_safe_mailto(self):
        assert not _is_dangerous_uri("mailto:user@example.com?subject=Hello%20World")

    def test_safe_relative_encoded(self):
        assert not _is_dangerous_uri("./file%2520name.md")

    def test_safe_fragment_encoded(self):
        assert not _is_dangerous_uri("#section%201")


class TestSafeEncodedInMarkdown:
    """Safe encoded URLs preserved in markdown output."""

    @pytest.mark.parametrize(
        "url",
        [
            "https://safe.example/path%2520with%2520space",
            "https://x.com/?q=%2525encoded",
            "http://example.org/double%2520encode",
        ],
        ids=["double-enc-path", "double-enc-query", "double-enc-http"],
    )
    def test_preserved(self, tmp_path, url):
        md = f"[link]({url})"
        path = tmp_path / "s.md"
        path.write_bytes(md.encode("utf-8"))
        result = extract_md(path)
        assert f"[link]({url})" in result


class TestCompletionProbes:
    """Final probes for all three reported issues."""

    def test_oversized_unresolved_rejected(self):
        """'%25' * 100 + 'javascript:x' must be True."""
        assert _is_dangerous_uri("%25" * 100 + "javascript:x")

    def test_webcal_custom_scheme_accepted(self):
        """'webcal%3A//safe.example/path%252520x' must be False."""
        assert not _is_dangerous_uri("webcal%3A//safe.example/path%252520x")

    def test_safe_https_double_encoded_path(self):
        """'https://safe.example/path%2520with%2520space' must be False."""
        assert not _is_dangerous_uri("https://safe.example/path%2520with%2520space")


# ═══ Restored Deep-Encoding Tests (with short IDs) ═══════════════════════


class TestDeepEncodedSchemes:
    """Deeply encoded dangerous schemes at various depths."""

    @pytest.mark.parametrize(
        "uri",
        [
            _encode_to_depth("javascript:alert(1)", 1),
            _encode_to_depth("javascript:alert(1)", 3),
            _encode_to_depth("javascript:alert(1)", 8),
            _encode_to_depth("data:text/html,x", 5),
            _encode_to_depth("vbscript:x", 4),
            _encode_to_depth("file:///etc/passwd", 6),
        ],
        ids=["js-d1", "js-d3", "js-d8", "data-d5", "vbs-d4", "file-d6"],
    )
    def test_deep_encoded_rejected(self, uri):
        assert _is_dangerous_uri(uri)

    @pytest.mark.parametrize(
        "uri",
        [
            "java\tscript:alert(1)",
            "java%09script:alert(1)",
            "java&#9;script:alert(1)",
            "java\nscript:x",
            "java%0Ascript:x",
            "java&#10;script:x",
            "java\rscript:x",
            "java%0Dscript:x",
            "java&#13;script:x",
            "vb%0Ascript:x",
            "da%0Dta:text/html,x",
            "fi&#9;le:///etc/passwd",
        ],
        ids=[
            "tab-lit",
            "tab-pct",
            "tab-ent",
            "lf-lit",
            "lf-pct",
            "lf-ent",
            "cr-lit",
            "cr-pct",
            "cr-ent",
            "vb-lf",
            "data-cr",
            "file-tab",
        ],
    )
    def test_embedded_controls_rejected(self, uri):
        assert _is_dangerous_uri(uri)

    @pytest.mark.parametrize(
        "uri",
        [
            "java%73cript:alert(1)",
            "javascript%3Aalert(1)",
            "javascript%253Aalert(1)",
            "javascript%2525253Aalert(1)",
            "javascript&#58;alert(1)",
            "javascript&#00058;alert(1)",
            "javascript&#x3a;alert(1)",
        ],
        ids=["pct-s", "pct-colon", "dbl-colon", "quad-colon", "ent-dec", "ent-pad", "ent-hex"],
    )
    def test_encoded_colons_rejected(self, uri):
        assert _is_dangerous_uri(uri)


class TestDeepEncodedMarkdown:
    """Encoded schemes in Markdown inline links, images, and refs."""

    @pytest.mark.parametrize(
        "uri",
        [
            _encode_to_depth("javascript:x", 1),
            _encode_to_depth("data:x", 1),
            _encode_to_depth("vbscript:x", 1),
            _encode_to_depth("file:x", 1),
        ],
        ids=["js-inline", "data-inline", "vbs-inline", "file-inline"],
    )
    def test_inline_link_neutralized(self, tmp_path, uri):
        path = tmp_path / "t.md"
        path.write_bytes(f"[click]({uri})".encode())
        result = extract_md(path)
        assert result.strip() == "click"
        assert uri not in result
        assert "](" not in result

    @pytest.mark.parametrize(
        "uri",
        [
            _encode_to_depth("javascript:x", 1),
            _encode_to_depth("data:x", 1),
        ],
        ids=["js-img", "data-img"],
    )
    def test_image_neutralized(self, tmp_path, uri):
        path = tmp_path / "t.md"
        path.write_bytes(f"![img]({uri})".encode())
        result = extract_md(path)
        assert result.strip() == "img"
        assert uri not in result
        assert "](" not in result

    @pytest.mark.parametrize(
        "uri",
        [
            _encode_to_depth("javascript:x", 1),
            _encode_to_depth("data:x", 1),
            _encode_to_depth("vbscript:x", 1),
            _encode_to_depth("file:x", 1),
        ],
        ids=["js-ref", "data-ref", "vbs-ref", "file-ref"],
    )
    def test_ref_definition_neutralized(self, tmp_path, uri):
        path = tmp_path / "t.md"
        md = f"[x][r]\n\n[r]: {uri}\n"
        path.write_bytes(md.encode("utf-8"))
        result = extract_md(path)
        assert "[r]: #" in result
        assert uri not in result


class TestSchemeBoundaryLengths:
    """Custom scheme lengths at and around configured boundaries."""

    @pytest.mark.parametrize(
        "length",
        [1, 10, 30, 31, 40, 60, 100, 124],
        ids=["len1", "len10", "len30", "len31", "len40", "len60", "len100", "len124"],
    )
    def test_custom_scheme_accepted(self, length):
        scheme = "x" * length
        uri = f"{scheme}%3A//safe.example/"
        assert not _is_dangerous_uri(uri), f"len={length} rejected"

    def test_scheme_at_candidate_limit(self):
        # 125 x's + %3A = 128 chars exactly; colon visible after decode
        scheme = "x" * 125
        uri = f"{scheme}%3A//x"
        assert not _is_dangerous_uri(uri)

    def test_scheme_beyond_candidate_limit_unresolved(self):
        # 126 x's + %3A starts at pos 126; first 128 chars = "x"*126 + "%3"
        # truncated, %3A split → no colon after decode → fail closed
        scheme = "x" * 126
        uri = f"{scheme}%3A//x"
        assert _is_dangerous_uri(uri)


class TestNormBoundaryRestore:
    """Normalization boundary and long URI tests."""

    def test_long_safe_https(self):
        assert not _is_dangerous_uri("https://example.com/" + "a" * 200)

    def test_long_safe_https_encoded_path(self):
        assert not _is_dangerous_uri("https://example.com/path%20with%20spaces/" + "x" * 100)

    def test_long_dangerous(self):
        assert _is_dangerous_uri("javascript:" + "a" * 200)

    def test_long_relative(self):
        assert not _is_dangerous_uri("./" + "dir/" * 50 + "file.md")

    def test_safe_fragment_encoded(self):
        assert not _is_dangerous_uri("#section%201")

    def test_safe_https_pct_near_boundary(self):
        assert not _is_dangerous_uri("https://example.com/012345678901234%20end")

    def test_oversized_unresolved(self):
        assert _is_dangerous_uri("%25" * 100 + "javascript:x")

    def test_dangerous_after_bound(self):
        """Content after the 128-char bound doesn't save a dangerous prefix."""
        assert _is_dangerous_uri(_encode_to_depth("javascript:x", 1) + "https://safe")


class TestSafeDestinationsRestore:
    """Safe encoded destinations preserved unchanged."""

    @pytest.mark.parametrize(
        "url",
        [
            "https://safe.example/path%20with%20spaces",
            "https://safe.example/?next=%2Fdashboard",
            "http://example.org/a%20b",
            "mailto:user@example.com?subject=Hello%20World",
            "./relative%20file.md",
            "../docs/file%20name.md",
            "#section%201",
            "https://safe.example/path%2520with%2520space",
            "https://safe.example/path%252520triple",
            "https://x.com/?next=%252Fdashboard",
        ],
        ids=[
            "https-space",
            "https-query",
            "http-space",
            "mailto-subject",
            "rel-space",
            "rel-parent",
            "frag-space",
            "dbl-enc-path",
            "triple-enc-path",
            "dbl-enc-query",
        ],
    )
    def test_safe_url_not_rejected(self, url):
        assert not _is_dangerous_uri(url)

    @pytest.mark.parametrize(
        "url",
        [
            "https://safe.example/path%2520with%2520space",
            "https://x.com/?q=%2525encoded",
            "http://example.org/double%2520encode",
        ],
        ids=["dbl-path-md", "dbl-query-md", "dbl-http-md"],
    )
    def test_safe_in_markdown(self, tmp_path, url):
        path = tmp_path / "s.md"
        path.write_bytes(f"[link]({url})".encode())
        result = extract_md(path)
        assert f"[link]({url})" in result


# ═══ Strengthened Markdown Sanitization Exact-Output Tests ════════════════


class TestExactMarkdownNeutralization:
    """Verify dangerous destinations are completely absent from output."""

    @pytest.mark.parametrize(
        "scheme_uri",
        [
            "javascript:alert(1)",
            "data:text/html,hello",
            "vbscript:MsgBox(1)",
            "file:///etc/passwd",
        ],
        ids=["js-exact", "data-exact", "vbs-exact", "file-exact"],
    )
    def test_inline_link_exact_output(self, tmp_path, scheme_uri):
        path = tmp_path / "t.md"
        path.write_bytes(f"[click]({scheme_uri})".encode())
        result = extract_md(path)
        assert result.strip() == "click"
        assert scheme_uri not in result
        assert "](" not in result

    @pytest.mark.parametrize(
        "scheme_uri",
        [
            "javascript:alert(1)",
            "data:image/svg+xml,<svg/>",
            "vbscript:x",
            "file:///x",
        ],
        ids=["js-img-ex", "data-img-ex", "vbs-img-ex", "file-img-ex"],
    )
    def test_image_exact_output(self, tmp_path, scheme_uri):
        path = tmp_path / "t.md"
        path.write_bytes(f"![alt]({scheme_uri})".encode())
        result = extract_md(path)
        assert result.strip() == "alt"
        assert scheme_uri not in result
        assert "](" not in result

    @pytest.mark.parametrize(
        "scheme_uri",
        [
            "javascript:alert(1)",
            "data:text/html,x",
            "vbscript:x",
            "file:///etc/passwd",
        ],
        ids=["js-ref-ex", "data-ref-ex", "vbs-ref-ex", "file-ref-ex"],
    )
    def test_ref_definition_exact(self, tmp_path, scheme_uri):
        path = tmp_path / "t.md"
        md = f"[x][r]\n\n[r]: {scheme_uri}\n"
        path.write_bytes(md.encode("utf-8"))
        result = extract_md(path)
        assert "[r]: #" in result
        assert scheme_uri not in result

    @pytest.mark.parametrize(
        "uri",
        [
            "java%73cript:x",
            "javascript&#58;x",
            "java%09script:x",
            " javascript:x",
        ],
        ids=["pct-s-ex", "entity-ex", "tab-ex", "space-ex"],
    )
    def test_encoded_inline_exact(self, tmp_path, uri):
        path = tmp_path / "t.md"
        path.write_bytes(f"[txt]({uri})".encode())
        result = extract_md(path)
        assert result.strip() == "txt"
        assert uri not in result
        assert "](" not in result
