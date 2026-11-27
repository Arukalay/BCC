"""Тесты loader.py. Контракт: loader.load(path) -> str."""
import pytest

from validator import loader


def _minimal_pdf(text: str) -> bytes:
    """Собирает минимальный валидный PDF с одной строкой ASCII-текста."""
    stream = f"BT /F1 18 Tf 50 700 Td ({text}) Tj ET".encode("latin-1")
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


def test_load_markdown(tmp_path):
    p = tmp_path / "req.md"
    p.write_text("# Заголовок\nТребование Ф-1: порог.\n", encoding="utf-8")
    text = loader.load(str(p))
    assert "Требование Ф-1" in text and "Заголовок" in text


def test_load_txt(tmp_path):
    p = tmp_path / "req.txt"
    p.write_text("Простой текст", encoding="utf-8")
    assert "Простой текст" in loader.load(str(p))


def test_load_pdf(tmp_path):
    p = tmp_path / "req.pdf"
    p.write_bytes(_minimal_pdf("Hello requirements 123"))
    assert "Hello requirements 123" in loader.load(str(p))


def test_unknown_format_raises(tmp_path):
    p = tmp_path / "req.xyz"
    p.write_text("data")
    with pytest.raises((ValueError, NotImplementedError, TypeError)):
        loader.load(str(p))


def test_missing_file_raises(tmp_path):
    with pytest.raises(OSError):
        loader.load(str(tmp_path / "nope.md"))


def test_normalize_collapses_whitespace():
    assert loader.normalize("a   b\n\n c") == "a b c"


def test_load_sample_pdf_with_cyrillic_and_tables():
    """PDF-образец собран из текста requirements.md (таблицы сохранены). Это не оригинал организаторов."""
    from pathlib import Path
    pdf = Path(__file__).resolve().parent / "data" / "requirements_sample.pdf"
    text = " ".join(loader.load(str(pdf)).split())
    for fragment in (
        "На карту приходится максимум один такой push за календарный месяц.",
        "Месяц определяется по дате совершения покупки в Алматы.",
        "Сервис передаёт push на отправку в течение минуты после получения события.",
        "Для чужой карты возвращается 404, как и для отсутствующей карты.",
    ):
        assert fragment in text
