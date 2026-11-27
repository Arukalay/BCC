"""Загрузка документов требований (.md, .txt, .pdf, .docx) и нормализация текста."""

from pathlib import Path
import re

SUPPORTED_EXTENSIONS = (".md", ".txt", ".pdf", ".docx")


class DocumentLoadError(Exception):
    """Ошибка чтения или разбора входного документа требований."""


def normalize(text: str) -> str:
    """Нормализует текст для сверки дословных цитат.

    Схлопывает последовательности пробельных символов в один пробел,
    приводит различные виды кавычек и тире к единому стандартному виду.
    """
    if not text:
        return ""

    # Неразрывные пробелы -> обычный пробел
    normalized = text.replace("\u00a0", " ").replace("\u202f", " ")

    # Единые двойные кавычки
    normalized = re.sub(r"[«»“”„‟]", '"', normalized)

    # Единые одинарные кавычки / апострофы
    normalized = re.sub(r"[‘’‚‛]", "'", normalized)

    # Единые дефисы / тире
    normalized = re.sub(r"[—–−‐‑]", "-", normalized)

    # Схлопывание всех пробельных символов (пробелы, табы, переносы строк)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _load_text_file(file_path: Path) -> str:
    """Читает текстовый файл (.md или .txt) в кодировке UTF-8."""
    try:
        content = file_path.read_text(encoding="utf-8-sig")  # -sig: убирает BOM, если файл сохранён в Windows
    except UnicodeDecodeError as exc:
        raise DocumentLoadError(
            f"Файл '{file_path}' не удалось прочитать в кодировке UTF-8: {exc}"
        ) from exc

    if not content.strip():
        raise DocumentLoadError(f"Файл '{file_path}' пуст.")
    return content


def _load_pdf_file(file_path: Path) -> str:
    """Читает текстовый слой PDF постранично через pypdf.

    Страницы склеиваются переводом строки без служебных меток: текст агента должен совпадать с
    документом, а предложение, перенесённое на следующую страницу, — оставаться цельным.
    """
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise DocumentLoadError(
            "Библиотека 'pypdf' не установлена. Выполните: pip install pypdf"
        ) from exc

    try:
        reader = PdfReader(str(file_path))
    except Exception as exc:
        raise DocumentLoadError(f"Не удалось открыть PDF '{file_path}': {exc}") from exc

    if len(reader.pages) == 0:
        raise DocumentLoadError(f"PDF-файл '{file_path}' не содержит страниц.")

    pages_text = [text.strip() for page in reader.pages if (text := page.extract_text() or "").strip()]
    if not pages_text:
        raise DocumentLoadError(
            f"В PDF-файле '{file_path}' не обнаружен текстовый слой (вероятно, это скан). "
            "Для проверки такого документа требуется предварительное распознавание текста (OCR)."
        )
    return "\n".join(pages_text)


def _format_docx_table(table) -> str:
    """Преобразует таблицу python-docx в Markdown-подобный вид."""
    rows_md: list[str] = []
    for row_idx, row in enumerate(table.rows):
        cells = [
            re.sub(r"\s+", " ", cell.text.strip()).replace("|", "\\|")
            for cell in row.cells
        ]
        rows_md.append("| " + " | ".join(cells) + " |")
        if row_idx == 0 and cells:
            rows_md.append("| " + " | ".join(["---"] * len(cells)) + " |")
    return "\n".join(rows_md)


def _format_docx_paragraph(paragraph) -> str:
    """Форматирует абзац python-docx с сохранением заголовков и списков."""
    text = paragraph.text.strip()
    if not text:
        return ""

    style_name = (paragraph.style.name or "").lower() if paragraph.style else ""

    # Заголовки Heading 1..6 / Заголовок 1..6
    heading_match = re.search(r"(?:heading|заголовок)\s*(\d)", style_name)
    if heading_match:
        level = min(int(heading_match.group(1)), 6)
        return f"{'#' * level} {text}"

    # Элементы списков
    if "list" in style_name or "список" in style_name:
        return f"- {text}"

    return text


def _load_docx_file(file_path: Path) -> str:
    """Читает .docx через python-docx, сохраняя порядок абзацев и таблиц."""
    try:
        from docx import Document
        from docx.oxml.table import CT_Tbl
        from docx.oxml.text.paragraph import CT_P
        from docx.table import Table
        from docx.text.paragraph import Paragraph
    except ImportError as exc:
        raise DocumentLoadError(
            "Библиотека 'python-docx' не установлена. Выполните: pip install python-docx"
        ) from exc

    try:
        doc = Document(str(file_path))
    except Exception as exc:
        raise DocumentLoadError(f"Не удалось открыть DOCX '{file_path}': {exc}") from exc

    blocks: list[str] = []
    for child in doc.element.body.iterchildren():
        if isinstance(child, CT_P):
            para = Paragraph(child, doc)
            formatted = _format_docx_paragraph(para)
            if formatted:
                blocks.append(formatted)
        elif isinstance(child, CT_Tbl):
            tbl = Table(child, doc)
            formatted_tbl = _format_docx_table(tbl)
            if formatted_tbl:
                blocks.append(formatted_tbl)

    if not blocks:
        raise DocumentLoadError(f"DOCX-файл '{file_path}' не содержит текста или таблиц.")

    return "\n\n".join(blocks)


def load(path: str) -> str:
    """Загружает документ требований из файла (.md, .txt, .pdf, .docx) и возвращает текст.

    Args:
        path: Путь к входному файлу документа требований.

    Returns:
        Извлечённый текст документа с сохранённой структурой (заголовки, списки, таблицы).

    Raises:
        FileNotFoundError: Если файл не найден.
        ValueError: Если расширение файла не поддерживается.
        DocumentLoadError: Если файл пуст, повреждён или PDF не содержит текстового слоя.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Файл требований не найден: '{path}'")
    if not file_path.is_file():
        raise ValueError(f"Указанный путь не является файлом: '{path}'")

    ext = file_path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        supported_str = ", ".join(SUPPORTED_EXTENSIONS)
        raise ValueError(
            f"Неподдерживаемый формат файла '{ext}'. "
            f"Поддерживаемые форматы: {supported_str}"
        )

    if ext in (".md", ".txt"):
        return _load_text_file(file_path)
    if ext == ".pdf":
        return _load_pdf_file(file_path)
    if ext == ".docx":
        return _load_docx_file(file_path)

    raise ValueError(f"Неподдерживаемый формат файла: '{ext}'")
