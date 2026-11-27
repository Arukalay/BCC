"""Разбор аргументов командной строки и запуск сквозного конвейера проверки."""

import argparse
from datetime import datetime
import logging
from pathlib import Path
import subprocess
import sys

from validator import llm, loader, pipeline, report
from validator.loader import DocumentLoadError
from validator.llm import LLMError
from validator.schemas import RunMeta

logger = logging.getLogger(__name__)


def _setup_logging(verbose: bool) -> None:
    """Настраивает логирование в stderr."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )


REPO_DIR = Path(__file__).resolve().parent.parent


def get_agent_version() -> str:
    """Короткий git-хеш коммита агента (с суффиксом -dirty, если код изменён после коммита) или 'unknown'.

    git запускается в папке проекта, а не в текущей: версия не зависит от того, откуда запущен агент.
    Неотслеживаемые файлы (старые отчёты, .env) не считаются правкой кода.
    """
    try:
        commit_res = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=REPO_DIR,
        )
        version = commit_res.stdout.strip()
        if not version:
            return "unknown"

        status_res = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=False,
            cwd=REPO_DIR,
        )
        if status_res.returncode == 0 and status_res.stdout.strip():
            version = f"{version}-dirty"
        return version
    except Exception:
        return "unknown"


def _build_parser() -> argparse.ArgumentParser:
    """Создаёт парсер аргументов командной строки."""
    parser = argparse.ArgumentParser(
        prog="python -m validator",
        description="Статический ИИ-валидатор документов требований (BCC-HUB x MNU FinTech).",
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        help="Путь к входному документу требований (.md, .txt, .pdf, .docx)",
    )
    parser.add_argument(
        "output_path",
        nargs="?",
        help="Путь для сохранения итогового отчёта (.md)",
    )
    parser.add_argument(
        "--check-llm",
        action="store_true",
        help="Проверить доступность модели Gemini по ключу из .env и выйти",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Включить подробное логирование (уровень DEBUG)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Главная функция CLI.

    Коды возврата: 0 — отчёт записан; 1 — непредвиденная ошибка; 2 — входной документ не прочитан;
    3 — ошибка модели (ключ, квота, недоступность); 4 — отчёт не удалось записать; 130 — прервано.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)

    if args.check_llm:
        try:
            ok, msg = llm.check_llm()
            print(msg)
            return 0 if ok else 1
        except LLMError as exc:
            print(f"[ОШИБКА LLM] {exc}", file=sys.stderr)
            return 1
        except Exception as exc:
            print(f"[ОШИБКА] Непредвиденный сбой при проверке LLM: {exc}", file=sys.stderr)
            return 1

    if not args.input_path or not args.output_path:
        parser.error(
            "Необходимо указать <путь_к_требованиям> и <путь_к_отчёту> "
            "(либо использовать флаг --check-llm)."
        )

    input_file = Path(args.input_path)
    output_file = Path(args.output_path)

    try:
        logger.info("Загрузка документа требований: %s", input_file)
        document_text = loader.load(str(input_file))
        logger.debug("Загружено символов: %d", len(document_text))
    except (OSError, ValueError, DocumentLoadError) as exc:
        print(f"[ОШИБКА ВХОДНОГО ДОКУМЕНТА] {exc}", file=sys.stderr)
        return 2

    try:
        logger.info("Запуск конвейера анализа требований...")
        analysis_result = pipeline.run(document_text)
        meta = RunMeta(
            document_name=input_file.name,
            date=datetime.now().astimezone().isoformat(timespec="seconds"),
            model=llm.model_name(),
            agent_version=get_agent_version(),
        )
        logger.info("Сборка итогового отчёта...")
        report_markdown = report.build(meta, analysis_result)
    except LLMError as exc:
        print(f"[ОШИБКА LLM] {exc}", file=sys.stderr)
        return 3
    except KeyboardInterrupt:
        print("\nПрервано пользователем, отчёт не записан.", file=sys.stderr)
        return 130
    except Exception as exc:
        logger.exception("Непредвиденная ошибка при выполнении валидатора")
        print(f"[ОШИБКА ВЫПОЛНЕНИЯ] {exc}", file=sys.stderr)
        return 1

    try:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(report_markdown, encoding="utf-8")
    except OSError as exc:
        print(f"[ОШИБКА ЗАПИСИ ОТЧЁТА] {exc}", file=sys.stderr)
        return 4
    logger.info("Отчёт успешно сохранён: %s", output_file)
    print(f"Готово! Отчёт сохранён в '{output_file}'.")
    return 0
