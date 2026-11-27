"""Генерация итогового Markdown-отчёта по результатам анализа (владелец: участник C).

ВНИМАНИЕ: Это временная заглушка для сквозного запуска (интерфейс C -> CLI).
Основная логика отчёта реализуется участником C в ветке `feat/report`.
"""

from validator.schemas import AnalysisResult, RunMeta


def build(meta: RunMeta, result: AnalysisResult) -> str:
    """Собирает Markdown-отчёт по метаданным запуска и результатам анализа."""
    total_reqs = len(result.requirements)
    req_violations = sum(len(c.findings) for c in result.checks)
    set_violations = len(result.set_findings)

    lines = [
        "# Отчёт статического анализа требований (предварительная сборка)",
        "",
        "## 1. Паспорт прогона",
        f"- **Документ:** `{meta.document_name}`",
        f"- **Дата:** `{meta.date}`",
        f"- **Модель:** `{meta.model}`",
        f"- **Версия агента:** `{meta.agent_version}`",
        "",
        "## 2. Нарезка требований",
        "| ID в отчёте | ID в документе | Краткая формулировка |",
        "| --- | --- | --- |",
    ]
    for req in result.requirements:
        doc_id = req.doc_id or "—"
        lines.append(f"| {req.report_id} | {doc_id} | {req.short} |")

    lines.extend([
        "",
        "## 3. Сводка",
        f"- Выделено требований: **{total_reqs}**",
        f"- Нарушений в отдельных требованиях: **{req_violations}**",
        f"- Дефектов на уровне набора требований: **{set_violations}**",
        "",
        "> Примечание: используется базовая сборка отчёта до влития ветки `feat/report`.",
    ])
    return "\n".join(lines) + "\n"
