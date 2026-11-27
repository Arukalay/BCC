"""Конвейер нарезки документа на требования и их валидации (владелец: участник B).

ВНИМАНИЕ: Это временная заглушка для сквозного запуска (интерфейс B -> C).
Основная логика реализуется участником B в ветке `feat/analysis`.
"""

from validator.schemas import AnalysisResult, Requirement, RequirementCheck


def run(text: str) -> AnalysisResult:
    """Анализирует текст документа требований и возвращает структурированный результат."""
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), text[:120])
    stub_req = Requirement(
        report_id="R-01",
        doc_id=None,
        section="Общие сведения (заглушка конвейера)",
        quote=first_line,
        short="Тестовое извлечение первого непустого фрагмента документа (заглушка B)",
    )
    stub_check = RequirementCheck(
        req_id="R-01",
        verdict="ok",
        findings=[],
    )
    return AnalysisResult(
        requirements=[stub_req],
        checks=[stub_check],
        set_findings=[],
    )
