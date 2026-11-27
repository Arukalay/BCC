"""Общие Pydantic-схемы и контракты данных для валидатора требований."""

from typing import Literal
from pydantic import BaseModel


class Requirement(BaseModel):
    """Единичное требование, выделенное из документа."""

    report_id: str      # "R-01"... идентификатор в отчёте
    doc_id: str | None  # идентификатор в документе ("Ф-1", "НФТ", ...), если был
    section: str        # раздел документа ("Часть A / Функциональные требования")
    quote: str          # ДОСЛОВНАЯ цитата из документа
    short: str          # краткая формулировка


class Finding(BaseModel):
    """Нарушение критерия качества в конкретном требовании."""

    req_id: str         # report_id требования
    criterion: str      # нарушенный критерий ("однозначность", "проверяемость"...)
    severity: Literal["critical", "major", "minor"]  # критично / существенно / замечание
    quote: str          # дословная цитата-доказательство из документа
    problem: str        # в чём проблема
    fix: str            # что исправить


class RequirementCheck(BaseModel):
    """Результат проверки одного требования."""

    req_id: str
    verdict: Literal["ok", "violations"]
    findings: list[Finding]


class SetFinding(BaseModel):
    """Дефект на уровне всего набора требований (противоречие, пробел, шаблон и т.д.)."""

    kind: str           # "противоречие" | "пробел" | "терминология" | "шаблон" | ...
    severity: Literal["critical", "major", "minor"]
    related_req_ids: list[str]
    quote: str          # дословная цитата (или две через " || ")
    problem: str
    fix: str


class RunMeta(BaseModel):
    """Метаданные запуска для паспорта отчёта."""

    document_name: str
    date: str           # ISO
    model: str
    agent_version: str  # git commit (короткий хеш) или тег


class AnalysisResult(BaseModel):
    """Итоговый результат анализа всего документа."""

    requirements: list[Requirement]
    checks: list[RequirementCheck]
    set_findings: list[SetFinding]
