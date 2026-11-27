"""Контракт данных между частями агента: конвейер (pipeline) отдаёт AnalysisResult, отчёт (report) его печатает."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

Severity = Literal["critical", "major", "minor"]  # критично / существенно / замечание


class Requirement(BaseModel):
    """Отдельное требование, выделенное из документа."""
    report_id: str      # идентификатор в отчёте: R-01, R-02, …
    doc_id: str | None  # номер пункта в документе, если он был
    section: str        # раздел документа: путь по заголовкам «Раздел / Подраздел»
    quote: str          # дословная цитата из документа
    short: str          # краткая формулировка


class Finding(BaseModel):
    """Нарушение критерия качества в одном требовании."""
    req_id: str         # report_id требования
    criterion: str      # нарушенный критерий (prompts.CRITERIA)
    severity: Severity
    quote: str          # дословная цитата-доказательство
    problem: str        # в чём проблема
    fix: str            # что исправить


class RequirementCheck(BaseModel):
    """Результат проверки одного требования."""
    req_id: str
    verdict: Literal["ok", "violations"]
    findings: list[Finding]


class SetFinding(BaseModel):
    """Дефект набора требований целиком (противоречие между пунктами, пробел, шаблон и т. п.)."""
    kind: str                 # вид (prompts.SET_KINDS)
    severity: Severity
    related_req_ids: list[str]
    quote: str                # дословная цитата; несколько мест — через « || »
    problem: str
    fix: str


class RunMeta(BaseModel):
    """Паспорт прогона."""
    document_name: str
    date: str                 # ISO 8601
    model: str
    agent_version: str        # короткий хеш git-коммита (с -dirty, если код изменён) или тег


class AnalysisResult(BaseModel):
    """Итог анализа документа."""
    requirements: list[Requirement]
    checks: list[RequirementCheck]
    set_findings: list[SetFinding]
