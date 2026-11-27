"""Фейковые данные в формате AnalysisResult для тестов отчёта."""
from __future__ import annotations

from validator.schemas import (AnalysisResult, Finding, Requirement,
                               RequirementCheck, RunMeta, SetFinding)

Q_F1 = "Держатель активной карты KZT задаёт целый порог от 1 000 до 1 000 000 ₸ включительно."
Q_F3A = "На карту приходится максимум один такой push за календарный месяц."
Q_F4 = "Если она ниже нового порога, следующее достижение вызывает push независимо от истории предыдущей настройки."
Q_NFR = "Сервис передаёт push на отправку в течение минуты после получения события."
Q_DATA = "Хранить идентификатор карты, месяц, идентификатор операции"
Q_PIPE = "Колонка A | Колонка B"

# Текст «документа», в котором все цитаты встречаются (перенос строки внутри Q_DATA намеренный).
DOC_TEXT = f"""Функциональные требования
- Ф-1. {Q_F1}
- Ф-3. {Q_F3A}
- Ф-4. {Q_F4}
Нефункциональные требования
{Q_NFR}
Требования к данным
Хранить идентификатор карты,
месяц, идентификатор операции, сумму покупки в тиынах.
{Q_PIPE}
"""

META = RunMeta(document_name="requirements.md", date="2026-11-27",
               model="gemini-test", agent_version="abc1234")


def make_meta() -> RunMeta:
    return META.model_copy()


def _req(i: int, doc_id: str | None, quote: str, short: str, section: str = "Часть A / Функциональные требования"):
    return Requirement(report_id=f"R-{i:02d}", doc_id=doc_id, section=section, quote=quote, short=short)


def make_requirements() -> list[Requirement]:
    return [
        _req(1, "Ф-1", Q_F1, "Порог 1 000–1 000 000 ₸"),
        _req(2, "Ф-3", Q_F3A, "Один push в месяц"),
        _req(3, "Ф-4", Q_F4, "Смена порога"),
        _req(4, None, Q_NFR, "Push в течение минуты", "Часть A / Нефункциональные требования"),
        _req(5, None, "Хранить идентификатор карты, месяц, идентификатор операции",
             "Состав данных", "Часть A / Требования к данным"),
    ]


def make_result() -> AnalysisResult:
    """5 требований; R-01 чистое; R-02 — 1 существенное; R-03 — 1 критичное + 1 замечание;
    R-04 — 1 критичное (непроверяемое); R-05 без проверки. По набору: 1 критичное + 1 замечание.
    Итого: критично 3, существенно 1, замечание 2."""
    checks = [
        RequirementCheck(req_id="R-01", verdict="ok", findings=[]),
        RequirementCheck(req_id="R-02", verdict="violations", findings=[
            Finding(req_id="R-02", criterion="однозначность", severity="major", quote=Q_F3A,
                    problem="Не определён часовой пояс месяца.", fix="Указать месяц по Алматы."),
        ]),
        RequirementCheck(req_id="R-03", verdict="violations", findings=[
            Finding(req_id="R-03", criterion="полнота", severity="minor", quote="независимо от истории предыдущей настройки",
                    problem="Размытая формулировка.", fix="Определить «историю»."),
            Finding(req_id="R-03", criterion="непротиворечивость", severity="critical", quote=Q_F4,
                    problem="Конфликт с Ф-3.", fix="Согласовать с Ф-3."),
        ]),
        RequirementCheck(req_id="R-04", verdict="violations", findings=[
            Finding(req_id="R-04", criterion="проверяемость", severity="critical", quote="в течение минуты",
                    problem="Нет процентиля и нагрузки.\nНет точки замера.", fix="Задать p95 и нагрузку."),
        ]),
    ]
    set_findings = [
        SetFinding(kind="противоречие", severity="critical", related_req_ids=["R-02", "R-03"],
                   quote=f"{Q_F3A} || {Q_F4}", problem="Ф-3 и Ф-4 несовместимы.", fix="Выбрать одно правило."),
        SetFinding(kind="терминология", severity="minor", related_req_ids=["R-05"],
                   quote=Q_PIPE, problem="Вот | символ в цитате.", fix="Привести термины к единому виду."),
    ]
    return AnalysisResult(requirements=make_requirements(), checks=checks, set_findings=set_findings)


def make_result_major_only() -> AnalysisResult:
    r = make_result()
    for c in r.checks:
        c.findings = [f for f in c.findings if f.severity != "critical"]
    r.set_findings = [SetFinding(kind="пробел", severity="major", related_req_ids=["R-01"],
                                 quote=Q_F1, problem="Нет отката.", fix="Добавить откат.")]
    return r


def make_result_clean() -> AnalysisResult:
    reqs = make_requirements()[:2]
    return AnalysisResult(
        requirements=reqs,
        checks=[RequirementCheck(req_id=q.report_id, verdict="ok", findings=[]) for q in reqs],
        set_findings=[],
    )


# ---------------------------------------------------------------- граничные случаи
# Цитаты в том виде, в каком их отдаёт pipeline: дословные фрагменты markdown-текста.
Q_UNCLOSED_BOLD = "Ф-3.** Когда расходы переходят"          # незакрытый **
Q_TABLE_ROW = "| Prod | — | Swagger | — |"                   # строка таблицы с |
Q_BACKTICKS = "`GET /alerts/saveLimit` и *курсив* и _подчёркивание_"
Q_EDGE_BACKTICK_END = "метод `saveLimit`"                      # кончается обратной кавычкой

EDGE_DOC_TEXT = f"""- **{Q_UNCLOSED_BOLD} до порога, клиент получает push.
{Q_TABLE_ROW}
Описание: {Q_BACKTICKS}
Вызывается {Q_EDGE_BACKTICK_END}
Блок согласующих
Подразделение | ФИО ответственного | Подпись
"""


def make_result_edge_cases() -> AnalysisResult:
    """Данные, которые report.build обязан переносить спокойно:
    - SetFinding с пустым related_req_ids («пустой блок согласующих»);
    - Requirement.doc_id=None и section — свободный текст модели (с |, * и переводом строки);
    - у каждого требования есть проверка; у verdict="ok" список findings пустой;
    - цитаты с незакрытым **, строкой таблицы с |, обратными кавычками, * и _;
    - две части через ' || '.
    """
    reqs = [
        Requirement(report_id="R-01", doc_id=None,
                    section="Часть A\n/ **Функциональные** требования | подпункт *3*",
                    quote=Q_UNCLOSED_BOLD, short="Push при достижении порога"),
        Requirement(report_id="R-02", doc_id="Ф-9", section="Часть B / Общие параметры",
                    quote=Q_TABLE_ROW, short="Prod и Swagger не заполнены | пусто"),
        Requirement(report_id="R-03", doc_id=None, section="Часть B",
                    quote=Q_EDGE_BACKTICK_END, short="Вызов метода"),
    ]
    checks = [
        RequirementCheck(req_id="R-01", verdict="ok", findings=[]),
        RequirementCheck(req_id="R-02", verdict="violations", findings=[
            Finding(req_id="R-02", criterion="полнота", severity="major", quote=Q_TABLE_ROW,
                    problem="Prod не задан.", fix="Указать Prod URL."),
            Finding(req_id="R-02", criterion="однозначность", severity="minor", quote=Q_BACKTICKS,
                    problem="Смешаны термины.", fix="Выбрать один термин."),
        ]),
        RequirementCheck(req_id="R-03", verdict="violations", findings=[
            Finding(req_id="R-03", criterion="проверяемость", severity="critical",
                    quote=Q_EDGE_BACKTICK_END, problem="Нет критерия.", fix="Добавить критерий."),
        ]),
    ]
    set_findings = [
        SetFinding(kind="шаблон", severity="major", related_req_ids=[],
                   quote="Блок согласующих", problem="Блок согласующих пуст.",
                   fix="Заполнить согласующих."),
        SetFinding(kind="противоречие", severity="critical", related_req_ids=["R-01", "R-02"],
                   quote=f"{Q_UNCLOSED_BOLD} || {Q_TABLE_ROW}", problem="Конфликт.", fix="Согласовать."),
    ]
    return AnalysisResult(requirements=reqs, checks=checks, set_findings=set_findings)
