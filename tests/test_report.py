"""Отчёт: шесть разделов по ТЗ, числа сводки, читаемость, экранирование, детерминированность (без LLM)."""
import re

from tests.fixtures import (EDGE_DOC_TEXT, Q_BACKTICKS, Q_EDGE_BACKTICK_END, Q_F3A, Q_F4, Q_PIPE, Q_TABLE_ROW,
                            Q_UNCLOSED_BOLD, make_meta, make_result, make_result_clean, make_result_edge_cases,
                            make_result_major_only)
from tests.helpers import extract_report_quotes, find_bad_quotes, normalize, parse_report_findings
from validator.report import TOP_FIXES, build, summarize
from validator.schemas import AnalysisResult, Finding, RequirementCheck, SetFinding

HEADINGS = [
    "## 1. Паспорт прогона",
    "## 2. Как агент нарезал требования",
    "## 3. Проверка каждого требования",
    "## 4. Проверка набора требований целиком",
    "## 5. Сводка",
    "## 6. Итог",
]


def section(text: str, n: int) -> str:
    start = text.index(HEADINGS[n - 1])
    return text[start:text.index(HEADINGS[n])] if n < len(HEADINGS) else text[start:]


# ---------------------------------------------------------------- структура по ТЗ

def test_all_six_sections_in_order_with_title_and_legend():
    text = build(make_meta(), make_result())
    positions = [text.index(h) for h in HEADINGS]
    assert positions == sorted(positions)
    assert text.startswith("# Отчёт о проверке требований: requirements.md")
    assert "**Как читать отчёт.**" in text.split(HEADINGS[0])[0]


def test_passport_has_document_date_model_version():
    passport = section(build(make_meta(), make_result()), 1)
    for value in ("requirements.md", "2026-11-27", "gemini-test", "abc1234"):
        assert value in passport


def test_slicing_table_has_ids_and_quote_only_without_doc_id():
    slicing = section(build(make_meta(), make_result()), 2)
    assert "Агент выделил **5** требований: с номером в документе — 3, без номера — 2." in slicing
    assert "| R-01 | Ф-1 |" in slicing and "| R-04 | — |" in slicing
    assert "`Сервис передаёт push" in slicing          # без номера — дословная цитата
    assert "Держатель активной карты" not in slicing    # с номером — цитата не нужна


# ---------------------------------------------------------------- раздел 3: компактно и понятно

def test_requirement_with_findings_has_heading_quote_and_numbered_findings():
    sec3 = section(build(make_meta(), make_result()), 3)
    assert "### R-03 (Ф-4) · Смена порога" in sec3
    block = sec3.split("### R-03")[1].split("###")[0]
    assert f"> `{Q_F4}`" in block
    assert block.index("**R-03.1 · Критично · непротиворечивость.**") < block.index("**R-03.2 · Замечание · полнота.**")


def test_finding_quote_shown_only_when_it_differs_from_requirement_quote():
    sec3 = section(build(make_meta(), make_result()), 3)
    r03 = sec3.split("### R-03")[1].split("###")[0]
    r031 = r03.split("R-03.1")[1].split("R-03.2")[0]
    assert "Цитата:" not in r031                            # совпадает с цитатой требования — не повторяем
    assert "- Цитата: `независимо от истории предыдущей настройки`" in r03.split("R-03.2")[1]
    assert sec3.count("- Как исправить:") == 4               # у каждого нарушения есть исправление


def test_clean_and_unchecked_requirements_listed_in_one_line():
    sec3 = section(build(make_meta(), make_result()), 3)
    assert "Проверено **4** из 5: с нарушениями — **3**, без замечаний — 1, проверка не выполнена — 1." in sec3
    assert "**Без замечаний (1):** R-01 (Ф-1)." in sec3
    assert "**Проверка не выполнена (1):** R-05" in sec3
    assert "### R-01" not in sec3 and "### R-05" not in sec3  # без отдельных блоков для чистых требований


# ---------------------------------------------------------------- раздел 4

def test_set_findings_grouped_by_kind_with_counts_and_two_part_quotes():
    sec4 = section(build(make_meta(), make_result()), 4)
    assert "Замечаний к набору: **2**." in sec4
    assert "### Противоречие (1)" in sec4 and "### Терминология (1)" in sec4
    assert "- **S-01 · Критично.** Ф-3 и Ф-4 несовместимы." in sec4
    assert "Затронуты требования: R-02 (Ф-3), R-03 (Ф-4)" in sec4
    assert f"- Цитата 1: `{Q_F3A}`" in sec4 and f"- Цитата 2: `{Q_F4}`" in sec4
    assert " || " not in sec4


def test_kind_heading_keeps_case_of_abbreviations():
    result = make_result()
    result.set_findings[1].kind = "API-контракт"
    assert "### API-контракт (1)" in build(make_meta(), result)


def test_empty_related_ids_shown_as_dash_and_sorted_last_within_severity():
    result = make_result_edge_cases()
    result.set_findings.append(SetFinding(kind="шаблон", severity="major", related_req_ids=["R-02"],
                                          quote="Блок согласующих", problem="p", fix="f"))
    block = build(make_meta(), result).split("### Шаблон")[1].split("###")[0]
    assert block.index("Затронуты требования: R-02") < block.index("Затронуты требования: —")


# ---------------------------------------------------------------- разделы 5 и 6

def test_summary_numbers_are_computed_from_data():
    result = make_result()
    s = summarize(result)
    assert (s["requirements"], s["checked"], s["unchecked"], s["with_violations"], s["clean"]) == (5, 4, 1, 3, 1)
    assert s["req"] == {"critical": 2, "major": 1, "minor": 1}
    assert s["set"] == {"critical": 1, "major": 0, "minor": 1}
    assert s["total"] == {"critical": 3, "major": 1, "minor": 2} and s["total_all"] == 6
    sec5 = section(build(make_meta(), result), 5)
    assert "| В отдельных требованиях | 2 | 1 | 1 | 4 |" in sec5
    assert "| В наборе целиком | 1 | 0 | 1 | 2 |" in sec5
    assert "| **Итого** | **3** | **1** | **2** | **6** |" in sec5
    assert "Требований выделено: **5**, с нарушениями: **3**" in sec5


def test_critical_defects_grouped_by_tz_types():
    sec5 = section(build(make_meta(), make_result()), 5)
    assert "### Критичные дефекты (3)" in sec5
    assert "**Противоречивые (2)**" in sec5 and "**Непроверяемые (1)**" in sec5
    assert "**Неполные" not in sec5
    assert "- R-04.1: Нет процентиля и нагрузки." in sec5          # только первое предложение проблемы
    assert "- S-01 · R-02 (Ф-3), R-03 (Ф-4): Ф-3 и Ф-4 несовместимы." in sec5


def test_verdict_stop_with_priorities():
    sec6 = section(build(make_meta(), make_result()), 6)
    assert "**Начинать разработку и тестирование по этому документу нельзя.**" in sec6
    assert "### Что исправить в первую очередь" in sec6
    assert "1. **S-01 · R-02 (Ф-3), R-03 (Ф-4)** — Выбрать одно правило." in sec6
    assert "**R-03.1 (Ф-4)** — Согласовать с Ф-3." in sec6
    assert "Оговорка: 1 из 5 требований не проверены" in sec6


def test_verdict_caveats_when_only_major():
    sec6 = section(build(make_meta(), make_result_major_only()), 6)
    assert "**Начинать можно с оговорками.**" in sec6 and "нельзя" not in sec6


def test_verdict_ok_when_clean():
    text = build(make_meta(), make_result_clean())
    assert "**Начинать разработку и тестирование можно.**" in section(text, 6)
    assert "Замечаний к набору нет." in text and "Критичных дефектов нет." in text


def test_priority_list_is_capped():
    result = make_result_clean()
    result.checks[0] = RequirementCheck(req_id="R-01", verdict="violations", findings=[
        Finding(req_id="R-01", criterion="полнота", severity="critical", quote="q", problem=f"П{i}.", fix=f"И{i}.")
        for i in range(TOP_FIXES + 3)])
    sec6 = section(build(make_meta(), result), 6)
    assert f"{TOP_FIXES}. **" in sec6 and f"{TOP_FIXES + 1}. **" not in sec6
    assert "Остальные нарушения той же критичности (3)" in sec6


# ---------------------------------------------------------------- вёрстка и устойчивость

def test_tables_have_consistent_columns_and_escaped_pipes():
    result = make_result()
    result.requirements[3].quote = Q_PIPE                     # у R-04 нет doc_id — цитата попадёт в ячейку
    slicing = section(build(make_meta(), result), 2)
    rows = [line for line in slicing.splitlines() if line.startswith("|")]
    assert len({len(re.findall(r"(?<!\\)\|", line)) for line in rows}) == 1
    assert "`Колонка A \\| Колонка B`" in slicing


def test_newlines_in_model_text_are_flattened():
    assert "Нет процентиля и нагрузки. Нет точки замера." in build(make_meta(), make_result())


def test_markdown_in_quotes_goes_into_code_spans():
    text = build(make_meta(), make_result_edge_cases())
    assert f"`{Q_UNCLOSED_BOLD}`" in text                       # незакрытый ** не вылезает наружу
    assert f"`{Q_TABLE_ROW}`" in text                           # | в списке не экранируем: это code-span
    assert f"`` {Q_BACKTICKS} ``" in text                       # внутри ` — двойной забор
    assert f"`` {Q_EDGE_BACKTICK_END} ``" in text


def test_free_text_section_with_pipes_survives_in_table():
    slicing = section(build(make_meta(), make_result_edge_cases()), 2)
    rows = [line for line in slicing.splitlines() if line.startswith("| R-")]
    assert len(rows) == 3 and "| R-01 | — |" in rows[0]
    assert len({len(re.findall(r"(?<!\\)\|", line)) for line in rows}) == 1


def test_quotes_roundtrip_and_exist_in_document():
    text = build(make_meta(), make_result_edge_cases())
    quotes = [normalize(q) for q in extract_report_quotes(text)]
    for q in (Q_UNCLOSED_BOLD, Q_TABLE_ROW, Q_BACKTICKS, Q_EDGE_BACKTICK_END):
        assert normalize(q) in quotes
    assert find_bad_quotes(text, EDGE_DOC_TEXT) == []


def test_every_finding_is_parsed_with_quote_and_problem():
    result = make_result()
    findings = parse_report_findings(build(make_meta(), result))
    expected = sum(len(c.findings) for c in result.checks) + len(result.set_findings)
    assert len(findings) == expected
    assert all(f["quotes"] and f["problem"] for f in findings)


def test_deterministic():
    assert build(make_meta(), make_result()) == build(make_meta(), make_result())


def test_empty_result_does_not_crash():
    text = build(make_meta(), AnalysisResult(requirements=[], checks=[], set_findings=[]))
    assert all(h in text for h in HEADINGS)
    assert "**Начинать разработку и тестирование можно.**" in text


def test_long_lists_of_related_requirements_are_shortened():
    result = make_result()
    ids = [r.report_id for r in result.requirements]
    result.set_findings[0].related_req_ids = ids                 # 5 требований
    text = build(make_meta(), result)
    assert "S-01 · R-01 (Ф-1), R-02 (Ф-3), R-03 (Ф-4) и ещё 2" in section(text, 5)     # в сводке — первые 3
    assert "Затронуты требования: R-01 (Ф-1), R-02 (Ф-3), R-03 (Ф-4), R-04, R-05" in section(text, 4)
