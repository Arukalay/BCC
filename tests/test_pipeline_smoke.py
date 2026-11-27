"""Механика конвейера на моке модели: цитаты, порядок, проверки, повторы, деление документа (без LLM)."""
from pathlib import Path

import pytest

from tests.helpers import normalize  # сверочная: без разметки markdown, как цитаты в отчёте
from tests.mock_llm import MockLLM
from validator import pipeline, prompts
from validator.schemas import Finding, RequirementCheck, SetFinding

DOC = Path(__file__).resolve().parent.parent / "docs" / "requirements.md"
pytestmark = pytest.mark.skipif(not DOC.exists(), reason="docs/requirements.md отсутствует")


@pytest.fixture(scope="module")
def text() -> str:
    return DOC.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def run(text):
    llm = MockLLM()
    return pipeline.run(text, ask_json=llm), llm


def test_all_quotes_are_from_the_document(run, text):
    result, _ = run
    norm = normalize(text)
    flat = " ".join(text.replace("**", "").replace("`", "").split())  # документ без разметки
    quotes = [r.quote for r in result.requirements]
    quotes += [f.quote for c in result.checks for f in c.findings] + [s.quote for s in result.set_findings]
    for quote in quotes:
        for part in quote.split(" || "):
            assert normalize(part) in norm and part in flat, part   # исходный текст документа без разметки


def test_slicing_drops_invented_and_duplicates_and_keeps_document_order(run, text):
    result, _ = run
    shorts = [r.short for r in result.requirements]
    assert "Выдуманное требование" not in shorts
    assert "Дубль Ф-1" not in shorts and shorts.count("Пункт Ф-1") == 1
    assert "НФТ: push за минуту" in shorts                     # цитата с многоточием исправлена повторным запросом
    assert [r.report_id for r in result.requirements] == [f"R-{i:02d}" for i in range(1, len(shorts) + 1)]
    norm = normalize(text)
    pos = [norm.find(normalize(r.quote)) for r in result.requirements]
    assert -1 not in pos and pos == sorted(pos)
    assert all(r.doc_id != "нет" for r in result.requirements) and any(r.doc_id is None for r in result.requirements)


def test_every_requirement_checked_once_findings_sorted_and_canonical(run):
    result, _ = run
    assert [c.req_id for c in result.checks] == [r.report_id for r in result.requirements]
    for c in result.checks:
        assert c.verdict == ("violations" if c.findings else "ok")
        assert all(f.req_id == c.req_id and f.criterion in prompts.CRITERIA for f in c.findings)
        ranks = [pipeline.SEVERITY_RANK[f.severity] for f in c.findings]
        assert ranks == sorted(ranks)
        assert all(f.problem != "Выдумка." for f in c.findings)
    assert any(c.verdict == "ok" for c in result.checks)


def test_set_findings_cleaned(run):
    result, llm = run
    kinds = [s.kind for s in result.set_findings]
    assert kinds == ["противоречие", "шаблон"]                  # выдумка отброшена, вид приведён к канону
    assert result.set_findings[0].related_req_ids == ["R-03"]   # несуществующий id убран
    assert " || " in result.set_findings[0].quote
    assert len(llm.calls) <= 25


def test_doc_resolve():
    doc = pipeline._Doc(DOC.read_text(encoding="utf-8"))
    assert doc.resolve("Ф-1. Держатель активной карты") is not None       # разметка опущена моделью
    assert doc.resolve("ф-1.   держатель  АКТИВНОЙ карты") is not None     # пробелы и регистр
    assert doc.resolve("Нет такого текста в документе") is None
    assert doc.resolve("ab") is None                                        # слишком коротко
    assert doc.resolve("Ф-3. Когда расходы переходят").quote == "Ф-3. Когда расходы переходят"  # без разметки **
    assert doc.resolve("Нет такого || Ф-1. Держатель") is None              # одна часть неверна


def test_chunks_follow_markdown_headings():
    big = "# Док\n\n## Часть A\n" + "### Раздел\nтекст\n" * 4000 + "## Часть B\n" + "### Т\nх\n" * 4000
    chunks = pipeline._chunks(big)
    assert len(chunks) > 1 and "".join(c for _, c in chunks) == big
    assert all(len(c) <= pipeline.MAX_CHUNK_CHARS for _, c in chunks)
    assert chunks[0][0] == "начало документа" and chunks[1][0] == "Часть A" and chunks[-1][0] == "Часть B"


def test_chunks_without_headings_split_by_paragraphs_and_long_lines():
    """Текст из PDF/TXT: заголовков markdown нет, делим по абзацам; сверхдлинная строка режется."""
    text = "".join(f"Пункт {i}. Требование номер {i} к системе.\n\n" for i in range(3000)) + "x" * 70_000
    chunks = pipeline._chunks(text)
    assert "".join(c for _, c in chunks) == text
    assert all(len(c) <= pipeline.MAX_CHUNK_CHARS for _, c in chunks)
    assert chunks[0][0].startswith("Пункт 0.") and chunks[1][0].startswith("Пункт ")


def test_short_document_is_one_chunk():
    assert pipeline._chunks("Короткий документ.") == [("", "Короткий документ.")]


def test_criteria_lists_match_prompt_texts():
    for name in prompts.CRITERIA:
        assert f"- {name} —" in prompts.REQUIREMENT_CRITERIA, name
    for name in prompts.SET_KINDS:
        assert f"«{name}»" in prompts.SET_CRITERIA, name


def test_dedup_keeps_different_problems_and_merges_repeats(text):
    q_req = "Повторная доставка одного события процессинга сохраняет исходный итог."
    q_set = "На карту приходится максимум один такой push за календарный месяц."

    def finding(severity, problem):
        return Finding(req_id="R-01", criterion="полнота", severity=severity, quote=q_req, problem=problem, fix="Исправить.")

    def gap(severity, problem):
        return SetFinding(kind="пробел", severity=severity, related_req_ids=["R-01"], quote=q_set, problem=problem, fix="Добавить.")

    def fake(prompt, schema, system=None):
        name = schema.__name__
        if name == "SlicingResult":
            return pipeline.SlicingResult(requirements=[pipeline.RawRequirement(
                doc_id=None, section="Интеграция", quote=q_req, short="Повтор события")])
        if name == "BatchChecks":
            return pipeline.BatchChecks(checks=[RequirementCheck(req_id="R-01", verdict="violations", findings=[
                finding("minor", "Не определено, как распознаётся повтор: нет ключа события."),
                finding("major", "Не сказано, что делать с повторным push."),
                finding("critical", "Не определён ключ, по которому распознаётся повтор события."),
            ])])
        if name == "SetResult":
            return pipeline.SetResult(findings=[
                gap("major", "Нет сценария отката функционала."),
                gap("major", "Нет требований информационной безопасности."),
                gap("critical", "Отсутствуют требования информационной безопасности."),
                gap("major", "Не описаны отмены и частичные возвраты."),
            ])
        raise AssertionError(name)

    res = pipeline.run(text, ask_json=fake)
    found = res.checks[0].findings
    assert len(found) == 2 and found[0].severity == "critical"          # повтор про ключ склеен, выжила критичная
    assert any("повторным push" in f.problem for f in found)             # другая проблема с той же цитатой осталась
    gaps = res.set_findings
    assert len(gaps) == 3 and gaps[0].problem.startswith("Отсутствуют")  # повтор про ИБ склеен (4 прохода → один раз)
