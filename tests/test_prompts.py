"""Структура промптов и проходов проверки набора, отсутствие хардкода (без LLM)."""
from pathlib import Path

import pytest

from validator import pipeline, prompts
from validator.schemas import Requirement

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "docs" / "requirements.md"


def _reqs() -> list[Requirement]:
    return [Requirement(report_id="R-01", doc_id="1.1", section="Раздел", quote="Текст требования.", short="Кратко")]


def test_set_passes_are_well_formed():
    assert len(prompts.SET_PASSES) >= 3
    for name, p in prompts.SET_PASSES.items():
        assert p["title"] and p["steps"].strip(), name
        assert p["kinds"] and set(p["kinds"]) <= set(prompts.SET_KINDS), name


def test_every_set_kind_has_a_pass():
    covered = {k for p in prompts.SET_PASSES.values() for k in p["kinds"]}
    assert covered == set(prompts.SET_KINDS)


@pytest.mark.parametrize("focus", list(prompts.SET_PASSES))
def test_focused_prompt_contains_procedure_and_only_its_kinds(focus):
    p = prompts.SET_PASSES[focus]
    text = prompts.build_set_prompt("Документ.", _reqs(), focus=focus)
    assert f"ПРОХОД: {p['title']}" in text
    assert p["steps"].strip().splitlines()[0] in text
    for k in p["kinds"]:
        assert f"«{k}»" in text
    assert prompts.DOC_START in text and prompts.DOC_END in text
    assert "Список требований:" in text and "R-01" in text


def test_prompt_without_focus_is_the_general_one():
    text = prompts.build_set_prompt("Документ.", _reqs())
    assert "ПРОХОД:" not in text


# Слова, номера и обороты проверяемого документа (docs/requirements.md). Ни одного из них не должно
# быть в промптах: агент обязан находить дефекты сам, а не по подсказкам, подогнанным под этот документ.
DOCUMENT_SPECIFIC = (
    "push", "порог", "тиын", "KZT", "₸", "Алмат", "saveLimit", "AlertsController", "cardId", "limit_kzt",
    "Ф-1", "Ф-2", "Ф-3", "Ф-4", "НФТ", "Часть A", "Часть B", "в течение минуты", "активной карт",
    "проведённ", "проведенн", "личного бюджета", "расходы", "карты", "процессинг", "Проверьте сумму",
)


def _all_prompt_texts() -> dict[str, str]:
    texts = {
        "system": prompts.SYSTEM_PROMPT,
        "slicing": prompts.build_slicing_prompt("Документ."),
        "check": prompts.build_check_prompt("Документ.", _reqs()),
        "quote_fix": prompts.build_quote_fix_prompt("Документ.", [(0, "цитата", "контекст")]),
        "merge": prompts.build_merge_prompt([(0, "пробел", "Нет отката.", "Добавить откат.")]),
        "set": prompts.build_set_prompt("Документ.", _reqs()),
    }
    texts.update({f"set:{k}": prompts.build_set_prompt("Документ.", _reqs(), focus=k) for k in prompts.SET_PASSES})
    return texts


@pytest.mark.parametrize("name", list(_all_prompt_texts()))
def test_prompts_have_no_document_specific_hardcode(name):
    """Промпты — общая методика ревью: в них нет слов, номеров и подсказок из проверяемого документа."""
    leaked = [w for w in DOCUMENT_SPECIFIC if w.lower() in _all_prompt_texts()[name].lower()]
    assert leaked == [], f"{name}: {leaked}"


def test_requirement_prompt_asks_to_walk_all_criteria():
    text = prompts.build_check_prompt("Документ.", _reqs())
    assert "ВСЕМ десяти критериям" in text


def test_pipeline_runs_one_set_call_per_pass():
    if not DOC.exists():
        pytest.skip("docs/requirements.md отсутствует")
    from tests import mock_llm
    llm = mock_llm.MockLLM()
    pipeline.run(DOC.read_text(encoding="utf-8"), ask_json=llm)
    assert llm.calls.count("SetResult") == len(prompts.SET_PASSES)
    assert len(llm.calls) <= 30
