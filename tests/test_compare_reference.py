from pathlib import Path

from tests.compare_reference import compare, overlaps, render
from tests.fixtures import make_meta, make_result
from tests.helpers import parse_report_findings
from validator.report import build

REF = Path(__file__).resolve().parent / "reference_defects.md"


def test_parse_report_findings_refs_and_quotes():
    text = build(make_meta(), make_result())
    findings = parse_report_findings(text)
    refs = [f["ref"] for f in findings]
    assert "R-03.1" in refs and "S-01" in refs
    s01 = next(f for f in findings if f["ref"] == "S-01")
    assert len(s01["quotes"]) == 2  # две части противоречия — две цитаты
    assert s01["problem"]


def test_compare_marks_found_missed_and_extra():
    text = build(make_meta(), make_result())
    rows, extra = compare(text, REF)
    by_id = {r["id"]: r for r in rows}
    assert by_id["D-01"]["found"] and "S-01" in by_id["D-01"]["where"]  # Ф-3 || Ф-4
    assert not by_id["D-04"]["found"] and by_id["D-04"]["where"] == "—"  # GET, в фикстуре нет
    assert any("Колонка A" in " ".join(f["quotes"]) for f in extra)      # цитата вне эталона
    out = render(rows, extra, [])
    assert "ПРОПУЩЕН" in out and "Лишнее" in out and "Найдено:" in out


# ---------------------------------------------------------------- нестрогое сравнение цитат

def test_overlap_containment_and_markup_stripped():
    # агент оставил разметку исходного markdown, эталон — без неё
    assert overlaps("Ф-3.** Когда расходы переходят от суммы ниже порога", "Когда расходы переходят от суммы")
    assert overlaps("`cardId` string", "cardId string")


def test_overlap_partial_quote_counts():
    # агент процитировал начало, эталон — продолжение той же фразы; общий фрагмент длинный
    assert overlaps("Ф-1.** Держатель активной", "Держатель активной карты KZT")


def test_overlap_rejects_common_phrase_of_different_statuses():
    # у кодов 400 и 401 одинаковое начало фразы: это разные места документа
    a = "Ответ в случае, если параметры запроса заполнены неверно."
    b = "Ответ в случае, если запрос выполнен без необходимой авторизации."
    assert not overlaps(a, b)


def test_overlap_rejects_unrelated_and_too_short():
    assert not overlaps("Совсем другая фраза про отчёт", "Держатель активной карты KZT")
    assert not overlaps("push", "push за месяц")  # короче MIN_OVERLAP
