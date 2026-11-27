"""Склейка повторов между проходами проверки набора (без LLM). Тексты взяты из реального прогона."""
from validator.pipeline import _drop_cross_pass_duplicates, _quotes_overlap, _same_problem
from validator.schemas import SetFinding


def _s(kind, severity, quote, problem):
    return SetFinding(kind=kind, severity=severity, related_req_ids=[], quote=quote, problem=problem, fix="f")


PROD_A = _s("пробел", "major", "Request URL** | Dev: `/alerts/saveLimit` . Prod: -",
            "Отсутствует URL для продуктовой среды, что делает невозможным планирование развертывания функционала.")
PROD_B = _s("шаблон", "minor", "Prod: -",
            "Отсутствует URL для продуктовой среды в описании API, что делает невозможным планирование интеграции в реальной среде.")
SIGN_A = _s("шаблон", "minor", "Блок согласующих",
            "Раздел согласующих пуст, что не позволяет идентифицировать ответственных за принятие требований.")
SIGN_B = _s("пробел", "major", "Блок согласующих",
            "Таблица согласующих пуста. Отсутствуют подписи ответственных лиц, что означает отсутствие формального согласования требований.")
SIGN_C = _s("шаблон", "minor", "Блок согласующих | Подразделение | ФИО ответственного | Подпись",
            "Блок согласующих пуст, что нарушает требования к формализации документации в банке.")
TZ_A = _s("пробел", "major", "Месяц определяется по дате совершения покупки в Алматы.",
          "Не определено, как обрабатывать покупки, совершенные в часовых поясах, отличных от Алматы, или как система должна реагировать на переход через полночь в конце месяца.")
TZ_B = _s("пробел", "major", "Месяц определяется по дате совершения покупки в Алматы.",
          "Не определено поведение системы при переходе на летнее/зимнее время или при изменении часового пояса в Алматы, а также не указан формат даты/времени.")
KEEP_A = _s("пробел", "major", "Хранить идентификатор карты, месяц, идентификатор операции",
            "Отсутствует требование к сроку хранения данных о расходах и настройках. Это создает риск переполнения хранилища.")
KEEP_B = _s("пробел", "major", "Хранить идентификатор карты, месяц, идентификатор операции",
            "Не указан часовой пояс для даты/месяца и не описан механизм обработки повторных событий (идемпотентность).")


def test_same_place_same_problem_from_different_passes_is_merged():
    assert _drop_cross_pass_duplicates([PROD_A, PROD_B]) == [PROD_A]


def test_chain_of_duplicates_collapses_to_first():
    # SIGN_A≈SIGN_B и SIGN_B≈SIGN_C, но SIGN_A и SIGN_C сами похожи слабее: сравнение и с отброшенными
    assert _drop_cross_pass_duplicates([SIGN_A, SIGN_B, SIGN_C]) == [SIGN_A]


def test_different_problems_on_one_quote_are_kept():
    assert _drop_cross_pass_duplicates([TZ_A, TZ_B]) == [TZ_A, TZ_B]
    assert _drop_cross_pass_duplicates([KEEP_A, KEEP_B]) == [KEEP_A, KEEP_B]


def test_same_problem_far_apart_in_document_is_kept():
    a = _s("пробел", "major", "Фрагмент один", PROD_A.problem)
    b = _s("пробел", "major", "Совсем другое место", PROD_A.problem)
    assert _drop_cross_pass_duplicates([a, b]) == [a, b]


def test_most_severe_survives_because_input_is_sorted_by_importance():
    critical = _s("пробел", "critical", "Prod: -", PROD_B.problem)
    assert _drop_cross_pass_duplicates([critical, PROD_A]) == [critical]


def test_quotes_overlap_handles_two_part_quotes():
    assert _quotes_overlap("Правило один || Правило два", "Правило два")
    assert not _quotes_overlap("Правило один || Правило два", "Правило три")
    assert not _quotes_overlap("", "Правило")


def test_same_problem_threshold_is_adjustable():
    a, b = SIGN_A.problem, SIGN_B.problem
    assert not _same_problem(a, b)                 # строгий порог внутри одного вида
    assert _same_problem(a, b, threshold=0.3)      # мягкий порог между проходами


# ---------------------------------------------------------------- склейка повторов по смыслу (вызов модели)

def _items():
    return [_s("противоречие", "critical", f"Цитата {i}", f"Проблема {i}").model_copy(update={"related_req_ids": [f"R-0{i}"]})
            for i in range(5)]


def test_merge_by_model_groups_keeps_most_important_and_unions_requirements():
    from validator.pipeline import MergeGroups, _merge_same_defects
    items = _items()
    def ask(prompt, schema):
        return MergeGroups(groups=[[3, 1], [1, 4], [999, -2], [2]])   # цепочка 1-3-4 и мусорные номера

    merged = _merge_same_defects(items, ask)
    assert [m.problem for m in merged] == ["Проблема 0", "Проблема 1", "Проблема 2"]
    assert merged[1].related_req_ids == ["R-01", "R-03", "R-04"]                      # номера склеенных не теряются


def test_merge_prompt_lists_every_finding_without_document():
    from validator import prompts
    text = prompts.build_merge_prompt([(0, "пробел", "Нет отката.", "Добавить откат."), (1, "шаблон", "Пустой блок.", "Заполнить.")])
    assert "[0] (пробел) Нет отката. | исправление: Добавить откат." in text and "[1] (шаблон) Пустой блок." in text
    assert "исчезнет ли" in text                                # критерий повтора: исправление одного устраняет другое
    assert prompts.DOC_START not in text


def test_merge_failure_keeps_findings_and_fatal_error_stops():
    import pytest
    from validator.llm import LLMError, LLMFatalError
    from validator.pipeline import _merge_same_defects
    items = _items()

    def broken(prompt, schema):
        raise LLMError("невалидный JSON")

    def fatal(prompt, schema):
        raise LLMFatalError("квота")

    assert _merge_same_defects(items, broken) == items
    with pytest.raises(LLMFatalError):
        _merge_same_defects(items, fatal)
    assert _merge_same_defects(items[:1], fatal) == items[:1]                         # одно замечание — вызова нет
