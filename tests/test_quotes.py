from tests.fixtures import DOC_TEXT, make_meta, make_result
from tests.helpers import extract_report_quotes, find_bad_quotes, quote_in_text
from validator.report import build


def test_accepts_exact_quote():
    assert quote_in_text("Сервис передаёт push на отправку в течение минуты", DOC_TEXT)


def test_accepts_quote_with_whitespace_differences():
    # в тексте перенос строки, в цитате пробел; и неразрывный пробел
    assert quote_in_text("Хранить идентификатор карты, месяц, идентификатор операции", DOC_TEXT)
    assert quote_in_text("от 1\u00a0000 до 1 000 000 ₸", DOC_TEXT)


def test_rejects_invented_quote():
    assert not quote_in_text("Система должна отвечать за 5 секунд", DOC_TEXT)
    assert not quote_in_text("", DOC_TEXT)


def test_two_part_quote_requires_both_parts():
    assert quote_in_text("На карту приходится максимум один такой push || Если она ниже нового порога", DOC_TEXT)
    assert not quote_in_text("На карту приходится максимум один такой push || выдумка", DOC_TEXT)


def test_report_quotes_all_found_in_document():
    text = build(make_meta(), make_result())
    quotes = extract_report_quotes(text)
    assert len(quotes) >= 10
    assert find_bad_quotes(text, DOC_TEXT) == []


def test_table_row_quote_with_backticks_and_escaped_pipes_roundtrips():
    """Реальная строка таблицы нарезки из прогона 4: цитата с `|` и обратными кавычками, двойной забор."""
    row = ("| R-38 | — | Часть B / Параметры запроса | Параметр cardId | "
           "`` `cardId` \\| string \\| 1-64 \\| Да \\| Карта клиента \\| Идентификатор карты \\| `card_id` `` |")
    doc = "| `cardId` | string | 1–64 | Да | Карта клиента | Идентификатор карты | `card_id` |"
    quotes = extract_report_quotes(row)
    assert quotes == ["`cardId` | string | 1-64 | Да | Карта клиента | Идентификатор карты | `card_id`"]
    assert find_bad_quotes(row, doc) == []
    assert extract_report_quotes("| R-01 | Ф-1 | раздел | кратко | — |") == []   # номер в документе: цитаты в таблице нет


def test_bad_quote_in_report_is_caught():
    result = make_result()
    result.checks[1].findings[0].quote = "Этой фразы нет в документе"
    text = build(make_meta(), result)
    assert find_bad_quotes(text, DOC_TEXT) == ["Этой фразы нет в документе"]
