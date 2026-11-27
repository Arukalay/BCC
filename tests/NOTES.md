# Тесты и сверка с эталоном

Запуск: `pytest` из корня (Python 3.11+, `pip install -r requirements.txt`). Сеть и ключи не нужны: модель
подменяется моком `tests/mock_llm.py` или поддельным SDK.

| Файл | Что проверяет |
|---|---|
| `test_core.py`, `test_loader.py` | чтение md/txt/pdf/docx, ошибки формата, клиент Gemini, CLI |
| `test_llm_resilience.py`, `test_openai_client.py` | квоты и повторы, быстрый отказ, таймаут, клиент OpenAI, строгие схемы |
| `test_prompts.py` | проходы проверки набора; в промптах нет слов и подсказок из проверяемого документа |
| `test_pipeline_smoke.py`, `test_pipeline_robustness.py`, `test_dedup.py` | нарезка, цитаты из исходного текста, деление длинного документа, склейка повторов, устойчивость к плохим ответам модели |
| `test_pipeline_report.py`, `test_cli_e2e.py` | весь конвейер на моке → отчёт; `python -m validator` на md и PDF |
| `test_report.py`, `test_quotes.py`, `test_schemas.py` | шесть разделов отчёта, числа сводки, экранирование, цитаты, схемы |
| `test_reference_defects.py`, `test_compare_reference.py` | формат эталона и его цитаты, сверка отчёта с эталоном |

## Эталон и сверка
- `reference_defects.md` — 58 дефектов документа с дословными цитатами, колонка «Тип» (`внутри требования` / `набор`).
  Номера D-NN стабильны: новые только в конец. Несколько цитат через ` // `; в цитатах нельзя ` | ` (разделитель столбцов).
- `python tests/compare_reference.py docs/requirements.md output/report.md` — «найден» означает, что цитата находки
  пересекается с цитатой дефекта (одна содержит другую или общий фрагмент ≥ 15 символов и ≥ половины короткой).
  Это верхняя оценка: смысл сверяется вручную, итоги прогонов — в `evaluation.md`.
- `data/requirements_sample.pdf` собран из текста `docs/requirements.md` (не оригинал организаторов).
