"""Мок llm.ask_json для тестов конвейера без реальной модели.

Специально возвращает «плохие» данные, чтобы проверить защиту конвейера:
выдуманные цитаты, цитаты без разметки/с другими тире, дубли, пропущенные проверки,
чужой req_id, упавший вызов.
"""
from __future__ import annotations

import re

from pydantic import BaseModel

from validator import prompts
from validator.pipeline import (
    BatchChecks, MergeGroups, QuoteFix, QuoteFixes, RawRequirement, SetResult, SlicingResult,
)
from validator.schemas import Finding, RequirementCheck, SetFinding

SEC_F = "Часть A / Функциональные требования"


def _document(prompt: str) -> str:
    a = prompt.index(prompts.DOC_START) + len(prompts.DOC_START)
    return prompt[a:prompt.index(prompts.DOC_END)].strip()


class MockLLM:
    def __init__(self, fail_first_big_batch: bool = True) -> None:
        self.calls: list[str] = []
        self._fail_pending = fail_first_big_batch

    def __call__(self, prompt: str, schema: type[BaseModel], system: str | None = None) -> BaseModel:
        assert system, "pipeline обязан передавать system-промпт"
        self.calls.append(schema.__name__)
        return getattr(self, "_" + schema.__name__)(prompt)

    # --- нарезка ---
    def _SlicingResult(self, prompt: str) -> SlicingResult:
        doc = _document(prompt)
        items = []
        for line in doc.splitlines():
            m = re.match(r"^\* \*\*(Ф-\d)\.\*\* (.*)$", line)
            if m:  # цитата без markdown-разметки
                items.append(RawRequirement(doc_id=m.group(1), section=SEC_F,
                                            quote=f"{m.group(1)}. {m.group(2)}", short=f"Пункт {m.group(1)}"))
        items += [
            # другое тире вместо «—»
            RawRequirement(doc_id=None, section="Часть B / Ответ",
                           quote="Если результат 200 или 201 – успешный. Если результат 400, 401 и далее – неуспешный.",
                           short="Правило успешного ответа"),
            # строка таблицы без `|` и обратных кавычек
            RawRequirement(doc_id=None, section="Часть B / Query-параметры",
                           quote="cardId string 1–64 Да Карта клиента Идентификатор карты card_id",
                           short="Параметр cardId"),
            # дубль уже выданного пункта
            RawRequirement(doc_id="Ф-1", section=SEC_F,
                           quote="Ф-1. Держатель активной карты KZT задаёт целый порог от 1 000 до 1 000 000 ₸ включительно.",
                           short="Дубль Ф-1"),
            # многоточие: чинится повторным запросом модели
            RawRequirement(doc_id="нет", section="Часть A / Нефункциональные требования",
                           quote="Сервис передаёт push … после получения события.",
                           short="НФТ: push за минуту"),
            # выдумка: должна быть отброшена
            RawRequirement(doc_id=None, section=SEC_F, quote="Система должна работать быстро и надёжно.",
                           short="Выдуманное требование"),
        ]
        return SlicingResult(requirements=list(reversed(items)))  # порядок документа восстанавливает код

    # --- исправление цитат ---
    def _QuoteFixes(self, prompt: str) -> QuoteFixes:
        fixes = []
        for m in re.finditer(r'^\[(\d+)\] цитата: "(.*)" \| о чём замечание: (.*)$', prompt, re.M):
            idx, ctx = int(m.group(1)), m.group(3)
            if ctx == "НФТ: push за минуту":
                fixes.append(QuoteFix(index=idx,
                                      quote="Сервис передаёт push на отправку в течение минуты после получения события."))
            else:
                fixes.append(QuoteFix(index=idx, quote=""))  # нет подходящего фрагмента
        return QuoteFixes(fixes=fixes)

    # --- склейка повторов среди замечаний к набору ---
    def _MergeGroups(self, prompt: str) -> MergeGroups:
        """Повторы — замечания с одинаковым текстом проблемы; плюс мусорные номера, которые код обязан отбросить."""
        entries = re.findall(r"^\[(\d+)\] \(([^)]*)\) (.*) \| исправление: .*$", prompt, re.M)
        by_text: dict[str, list[int]] = {}
        for idx, _, problem in entries:
            by_text.setdefault(problem, []).append(int(idx))
        return MergeGroups(groups=[g for g in by_text.values() if len(g) > 1] + [[999, -1], [0]])

    # --- проверка пачки ---
    def _BatchChecks(self, prompt: str) -> BatchChecks:
        reqs = re.findall(r'^- id: (R-\d+) \|.*цитата: "(.*)"$', prompt, re.M)
        if self._fail_pending and len(reqs) == 4:
            self._fail_pending = False
            raise RuntimeError("имитация сбоя API (429)")
        checks = []
        for n, (rid, quote) in enumerate(reqs):
            if n == 0:    # настоящая находка + выдуманная цитата
                checks.append(RequirementCheck(req_id=rid, verdict="violations", findings=[
                    Finding(req_id=rid, criterion="Однозначность", severity="minor", quote=quote[:25],
                            problem="Мелкое замечание.", fix="Поправить."),
                    Finding(req_id=rid, criterion="проверяемость", severity="critical", quote=quote[:40],
                            problem="Нет критерия проверки.", fix="Добавить критерий."),
                    Finding(req_id=rid, criterion="полнота", severity="major",
                            quote="Такой фразы в документе нет.", problem="Выдумка.", fix="-"),
                ]))
            elif n == 1:  # чистое требование, но модель пишет «violations» без находок
                checks.append(RequirementCheck(req_id=rid, verdict="violations", findings=[]))
            elif n == 2:  # пропущено моделью — pipeline должен переспросить
                continue
            else:         # чужой req_id внутри находки
                checks.append(RequirementCheck(req_id=rid, verdict="violations", findings=[
                    Finding(req_id="R-99", criterion="полнота", severity="major", quote=quote[:30],
                            problem="Не указаны исключения.", fix="Описать исключения."),
                ]))
        return BatchChecks(checks=checks)

    # --- проверка набора ---
    def _SetResult(self, prompt: str) -> SetResult:
        return SetResult(findings=[
            SetFinding(kind="Шаблон", severity="minor", related_req_ids=[],
                       quote="Документ (версия/дата) в источнике не заполнен.",
                       problem="Версия документа не заполнена.", fix="Указать версию и дату."),
            SetFinding(kind="противоречие", severity="critical", related_req_ids=["R-03", "R-99", "R-03"],
                       quote="На карту приходится максимум один такой push за календарный месяц. || "
                             "следующее достижение вызывает push независимо от истории предыдущей настройки.",
                       problem="Ф-3 и Ф-4 противоречат друг другу.", fix="Согласовать правила."),
            SetFinding(kind="пробел", severity="major", related_req_ids=[],
                       quote="Это придуманная цитата.", problem="Выдумка.", fix="-"),
        ])
