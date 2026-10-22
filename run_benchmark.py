"""Собирает фактические ответы работающего HSEHelper по 30 сценариям.

Пример:
python run_benchmark.py --url http://127.0.0.1:8000 --delay 4.5 --output docs/Результаты_фактического_прогона.md

Прогоны последовательные, так как сервер ограничивает частоту запросов. Оценка PASS/FAIL
оставлена человеку: наличие HTTP 200 ещё не доказывает корректность фактического ответа.
"""
from __future__ import annotations
import argparse
import json
import time
from datetime import datetime
from pathlib import Path
import requests

BASE_DIR = Path(__file__).resolve().parent
CASES_PATH = BASE_DIR / "benchmark_cases.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Фактический прогон 30 вопросов HSEHelper")
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="Базовый URL работающего API")
    parser.add_argument("--delay", type=float, default=4.5, help="Пауза между вопросами, чтобы не превышать rate limit")
    parser.add_argument("--timeout", type=float, default=180.0, help="Тайм-аут одного ответа, секунд")
    parser.add_argument("--output", default="docs/Результаты_фактического_прогона.md", help="Путь к отчёту Markdown")
    parser.add_argument("--limit", type=int, default=30, help="Сколько первых кейсов выполнить для пробного прогона")
    args = parser.parse_args()
    if args.delay < 0 or not 1 <= args.limit <= 30:
        parser.error("--delay должен быть >= 0; --limit должен быть от 1 до 30")

    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))[:args.limit]
    base_url = args.url.rstrip("/")
    output = Path(args.output)
    if not output.is_absolute():
        output = BASE_DIR / output
    output.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for ix, case in enumerate(cases, start=1):
        print(f"[{ix}/{len(cases)}] {case['id']}: {case['question']}", flush=True)
        record = {**case, "http_status": None, "actual_answer": "", "actual_sources": [], "technical_error": "", "manual_grade": "НЕ ОЦЕНЕНО"}
        try:
            response = requests.post(
                f"{base_url}/api/ask",
                json={"question": case["question"], "session_id": f"bench-{case['id'].lower()}"},
                timeout=args.timeout,
            )
            record["http_status"] = response.status_code
            try:
                payload = response.json()
            except ValueError:
                payload = {"detail": response.text[:2000]}
            if response.ok:
                record["actual_answer"] = str(payload.get("answer", ""))
                record["actual_sources"] = payload.get("sources", [])
            else:
                record["technical_error"] = str(payload.get("detail", payload))
        except Exception as exc:
            record["technical_error"] = f"{type(exc).__name__}: {exc}"
        rows.append(record)
        if ix < len(cases) and args.delay:
            time.sleep(args.delay)

    ok = sum(1 for r in rows if r["http_status"] == 200)
    out = [
        "# Фактический прогон HSEHelper — 30 тестовых вопросов", "",
        f"- Дата запуска: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"- API: `{base_url}`", f"- Выполнено кейсов: {len(rows)}", f"- HTTP 200: {ok}/{len(rows)}",
        "- Фактическая оценка правильности требует ручной сверки с эталоном; HTTP 200 не равен PASS.", "",
        "## Результаты", ""
    ]
    for r in rows:
        out += [f"### {r['id']} — {r['kind']}", "", f"**Вопрос:** {r['question']}", "", f"**Эталон / ожидаемое поведение:** {r['reference']}", "", f"**Критерий оценки:** {r['criteria']}", "", f"**HTTP:** {r['http_status']}", "", "**Фактический ответ:**", "", (r['actual_answer'] or f"Техническая ошибка: {r['technical_error']}") , "", "**Источники, возвращённые API:**"]
        if r["actual_sources"]:
            for source in r["actual_sources"]:
                out.append(f"- [{source.get('title','Источник')}]({source.get('url','')})")
        else:
            out.append("- Источники не возвращены")
        out += ["", "**Ручная оценка:** `PASS / FAIL / PARTIAL` — заполнить после сверки.", "", f"**Официальный эталонный источник:** {r['source']}", "", "---", ""]
    output.write_text("\n".join(out), encoding="utf-8")
    print(f"Отчёт сохранён: {output}")
    print(f"HTTP 200: {ok}/{len(rows)}; проверь ответы и вручную выставь PASS/FAIL.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
