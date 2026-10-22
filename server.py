"""FastAPI app serving the web UI and shared API for web, Telegram and mobile clients."""
from __future__ import annotations

import hmac
import os
import threading
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from conversation_store import initialize_store, load_history, save_history
from semantic_search import load_courses

BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"
load_dotenv(BASE_DIR / ".env")

API_INTERNAL_TOKEN = os.getenv("API_INTERNAL_TOKEN", "").strip()
REQUESTS_PER_MINUTE = max(2, int(os.getenv("API_REQUESTS_PER_MINUTE", "15")))
MAX_QUESTION_CHARS = max(200, int(os.getenv("MAX_QUESTION_CHARS", "1500")))

app = FastAPI(
    title="HSE Course Assistant API",
    description="API помощника по открытым программам дисциплин ИВТ МИЭМ НИУ ВШЭ.",
    version="1.0.0",
)
app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

_assistant: Any | None = None
_assistant_lock = threading.RLock()
_rate_lock = threading.Lock()
_rate_buckets: dict[str, deque[float]] = defaultdict(deque)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    session_id: str = Field(default="web-default", min_length=1, max_length=80)


class Source(BaseModel):
    title: str
    url: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    session_id: str


def _get_assistant():
    global _assistant
    with _assistant_lock:
        if _assistant is None:
            from assistant import HSECourseAssistant
            _assistant = HSECourseAssistant()
        return _assistant


def _trusted_bot(x_internal_token: str | None) -> bool:
    return bool(
        API_INTERNAL_TOKEN
        and x_internal_token
        and hmac.compare_digest(x_internal_token, API_INTERNAL_TOKEN)
    )


def _apply_rate_limit(bucket_key: str) -> None:
    now = time.monotonic()
    with _rate_lock:
        bucket = _rate_buckets[bucket_key]
        while bucket and now - bucket[0] >= 60:
            bucket.popleft()
        if len(bucket) >= REQUESTS_PER_MINUTE:
            raise HTTPException(
                status_code=429,
                detail=f"Слишком много запросов. Лимит: {REQUESTS_PER_MINUTE} за 60 секунд. Попробуй позже.",
            )
        bucket.append(now)
        # Avoid unbounded memory use if many one-off clients connect.
        if len(_rate_buckets) > 5000:
            expired = [key for key, values in _rate_buckets.items() if not values or now - values[-1] > 3600]
            for key in expired[:2000]:
                _rate_buckets.pop(key, None)


def _client_identity(request: Request) -> str:
    # Do not trust X-Forwarded-For unless a known reverse proxy has been configured.
    return request.client.host if request.client else "unknown"


@app.on_event("startup")
def _startup() -> None:
    initialize_store()


@app.get("/", include_in_schema=False)
def homepage():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "hse-course-assistant"}


@app.get("/api/status")
def status() -> dict[str, Any]:
    try:
        courses = load_courses()
        course_count = len(courses)
        data_ready = course_count > 0
    except (FileNotFoundError, ValueError):
        course_count = 0
        data_ready = False
    return {
        "status": "ok",
        "data_ready": data_ready,
        "courses_count": course_count,
        "model_initialized": _assistant is not None,
        "model": os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"),
    }


@app.get("/api/courses")
def courses() -> dict[str, Any]:
    try:
        items = load_courses()
    except (FileNotFoundError, ValueError):
        return {"count": 0, "courses": [], "message": "Сначала запусти scrape_hse.py для сбора каталога."}
    public_items = []
    for item in items:
        public_items.append({
            "title": str(item.get("title", "Без названия")),
            "url": str(item.get("url", "")),
            "year": str(item.get("year", "2026/2027")),
            "credits": item.get("credits"),
            "language": item.get("language"),
            "description": str(item.get("description") or "")[:500],
        })
    return {"count": len(public_items), "courses": public_items}


@app.post("/api/ask", response_model=AskResponse)
def ask(
    body: AskRequest,
    request: Request,
    x_internal_token: str | None = Header(default=None, alias="X-Internal-Token"),
) -> AskResponse:
    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Вопрос не должен быть пустым.")

    try:
        available_courses = load_courses()
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(
            status_code=503,
            detail="Локальная база дисциплин не найдена. Сначала выполни сбор данных командой scrape_hse.py.",
        ) from exc
    if not available_courses:
        raise HTTPException(
            status_code=503,
            detail="Каталог пуст. Сначала выполни сбор данных командой scrape_hse.py.",
        )

    is_internal = _trusted_bot(x_internal_token)
    identity = f"session:{body.session_id}" if is_internal else f"ip:{_client_identity(request)}"
    _apply_rate_limit(identity)

    # Avoid collisions and keep session IDs from containing unusual control characters.
    session_id = "".join(ch for ch in body.session_id if ch.isalnum() or ch in "-_:.")[:80]
    if not session_id:
        raise HTTPException(status_code=422, detail="Некорректный session_id.")

    try:
        history = load_history(session_id)
        # The model/index is shared; serialize inference so session histories can't cross-contaminate.
        with _assistant_lock:
            assistant = _get_assistant()
            answer_text, source_items = assistant.answer(question, history=history)
        save_history(session_id, history)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail="Локальная база дисциплин не найдена. Сначала выполни сбор данных командой scrape_hse.py.",
        ) from exc
    except Exception as exc:
        # The API log contains details; the client gets a useful but non-secret error.
        print(f"[api] ask failed: {type(exc).__name__}: {exc}", flush=True)
        raise HTTPException(
            status_code=503,
            detail="Не удалось подготовить ответ. Проверь, что база собрана, зависимости установлены и GEMINI_API_KEY настроен.",
        ) from exc

    sources: list[Source] = []
    seen: set[str] = set()
    for source in source_items:
        url = str(source.get("url", ""))
        title = str(source.get("title", "Источник"))
        if url.startswith("https://www.hse.ru/") and url not in seen:
            seen.add(url)
            sources.append(Source(title=title, url=url))
    return AskResponse(answer=answer_text, sources=sources, session_id=session_id)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=False)
