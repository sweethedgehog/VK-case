"""Локальный семантический поиск по страницам дисциплин с кэшированием embeddings."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from query_utils import expand_search_query, mentioned_course_aliases, normalize_text, is_teacher_question

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
COURSES_FILE = DATA_DIR / "courses.json"
CHUNKS_FILE = DATA_DIR / "chunks.json"
EMBEDDINGS_FILE = DATA_DIR / "embeddings.npy"
INDEX_META_FILE = DATA_DIR / "index_meta.json"
DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHUNK_CHARS = 1600
CHUNK_OVERLAP = 220
INDEX_VERSION = "hybrid-title-prefix-v2"


def load_courses(path: Path = COURSES_FILE) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Не найден {path}. Сначала запусти `python scrape_hse.py --limit 30`."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("courses.json должен содержать JSON-массив.")
    return [c for c in data if isinstance(c, dict) and c.get("title") and c.get("url")]


def course_source_text(course: dict[str, Any]) -> str:
    parts = [f"Дисциплина: {course.get('title', '')}"]
    for key in ("year", "credits", "language"):
        value = course.get(key)
        if value:
            parts.append(f"{key}: {value}")
    if course.get("sections"):
        for heading, values in course["sections"].items():
            if isinstance(values, list):
                parts.append(f"{heading}: " + "\n".join(str(v) for v in values))
            else:
                parts.append(f"{heading}: {values}")
    full_text = str(course.get("full_text", "")).strip()
    if full_text:
        parts.append(full_text)
    return "\n".join(parts)


def split_into_chunks(text: str, chunk_chars: int = CHUNK_CHARS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    text = re.sub(r"[ \t]+", " ", text.strip())
    if not text:
        return []
    # Prefer paragraph or sentence boundaries; then fall back to a hard character window.
    paragraphs = [p.strip() for p in re.split(r"\n+", text) if p.strip()]
    packed: list[str] = []
    current = ""

    for paragraph in paragraphs:
        remaining = paragraph
        while len(remaining) > chunk_chars:
            piece = remaining[:chunk_chars]
            cut = max(piece.rfind(". "), piece.rfind("; "), piece.rfind(" "))
            if cut > int(chunk_chars * 0.55):
                piece = piece[:cut + 1]
            if current:
                packed.append(current)
                current = ""
            packed.append(piece.strip())
            remaining = remaining[max(1, len(piece) - overlap):].strip()
        if not remaining:
            continue
        candidate = f"{current}\n{remaining}".strip() if current else remaining
        if len(candidate) <= chunk_chars:
            current = candidate
        else:
            if current:
                packed.append(current)
            current = remaining
    if current:
        packed.append(current)
    return packed


def source_fingerprint(courses: list[dict[str, Any]], model_name: str) -> str:
    serialized = json.dumps(
        [{"title": c.get("title"), "url": c.get("url"), "full_text": c.get("full_text"),
          "sections": c.get("sections"), "credits": c.get("credits"), "year": c.get("year")} for c in courses],
        ensure_ascii=False, sort_keys=True,
    )
    return hashlib.sha256((INDEX_VERSION + "\n" + model_name + "\n" + serialized).encode("utf-8")).hexdigest()


class SemanticIndex:
    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self.model: Any | None = None
        self.courses = load_courses()
        self.chunks: list[dict[str, Any]] = []
        self.embeddings: np.ndarray | None = None
        self._ensure_index()

    def _load_model(self):
        if self.model is None:
            print(f"Загружаю локальную embedding-модель: {self.model_name}")
            print("При первом запуске модель будет скачана из Hugging Face; это не расходует API-квоту.")
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "Не установлен sentence-transformers. Выполни `pip install -r requirements.txt`."
                ) from exc
            self.model = SentenceTransformer(self.model_name)
        return self.model

    def _build_chunks(self) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for idx, course in enumerate(self.courses):
            text = course_source_text(course)
            for chunk_no, chunk in enumerate(split_into_chunks(text)):
                result.append({
                    "course_index": idx,
                    "title": str(course["title"]),
                    "url": str(course["url"]),
                    "chunk_no": chunk_no,
                    "text": chunk,
                })
        return result

    def _ensure_index(self) -> None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        fingerprint = source_fingerprint(self.courses, self.model_name)
        if CHUNKS_FILE.exists() and EMBEDDINGS_FILE.exists() and INDEX_META_FILE.exists():
            try:
                meta = json.loads(INDEX_META_FILE.read_text(encoding="utf-8"))
                if meta.get("fingerprint") == fingerprint and meta.get("model") == self.model_name:
                    chunks = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
                    embeddings = np.load(EMBEDDINGS_FILE, allow_pickle=False)
                    if len(chunks) == len(embeddings) and len(chunks) > 0:
                        self.chunks = chunks
                        self.embeddings = embeddings.astype(np.float32, copy=False)
                        print(f"Семантический индекс загружен из кэша: {len(self.chunks)} фрагментов.")
                        return
            except (OSError, ValueError, json.JSONDecodeError):
                pass

        self.chunks = self._build_chunks()
        if not self.chunks:
            raise ValueError("В курсах нет текста для индексации. Проверь courses.json или повтори scrape.")
        model = self._load_model()
        texts = [f"Дисциплина: {chunk['title']}\n{chunk['text']}" for chunk in self.chunks]
        self.embeddings = model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)
        CHUNKS_FILE.write_text(json.dumps(self.chunks, ensure_ascii=False, indent=2), encoding="utf-8")
        np.save(EMBEDDINGS_FILE, self.embeddings)
        INDEX_META_FILE.write_text(json.dumps({
            "fingerprint": fingerprint,
            "model": self.model_name,
            "chunks": len(self.chunks),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Индекс построен и сохранён: {len(self.chunks)} фрагментов.")

    def search(self, query: str, top_k: int = 7, max_chunks_per_course: int = 2) -> list[dict[str, Any]]:
        if not query.strip():
            return []
        model = self._load_model()
        expanded_query = expand_search_query(query)
        query_vector = model.encode(
            [expanded_query], convert_to_numpy=True, normalize_embeddings=True
        )[0].astype(np.float32)
        assert self.embeddings is not None
        scores = self.embeddings @ query_vector

        # Hybrid reranking: embeddings help with meaning; lexical signals keep exact
        # course names and labeled fields (e.g. "Преподаватели") from being missed.
        normalized_query = normalize_text(expanded_query)
        words = {
            word for word in normalized_query.split()
            if len(word) >= 4 and word not in {
                "какие", "какой", "какая", "какое", "есть", "этот", "эта", "это",
                "когда", "сколько", "мне", "для", "про", "при", "который", "которая",
            }
        }
        aliases = mentioned_course_aliases(query)
        teacher_intent = is_teacher_question(query)
        teacher_terms = ("преподавател", "кто читает", "ведущий преподавател")

        for i, chunk in enumerate(self.chunks):
            title = normalize_text(str(chunk.get("title", "")))
            body = normalize_text(str(chunk.get("text", "")))
            combined = f"{title} {body}"
            if title and title in normalize_text(query):
                scores[i] += 0.18
            if any(alias in title or title in alias for alias in aliases):
                scores[i] += 0.20
            title_tokens = set(title.split())
            if words and title_tokens.intersection(words):
                scores[i] += min(0.10, 0.035 * len(title_tokens.intersection(words)))
            matched = sum(1 for word in words if word in combined)
            if words:
                scores[i] += min(0.12, 0.12 * matched / len(words))
            if teacher_intent and any(term in body for term in teacher_terms):
                scores[i] += 0.22
            # A chunk for the requested subject should beat a similarly worded
            # fragment from another subject when the alias identifies its title.
            if aliases and any(alias in title or title in alias for alias in aliases):
                if teacher_intent and any(term in body for term in teacher_terms):
                    scores[i] += 0.12

        order = np.argsort(scores)[::-1]
        results: list[dict[str, Any]] = []
        per_course: dict[int, int] = {}
        for raw_i in order:
            i = int(raw_i)
            chunk = self.chunks[i]
            ci = int(chunk["course_index"])
            if per_course.get(ci, 0) >= max_chunks_per_course:
                continue
            hit = dict(chunk)
            hit["score"] = float(scores[i])
            results.append(hit)
            per_course[ci] = per_course.get(ci, 0) + 1
            if len(results) >= top_k:
                break
        return results

