"""Query normalization, Russian course-name aliases and direct metadata extraction."""
from __future__ import annotations

import re
from typing import Any


def normalize_text(value: str) -> str:
    value = (value or "").casefold().replace("ё", "е")
    value = re.sub(r"[^0-9a-zа-я]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


# Alias -> canonical course phrase. Keep this small and explicit: an alias should not
# silently map to an unrelated subject.
COURSE_ALIASES: dict[str, str] = {
    "матанализ": "математический анализ",
    "матан": "математический анализ",
    "мат анализ": "математический анализ",
    "математический анализ": "математический анализ",
    "линалгебра": "линейная алгебра",
    "линал": "линейная алгебра",
    "линейная алгебра": "линейная алгебра",
    "дискретка": "дискретная математика",
    "дискра": "дискретная математика",
    "дискретная математика": "дискретная математика",
    "тервер": "теория вероятностей",
    "теорвер": "теория вероятностей",
    "теория вероятностей": "теория вероятностей",
    "матстат": "математическая статистика",
    "математическая статистика": "математическая статистика",
    "вышмат": "высшая математика",
    "высшая математика": "высшая математика",
}

_ALIAS_ITEMS = sorted(
    ((normalize_text(alias), canonical) for alias, canonical in COURSE_ALIASES.items()),
    key=lambda item: len(item[0]), reverse=True,
)

TEACHER_INTENT_RE = re.compile(
    r"\b(кто вед[её]т|кто читает|кто препода[её]т|преподавател\w*|препод\b|"
    r"кто проводит|ведущие преподаватели|состав преподавател)\b",
    re.IGNORECASE,
)
GROUP_ASSIGNMENT_RE = re.compile(
    r"(у какой группы|какой группе|какие группы|по группам|"
    r"кто у .* группы|кто у какой групп|распределени[ея].{0,40}групп|"
    r"какую группу вед[её]т|для какой группы|преподавател.{0,40}групп)",
    re.IGNORECASE,
)
FOLLOWUP_RE = re.compile(
    r"\b(а кто|а что|а как|а у него|а у нее|а у них|этот предмет|этого курса|"
    r"его преподавател|ее преподавател|у него|у нее|у них)\b",
    re.IGNORECASE,
)


def mentioned_course_aliases(query: str) -> list[str]:
    normalized = normalize_text(query)
    results: list[str] = []
    for alias, canonical in _ALIAS_ITEMS:
        if re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", normalized):
            if canonical not in results:
                results.append(canonical)
    return results


def expand_search_query(query: str) -> str:
    """Add canonical course names and likely section vocabulary to a natural-language query."""
    normalized = normalize_text(query)
    additions = mentioned_course_aliases(query)
    if TEACHER_INTENT_RE.search(query):
        additions.extend([
            "преподаватели преподаватель преподавательский состав кто ведет курс кто читает дисциплину",
        ])
    if re.search(r"\b(экзамен|зачет|зачёт|контроль|оценивают|оценка|аттестация)\b", normalized):
        additions.append("элементы контроля промежуточная аттестация форма оценивания")
    if re.search(r"\b(изучают|изучается|содержание|темы курса|о чем курс|про что курс)\b", normalized):
        additions.append("аннотация содержание дисциплины планируемые результаты обучения")
    return (query.strip() + " " + " ".join(additions)).strip()


def is_teacher_question(query: str) -> bool:
    return bool(TEACHER_INTENT_RE.search(query))


def is_group_assignment_question(query: str) -> bool:
    """Detect questions needing a teacher-to-academic-group schedule mapping."""
    normalized = normalize_text(query)
    return bool(GROUP_ASSIGNMENT_RE.search(query)) or (
        "групп" in normalized
        and bool(re.search(r"\b(кто|преподавател|ведет|ведёт|читает|распределени)\w*\b", normalized))
    )


def is_followup_question(query: str) -> bool:
    normalized = normalize_text(query)
    return len(normalized.split()) <= 8 and bool(FOLLOWUP_RE.search(query))


def _course_full_text(course: dict[str, Any]) -> str:
    parts = [str(course.get("full_text", ""))]
    sections = course.get("sections", {})
    if isinstance(sections, dict):
        for heading, values in sections.items():
            parts.append(str(heading))
            if isinstance(values, list):
                parts.extend(str(v) for v in values)
            else:
                parts.append(str(values))
    return "\n".join(parts)


def _course_name_match_score(query: str, course: dict[str, Any]) -> int:
    query_n = normalize_text(query)
    title_n = normalize_text(str(course.get("title", "")))
    if not title_n:
        return 0
    score = 0
    if re.search(rf"(?<!\w){re.escape(title_n)}(?!\w)", query_n):
        score = max(score, 100)
    for canonical in mentioned_course_aliases(query):
        if canonical == title_n:
            score = max(score, 95)
        elif canonical in title_n:
            score = max(score, 90)
        elif title_n in canonical:
            score = max(score, 75)
        # Some catalog titles are more detailed than the everyday alias.
        elif canonical.startswith(title_n) or title_n.startswith(canonical):
            score = max(score, 70)
    # Also handle explicit title mentions with the words in any order.
    title_tokens = [w for w in title_n.split() if len(w) >= 4]
    matched = sum(1 for token in title_tokens if re.search(rf"(?<!\w){re.escape(token)}(?!\w)", query_n))
    if title_tokens and matched:
        score = max(score, int(45 * matched / len(title_tokens)))
    return score


def resolve_course_mention(query: str, courses: list[dict[str, Any]]) -> dict[str, Any] | None:
    scored: list[tuple[int, int, dict[str, Any]]] = []
    for index, course in enumerate(courses):
        score = _course_name_match_score(query, course)
        if not score:
            continue
        full = normalize_text(_course_full_text(course))
        # In this assistant's context, a first-year course is preferred when the
        # catalog includes multiple editions with the same title.
        first_year_bonus = 3 if "1 й курс" in full or "1 курс" in full else 0
        current_year_bonus = 1 if "2026 2027" in full or normalize_text(str(course.get("year", ""))) == "2026 2027" else 0
        scored.append((score + first_year_bonus + current_year_bonus, -index, course))
    if not scored:
        return None
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    # If user used a short alias, course alias matching is already strong enough.
    return scored[0][2]


_TEACHER_HEADINGS = {"преподаватели", "преподаватель", "преподаватели курса", "преподавательский состав"}
_END_HEADINGS = {
    "программа дисциплины", "аннотация", "цель освоения дисциплины",
    "планируемые результаты обучения", "содержание учебной дисциплины",
    "элементы контроля", "промежуточная аттестация", "авторы", "литература",
    "дополнительные материалы в lms", "задать вопрос", "компетенции",
}
_PERSON_RE = re.compile(r"^[А-ЯЁ][а-яё-]+(?:\s+[А-ЯЁ][а-яё-]+){1,3}$")


def extract_teachers(course: dict[str, Any]) -> list[str]:
    """Extract names from the explicit teacher list or legacy labeled sections."""
    candidates: list[str] = []
    # Newer scraper output stores this field explicitly, avoiding HTML-layout ambiguity.
    stored_teachers = course.get("teachers", [])
    if isinstance(stored_teachers, list):
        candidates.extend(str(value).strip() for value in stored_teachers)

    sections = course.get("sections", {})
    if isinstance(sections, dict):
        for heading, values in sections.items():
            if "преподавател" in normalize_text(str(heading)):
                if isinstance(values, list):
                    candidates.extend(str(v).strip() for v in values)
                elif values:
                    candidates.extend(str(values).splitlines())

    lines = [line.strip() for line in str(course.get("full_text", "")).splitlines() if line.strip()]
    start = None
    for i, line in enumerate(lines):
        if normalize_text(line) in _TEACHER_HEADINGS or normalize_text(line).startswith("преподаватели "):
            start = i + 1
            break
    if start is not None:
        for line in lines[start:]:
            normalized = normalize_text(line)
            if normalized in _END_HEADINGS:
                break
            if _PERSON_RE.fullmatch(line) and len(normalized.split()) >= 2:
                candidates.append(line)

    found: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        line = re.sub(r"\s+", " ", item).strip(" •\t")
        if _PERSON_RE.fullmatch(line) and normalize_text(line) not in seen:
            seen.add(normalize_text(line))
            found.append(line)
    return found
