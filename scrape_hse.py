"""Загрузка публичных страниц дисциплин бакалаврской программы ИВТ МИЭМ НИУ ВШЭ."""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlencode, urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup, Tag

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_FILE = DATA_DIR / "courses.json"
CATALOG_URL = "https://www.hse.ru/ba/isct/courses"
DEFAULT_COURSE_FILTER = "1.1.1.4"
COURSE_PATH_RE = re.compile(r"^/ba/isct/courses/\d+\.html$")
USER_AGENT = (
    "HSE-ISCT-Course-Assistant/1.0 "
    "(personal educational project; respectful low-rate access)"
)


def clean_text(value: str) -> str:
    """Normalize whitespace while preserving paragraph boundaries outside this helper."""
    return re.sub(r"\s+", " ", value or "").strip()


_PERSON_NAME_RE = re.compile(r"^[А-ЯЁ][а-яё-]+(?:\s+[А-ЯЁ][а-яё-]+){1,3}$")
_TEACHER_SECTION_ENDS = {
    "программа дисциплины", "аннотация", "цель освоения дисциплины",
    "планируемые результаты обучения", "содержание учебной дисциплины",
    "элементы контроля", "промежуточная аттестация", "список литературы",
    "рекомендуемая основная литература", "авторы", "компетенции",
}


def normalize_heading(value: str) -> str:
    value = (value or "").casefold().replace("ё", "е")
    return re.sub(r"[^0-9a-zа-я]+", " ", value).strip()


def extract_teacher_names(root: Tag) -> list[str]:
    """Collect all names listed under the page's 'Преподаватели' heading.

    HSE renders names as individual links inside nested containers. A scraper that
    only reads p/li/tr elements can miss most names because they are anchor-only.
    """
    headings = root.find_all(re.compile(r"^h[1-6]$"))
    for heading in headings:
        label = normalize_heading(clean_text(heading.get_text(" ", strip=True)))
        if "преподавател" not in label:
            continue

        match = re.fullmatch(r"h([1-6])", heading.name or "")
        heading_level = int(match.group(1)) if match else 6
        found: list[str] = []
        seen: set[str] = set()

        for node in heading.find_all_next():
            if node.name and re.fullmatch(r"h[1-6]", node.name):
                level = int(node.name[1])
                heading_text = normalize_heading(clean_text(node.get_text(" ", strip=True)))
                if level <= heading_level or heading_text in _TEACHER_SECTION_ENDS:
                    break

            if node.name not in {"a", "p", "li", "dt", "dd", "span", "div"}:
                continue
            candidate = clean_text(node.get_text(" ", strip=True))
            if not _PERSON_NAME_RE.fullmatch(candidate):
                continue
            key = normalize_heading(candidate)
            if key not in seen:
                seen.add(key)
                found.append(candidate)

        if found:
            return found
    return []


def canonical_course_url(href: str, base_url: str) -> str | None:
    url = urljoin(base_url, href)
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or parsed.netloc.lower() not in {
        "www.hse.ru", "hse.ru"
    }:
        return None
    if not COURSE_PATH_RE.match(parsed.path):
        return None
    return urlunparse(("https", "www.hse.ru", parsed.path, "", "", ""))


def fetch(session: requests.Session, url: str, timeout: int = 25) -> str:
    response = session.get(url, timeout=timeout)
    response.raise_for_status()
    response.encoding = response.apparent_encoding or response.encoding
    return response.text


def catalog_courses(html: str, page_url: str) -> list[dict[str, str]]:
    soup = BeautifulSoup(html, "html.parser")
    found: list[dict[str, str]] = []
    seen: set[str] = set()

    for anchor in soup.find_all("a", href=True):
        href = str(anchor.get("href", ""))
        url = canonical_course_url(href, page_url)
        if not url or url in seen:
            continue
        title = clean_text(anchor.get_text(" ", strip=True))
        # Avoid collecting empty icon links and hidden navigational anchors.
        if len(title) < 3:
            continue
        seen.add(url)
        found.append({"title": title, "url": url})

    return found


def choose_content_root(soup: BeautifulSoup) -> Tag:
    """Prefer the main article container, with broad fallbacks for HSE markup changes."""
    for selector in ("main", "article", "#content", ".content", ".post", ".b-article"):
        node = soup.select_one(selector)
        if isinstance(node, Tag) and len(node.get_text(" ", strip=True)) > 300:
            return node

    h1 = soup.find("h1")
    if h1:
        # Walk upward until we have a sizeable container but don't grab the full page if avoidable.
        candidate = h1.parent
        best = candidate
        while isinstance(candidate, Tag) and candidate.name not in {"body", "html"}:
            text_len = len(candidate.get_text(" ", strip=True))
            if text_len >= 700:
                best = candidate
                break
            best = candidate
            candidate = candidate.parent
        return best

    return soup.body if soup.body else soup


def extract_course_page(html: str, url: str, fallback_title: str) -> dict[str, Any]:
    soup = BeautifulSoup(html, "html.parser")
    title_node = soup.find("h1")
    title = clean_text(title_node.get_text(" ", strip=True)) if title_node else fallback_title
    root = choose_content_root(soup)

    # Remove active/boilerplate elements while preserving course content.
    for node in root.select("script, style, noscript, nav, header, footer, aside, form, button"):
        node.decompose()

    # The teacher names may be anchors nested in divs, not p/li nodes, so extract them
    # separately before assembling the generic page text.
    teachers = extract_teacher_names(root)

    # Full page text: keep headings, labels, tables, lists, evaluation formulae and literature.
    blocks: list[str] = []
    seen_lines: set[str] = set()
    for node in root.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "dt", "dd", "tr"]):
        value = clean_text(node.get_text(" ", strip=True))
        if not value or len(value) < 2:
            continue
        # Avoid duplicate heading/title text and adjacent duplicate list/table content.
        if value in seen_lines:
            continue
        seen_lines.add(value)
        blocks.append(value)

    # Replace the text between the teacher heading and the next section with the full
    # list found in anchor links. This avoids keeping an incomplete list in RAG context.
    if teachers:
        teacher_heading_idx = next(
            (i for i, value in enumerate(blocks)
             if "преподавател" in normalize_heading(value)
             and len(normalize_heading(value).split()) <= 4),
            None,
        )
        if teacher_heading_idx is not None:
            end_idx = len(blocks)
            for i in range(teacher_heading_idx + 1, len(blocks)):
                if normalize_heading(blocks[i]) in _TEACHER_SECTION_ENDS:
                    end_idx = i
                    break
            blocks = blocks[:teacher_heading_idx + 1] + teachers + blocks[end_idx:]
        else:
            blocks.extend(["Преподаватели", *teachers])

    full_text = "\n".join(blocks)
    if len(full_text) < 200:
        full_text = "\n".join(
            line for line in (clean_text(x) for x in root.stripped_strings) if line
        )
        for name in teachers:
            if name not in full_text:
                full_text += f"\n{name}"

    # Capture headings and the text following them as machine-readable sections too.
    sections: dict[str, list[str]] = {}
    active_section = "Основная информация"
    for node in root.find_all(["h2", "h3", "h4", "h5", "p", "li", "dt", "dd"]):
        value = clean_text(node.get_text(" ", strip=True))
        if not value:
            continue
        if node.name in {"h2", "h3", "h4", "h5"}:
            active_section = value
            sections.setdefault(active_section, [])
        else:
            values = sections.setdefault(active_section, [])
            if not values or values[-1] != value:
                values.append(value)

    # Store the teacher roster explicitly; generic section parsing does not always see
    # all names because they are anchor-only children in the current HSE markup.
    if teachers:
        teacher_key = next(
            (key for key in sections if "преподавател" in normalize_heading(key)),
            "Преподаватели",
        )
        sections[teacher_key] = teachers

    # Metadata is informational only; full_text remains the source of truth.
    joined = "\n".join(blocks)
    credit_match = re.search(r"(?:^|\n)(\d+(?:[.,]\d+)?)\s*\n\s*Кредиты", joined, re.I)
    year_match = re.search(r"20\d{2}\s*/\s*20\d{2}", joined)
    language = None
    if re.search(r"русск", joined, re.I):
        language = "русский"
    elif re.search(r"английск|English", joined, re.I):
        language = "английский"

    return {
        "title": title,
        "url": url,
        "year": year_match.group(0).replace(" ", "") if year_match else "2026/2027",
        "credits": credit_match.group(1).replace(",", ".") if credit_match else None,
        "language": language,
        "teachers": teachers,
        "full_text": full_text[:100_000],
        "sections": {key: vals for key, vals in sections.items() if vals},
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }


def build_catalog_url(page: int, year: int, course_filter: str) -> str:
    params = {"course": course_filter, "page": str(page), "year": str(year)}
    return f"{CATALOG_URL}?{urlencode(params)}"


def scrape(limit: int, max_pages: int, year: int, course_filter: str, delay: float) -> list[dict[str, Any]]:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept-Language": "ru,en;q=0.8"})

    catalog: list[dict[str, str]] = []
    seen_urls: set[str] = set()
    consecutive_empty = 0

    for page in range(1, max_pages + 1):
        list_url = build_catalog_url(page, year, course_filter)
        print(f"[каталог] Страница {page}: {list_url}")
        try:
            html = fetch(session, list_url)
        except requests.RequestException as exc:
            print(f"[предупреждение] Не удалось загрузить каталог: {exc}", file=sys.stderr)
            break

        page_items = catalog_courses(html, list_url)
        new_items = [item for item in page_items if item["url"] not in seen_urls]
        for item in new_items:
            seen_urls.add(item["url"])
            catalog.append(item)
            if len(catalog) >= limit:
                break
        consecutive_empty = consecutive_empty + 1 if not new_items else 0
        if len(catalog) >= limit or consecutive_empty >= 1:
            break
        time.sleep(delay)

    if not catalog:
        raise RuntimeError(
            "В каталоге не найдены ссылки на страницы дисциплин. "
            "Возможно, ВШЭ изменила HTML или сайт временно недоступен."
        )

    courses: list[dict[str, Any]] = []
    for i, item in enumerate(catalog[:limit], start=1):
        print(f"[{i}/{min(limit, len(catalog))}] {item['title']}")
        try:
            html = fetch(session, item["url"])
            course = extract_course_page(html, item["url"], item["title"])
            course["catalog_title"] = item["title"]
            course["catalog_url"] = build_catalog_url(1, year, course_filter)
            course["scrape_status"] = "ok" if len(course["full_text"]) >= 200 else "short_text"
            courses.append(course)
        except requests.RequestException as exc:
            print(f"  [ошибка] {exc}", file=sys.stderr)
            courses.append({
                "title": item["title"],
                "url": item["url"],
                "year": f"{year}/{year + 1}",
                "full_text": "",
                "sections": {},
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "scrape_status": f"error: {exc}",
            })
        time.sleep(delay)

    return courses


def main() -> int:
    parser = argparse.ArgumentParser(description="Собрать страницы курсов ИВТ МИЭМ с сайта ВШЭ")
    parser.add_argument("--limit", type=int, default=30, help="Сколько дисциплин собрать (по умолчанию: 30)")
    parser.add_argument("--max-pages", type=int, default=5, help="Максимум страниц каталога")
    parser.add_argument("--year", type=int, default=2026, help="Первый год учебного года, например 2026 для 2026/2027")
    parser.add_argument("--course-filter", default=DEFAULT_COURSE_FILTER, help="Значение параметра course из URL каталога")
    parser.add_argument("--delay", type=float, default=0.8, help="Пауза между запросами в секундах")
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE, help="Файл JSON для результатов")
    args = parser.parse_args()

    if args.limit < 1 or args.max_pages < 1 or args.delay < 0:
        parser.error("--limit и --max-pages должны быть >= 1; --delay должен быть >= 0")

    try:
        courses = scrape(args.limit, args.max_pages, args.year, args.course_filter, args.delay)
    except (RuntimeError, requests.RequestException) as exc:
        print(f"Ошибка сбора: {exc}", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(courses, ensure_ascii=False, indent=2), encoding="utf-8")
    successful = sum(1 for c in courses if c.get("scrape_status") == "ok")
    print(f"\nСохранено {len(courses)} записей: {args.output}")
    print(f"Полноценно загружено страниц: {successful}/{len(courses)}")
    print("Дальше запусти: python assistant.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
