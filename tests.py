"""Быстрые smoke tests без сети и без загрузки embedding-модели: python tests.py"""
from scrape_hse import catalog_courses, extract_course_page
from semantic_search import split_into_chunks
from query_utils import (
    extract_teachers, is_teacher_question, mentioned_course_aliases,
    resolve_course_mention, expand_search_query, is_group_assignment_question,
)


def test_catalog_link_filter():
    html = '''<html><body>
      <a href="/ba/isct/courses/1197128571.html">Алгебра и геометрия</a>
      <a href="/ba/isct/courses/1197128565.html">Алгоритмизация и программирование</a>
      <a href="/ba/isct/courses/?page=2">Следующая</a>
      <a href="https://evil.example/ba/isct/courses/123.html">Не брать</a>
    </body></html>'''
    got = catalog_courses(html, "https://www.hse.ru/ba/isct/courses?year=2026")
    assert len(got) == 2
    assert got[0]["url"] == "https://www.hse.ru/ba/isct/courses/1197128571.html"


def test_course_text_capture():
    html = '''<html><body><nav>Navigation noise</nav><main>
      <h1>Тестовый курс</h1><p>2026/2027</p><p>5</p><p>Кредиты</p>
      <h3>Аннотация</h3><p>Основной текст курса.</p>
      <h3>Элементы контроля</h3><ul><li>Экзамен</li><li>Контрольная работа</li></ul>
      </main><footer>Footer noise</footer></body></html>'''
    course = extract_course_page(html, "https://www.hse.ru/ba/isct/courses/123.html", "Тестовый курс")
    assert course["title"] == "Тестовый курс"
    assert "Основной текст курса" in course["full_text"]
    assert "Экзамен" in course["full_text"]
    assert "Navigation noise" not in course["full_text"]


def test_all_teacher_links_captured_from_nested_html():
    names = [
        "Воронцова Татьяна Дмитриевна",
        "Жукова Лилия Фаилевна",
        "Ищенко Анна Романовна",
        "Марычева Светлана Олеговна",
        "Суховерхова Диана Дмитриевна",
        "Фролов Андрей Александрович",
    ]
    teacher_links = "".join(
        '<div class="teacher-card"><a href="/staff/{0}">{1}</a></div>'.format(i, name)
        for i, name in enumerate(names)
    )
    html = (
        "<html><body><main>"
        "<h1>Математический анализ</h1>"
        "<p>2026/2027</p>"
        "<h4>Преподаватели</h4>"
        f'<div class="teachers">{teacher_links}</div>'
        "<h3>Программа дисциплины</h3>"
        "<h4>Аннотация</h4><p>Тестовый текст программы.</p>"
        "</main></body></html>"
    )
    course = extract_course_page(
        html,
        "https://www.hse.ru/ba/isct/courses/1197128651.html",
        "Математический анализ",
    )
    assert course["teachers"] == names
    assert course["sections"]["Преподаватели"] == names
    for name in names:
        assert name in course["full_text"]
    assert extract_teachers(course) == names


def test_chunking():
    text = "А" * 5000
    chunks = split_into_chunks(text, chunk_chars=1000, overlap=100)
    assert len(chunks) >= 5
    assert all(0 < len(chunk) <= 1000 for chunk in chunks)


def test_slang_teacher_query():
    question = "кто ведет матан"
    assert is_teacher_question(question)
    assert mentioned_course_aliases(question) == ["математический анализ"]
    expanded = expand_search_query(question).casefold()
    assert "математический анализ" in expanded
    assert "преподаватели" in expanded


def test_group_assignment_question_detected():
    assert is_group_assignment_question("а кто у какой группы ведет?")
    assert is_group_assignment_question("Кто ведёт у первой группы?")
    assert not is_group_assignment_question("кто ведет матан")


def test_teacher_extraction_and_course_resolution():
    course = {
        "title": "Математический анализ",
        "url": "https://www.hse.ru/ba/isct/courses/1197128651.html",
        "year": "2026/2027",
        "full_text": "Математический анализ\n2026/2027\nКогда читается:\n1-й курс, 1-4 модуль\nПреподаватели\n"
                     "Воронцова Татьяна Дмитриевна\nЖукова Лилия Фаилевна\n"
                     "Ищенко Анна Романовна\nПрограмма дисциплины\nАннотация\nТекст курса",
        "sections": {"Преподаватели": [
            "Воронцова Татьяна Дмитриевна", "Жукова Лилия Фаилевна", "Ищенко Анна Романовна"
        ]},
    }
    found = resolve_course_mention("кто ведет матан", [course])
    assert found is course
    assert extract_teachers(course) == [
        "Воронцова Татьяна Дмитриевна", "Жукова Лилия Фаилевна", "Ищенко Анна Романовна"
    ]


if __name__ == "__main__":
    test_catalog_link_filter()
    test_course_text_capture()
    test_all_teacher_links_captured_from_nested_html()
    test_chunking()
    test_slang_teacher_query()
    test_group_assignment_question_detected()
    test_teacher_extraction_and_course_resolution()
    print("Все smoke tests пройдены.")
