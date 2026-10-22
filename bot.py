"""Telegram bot client. It calls the same FastAPI used by the website/mobile app."""
from __future__ import annotations

import asyncio
import html
import logging
import os
import re

import httpx
from aiogram import Bot, Dispatcher, F
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import Command, CommandStart
from aiogram.types import Message
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
API_INTERNAL_TOKEN = os.getenv("API_INTERNAL_TOKEN", "").strip()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("hse-telegram")
dp = Dispatcher()


def plain_from_markdown(text: str) -> str:
    """Telegram fallback formatter; sends safe plain text and keeps URLs readable."""
    text = re.sub(r"\[([^\]]+)\]\((https?://[^\s)]+)\)", r"\1 — \2", text)
    text = text.replace("**", "").replace("__", "").replace("`", "")
    text = re.sub(r"^\s*#{1,6}\s*", "", text, flags=re.MULTILINE)
    return html.escape(text)


async def api_get_courses() -> dict:
    headers = {"X-Internal-Token": API_INTERNAL_TOKEN} if API_INTERNAL_TOKEN else {}
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(f"{API_BASE_URL}/api/courses", headers=headers)
        response.raise_for_status()
        return response.json()


async def api_ask(question: str, session_id: str) -> dict:
    headers = {"X-Internal-Token": API_INTERNAL_TOKEN} if API_INTERNAL_TOKEN else {}
    payload = {"question": question, "session_id": session_id}
    async with httpx.AsyncClient(timeout=150) as client:
        response = await client.post(f"{API_BASE_URL}/api/ask", json=payload, headers=headers)
        response.raise_for_status()
        return response.json()


@dp.message(CommandStart())
async def on_start(message: Message) -> None:
    await message.answer(
        "<b>Привет! Я Навигатор ИВТ.</b>\n\n"
        "Помогаю разобраться в официальных программах дисциплин ИВТ МИЭМ НИУ ВШЭ за 2026/2027 год.\n\n"
        "Задай вопрос обычным сообщением. Команды: /courses — список программ, /help — помощь.\n\n"
        "Я стараюсь отвечать по источникам, но для важных решений сверяйся с официальной программой."
    )


@dp.message(Command("help"))
async def on_help(message: Message) -> None:
    await message.answer(
        "Напиши вопрос, например:\n"
        "• Что изучают на алгебре и геометрии?\n"
        "• Какие формы контроля предусмотрены?\n"
        "• Кто ведёт дисциплину?\n\n"
        "/courses — список загруженных дисциплин."
    )


@dp.message(Command("courses"))
async def on_courses(message: Message) -> None:
    try:
        payload = await api_get_courses()
        courses = payload.get("courses", [])
        if not courses:
            await message.answer("Каталог пока не загружен на сервер. Администратору нужно запустить сборщик.")
            return
        lines = [f"<b>Дисциплины в локальной базе ({len(courses)}):</b>"]
        for course in courses[:80]:
            title = html.escape(str(course.get("title", "Без названия")))
            url = str(course.get("url", ""))
            if url.startswith("https://www.hse.ru/"):
                lines.append(f'• <a href="{html.escape(url, quote=True)}">{title}</a>')
            else:
                lines.append(f"• {title}")
        if len(courses) > 80:
            lines.append("… Полный список доступен на сайте помощника.")
        await message.answer("\n".join(lines), disable_web_page_preview=True)
    except Exception as exc:
        logger.exception("Can't load courses: %s", exc)
        await message.answer("Не получилось получить каталог. Проверь, что сервер API работает.")


@dp.message(F.text)
async def on_question(message: Message) -> None:
    question = (message.text or "").strip()
    if not question or question.startswith("/"):
        return
    if len(question) > 1500:
        await message.answer("Вопрос слишком длинный — сократи его до 1500 символов.")
        return
    status = await message.answer("Ищу в официальных программах…")
    try:
        user_id = message.from_user.id if message.from_user else message.chat.id
        payload = await api_ask(question, f"tg-{user_id}")
        answer = plain_from_markdown(payload.get("answer", "Не удалось получить ответ."))
        sources = payload.get("sources", [])
        if sources:
            links = ["<b>Источники:</b>"]
            seen = set()
            for source in sources:
                url = str(source.get("url", ""))
                title = str(source.get("title", "Источник"))
                if url.startswith("https://www.hse.ru/") and url not in seen:
                    seen.add(url)
                    links.append(f'• <a href="{html.escape(url, quote=True)}">{html.escape(title)}</a>')
            answer += "\n\n" + "\n".join(links)
        # Telegram text length limit is 4096 characters; leave headroom for HTML markup.
        chunks = [answer[i:i + 3500] for i in range(0, max(1, len(answer)), 3500)]
        await status.delete()
        for chunk in chunks:
            await message.answer(chunk, disable_web_page_preview=True)
    except httpx.HTTPStatusError as exc:
        logger.warning("API returned %s: %s", exc.response.status_code, exc.response.text[:300])
        try:
            await status.edit_text("Сервер не смог обработать вопрос. Проверь настройки, данные и квоты API.")
        except Exception:
            pass
    except Exception as exc:
        logger.exception("Question failed: %s", exc)
        try:
            await status.edit_text("Не удалось связаться с сервером. Попробуй позже.")
        except Exception:
            pass


async def main() -> None:
    if not BOT_TOKEN:
        raise RuntimeError("Задай TELEGRAM_BOT_TOKEN в .env (получи токен у @BotFather).")
    bot = Bot(token=BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    logger.info("Telegram bot starting; API=%s", API_BASE_URL)
    try:
        await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())
    finally:
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())
