"""Системные промпты и JSON-схемы структурированных ответов.

Схемы отдаются в `output_config.format` — Anthropic гарантирует валидный JSON ровно
этой формы, поэтому в промптах нет ни слова про «ответь JSON-ом»: там только правила
содержания. `additionalProperties: false` и полный `required` обязательны для
structured outputs.
"""

from __future__ import annotations

from typing import Any

# Резюме отзывов: и критики, и пользователи дают одну и ту же форму — так карточка
# игры рендерит оба блока одним шаблоном.
SUMMARY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "likes": {
            "type": "array",
            "description": (
                "3-6 конкретных плюсов игры, каждый — отдельная короткая фраза на русском языке."
            ),
            "items": {"type": "string"},
        },
        "dislikes": {
            "type": "array",
            "description": (
                "3-6 конкретных минусов игры, каждый — отдельная короткая фраза на русском языке."
            ),
            "items": {"type": "string"},
        },
        "tl_dr": {
            "type": "string",
            "description": "Итог в 1-2 предложениях на русском языке.",
        },
    },
    "required": ["likes", "dislikes", "tl_dr"],
    "additionalProperties": False,
}

LETSPLAY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "conclusion": {
            "type": "string",
            "description": (
                "Заключение об игре на основе летсплея: 4-8 предложений на русском "
                "языке о геймплее, впечатлениях автора и техническом состоянии."
            ),
        },
    },
    "required": ["conclusion"],
    "additionalProperties": False,
}

_COMMON_RULES = """
Правила:
1. Опирайся ТОЛЬКО на текст переданных отзывов. Ничего не додумывай: если о чём-то
   не написано, этого в резюме быть не должно.
2. В likes и dislikes — от 3 до 6 пунктов. Каждый пункт про конкретную сторону игры:
   «боевая система», «оптимизация на PC», «озвучка второстепенных персонажей»,
   «короткая кампания». Оценочные пустышки вида «хорошая игра», «мне понравилось»,
   «плохо» запрещены.
3. Пункт — короткая фраза до 15 слов, без нумерации и без имён рецензентов.
4. Объединяй повторяющиеся претензии в один пункт, а не дублируй их.
5. Если минусов (или плюсов) в отзывах почти нет — верни столько пунктов, сколько
   реально нашлось, вплоть до пустого списка. Выдумывать недостающие нельзя.
6. Весь ответ строго на русском языке, даже если отзывы на английском.
""".strip()

CRITIC_SUMMARY_SYSTEM = f"""
Ты — редактор игрового издания. На вход даны рецензии профессиональных критиков
с Metacritic (оценка по шкале 0-100). Сожми их в структурированное резюме:
что критики хвалят, что ругают, и общий вывод.

{_COMMON_RULES}
7. Учитывай оценку рецензии: разногласие критиков («часть хвалит сюжет, часть
   считает его затянутым») — это ценная информация, отрази её в tl_dr.
""".strip()

USER_SUMMARY_SYSTEM = f"""
Ты — редактор игрового издания. На вход даны отзывы обычных игроков с Metacritic
(оценка по шкале 0-10). Сожми их в структурированное резюме: что игроки хвалят,
что ругают, и общий вывод.

{_COMMON_RULES}
7. Пользовательские отзывы бывают эмоциональными и содержат ревью-бомбинг. Бери из
   них фактические претензии (баги, микротранзакции, требования к железу) и
   игнорируй оскорбления и офтоп.
""".strip()

LETSPLAY_SYSTEM = """
Ты — редактор игрового издания. На вход дан транскрипт летсплея (автоматические
субтитры YouTube: без пунктуации, с ошибками распознавания, с болтовнёй не по теме).

Правила:
1. Опирайся только на транскрипт. Не подставляй знания об игре из других источников.
2. Игнорируй приветствия, просьбы подписаться, рекламу и разговоры не об игре.
3. Расскажи, что происходит в игре, как автор оценивает геймплей, что его радует и
   что раздражает, упомяни технические проблемы, если о них говорят.
4. Если транскрипт обрывочный и понять по нему игру нельзя — честно напиши об этом
   вместо пересказа догадок.
5. Ответ строго на русском языке.
""".strip()

# Отображения для вызывающего кода: kind резюме -> промпт и purpose JSONL-лога.
SUMMARY_SYSTEM_BY_KIND: dict[str, str] = {
    "critic": CRITIC_SUMMARY_SYSTEM,
    "user": USER_SUMMARY_SYSTEM,
}
PURPOSE_BY_KIND: dict[str, str] = {
    "critic": "critic_summary",
    "user": "user_summary",
}
LETSPLAY_PURPOSE = "letsplay_conclusion"


def format_review(
    *,
    index: int,
    kind: str,
    quote: str,
    score: int | None = None,
    author: str | None = None,
    publication: str | None = None,
) -> str:
    """Один отзыв в виде блока для промпта."""
    scale = "100" if kind == "critic" else "10"
    head_parts = [f"#{index}"]
    who = publication or author
    if who:
        head_parts.append(who)
    if score is not None:
        head_parts.append(f"оценка {score}/{scale}")
    return f"[{' | '.join(head_parts)}]\n{quote}"


def build_summary_user_prompt(
    *,
    title: str,
    kind: str,
    reviews: list[str],
    genres: list[str] | None = None,
) -> str:
    """Пользовательская часть запроса на резюме: заголовок игры + корпус отзывов."""
    who = "критиков" if kind == "critic" else "игроков"
    header = f"Игра: {title}"
    if genres:
        header += f"\nЖанры: {', '.join(genres)}"
    header += f"\nОтзывов {who} в выборке: {len(reviews)}"
    body = "\n\n".join(reviews)
    return f"{header}\n\nОтзывы:\n\n{body}"


def build_letsplay_user_prompt(
    *,
    title: str,
    transcript: str,
    video_title: str | None = None,
    channel: str | None = None,
) -> str:
    """Пользовательская часть запроса на заключение по летсплею."""
    header = f"Игра: {title}"
    if video_title:
        header += f"\nРолик: {video_title}"
    if channel:
        header += f"\nКанал: {channel}"
    return f"{header}\n\nТранскрипт:\n\n{transcript}"
