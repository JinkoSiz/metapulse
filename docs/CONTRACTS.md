# Контракты модулей MetaPulse

Документ фиксирует границы между модулями: каждый пишется независимо и общается с
остальными только через перечисленные здесь сигнатуры и через модели `app/db/models.py`.

## Общее

* Python 3.12, async везде. SQLAlchemy 2.0 (`AsyncSession`), `select()`-стиль, без legacy Query.
* Конфиг — только `from app.config import settings`. Никаких `os.environ` по коду.
* Логи — `structlog.get_logger(__name__)`.
* Любая внешняя сеть — через `httpx.AsyncClient` с таймаутом и ретраями (`tenacity`).
* Комментарии и пользовательские строки — по-русски; имена кода — по-английски.

## 1. `app/metacritic/` — клиент Metacritic

Проверено живыми запросами 2026-08-20 (фикстуры в `tests/fixtures/`):

| Что | Запрос |
|---|---|
| New Releases (первый заход дня) | `GET https://backend.metacritic.com/finder/metacritic/web?componentName=new-releases-carousel&componentDisplayName=Newly+Released&componentType=ProductList&sortBy=-releaseDate&metaScoreMin=1&offset=0&limit=20&mcoTypeId=13` |
| Browse «Новые» (последующие) | `GET .../finder/metacritic/web?sortBy=-releaseDate&productType=games&offset={n}&limit={<=50}` |
| Деталка игры | `GET .../games/metacritic/{slug}/web?apiKey=...` |
| Userscore платформы | `GET .../reviews/metacritic/user/games/{slug}/platform/{platform_slug}/stats/web?apiKey=...` |
| Отзывы критиков | `GET .../reviews/metacritic/critic/games/{slug}/web?apiKey=...&offset={n}&limit=10` |
| Отзывы пользователей | `GET .../reviews/metacritic/user/games/{slug}/web?apiKey=...&offset={n}&limit={<=200}` |
| Обложка | `https://www.metacritic.com/a/img/{image.bucketType}{image.bucketPath}` |

**Критичные особенности, подтверждённые замерами:**

1. **Отзывы критиков игнорируют `limit`** — всегда ровно 10 элементов на страницу.
   Нужна пагинация по `offset` шагом 10 до `data.totalResults`.
   Отзывы пользователей `limit` уважают (проверено до 200).
2. **Userscore платформы берётся ТОЛЬКО path-сегментом** `/platform/{slug}/stats/web`.
   Query-параметр `?filterByPlatform=` молча игнорируется и вернёт данные lead-платформы —
   это тихая порча данных, закрыть тестом.
3. `finder`-элементы **не содержат** списка платформ и почти всегда имеют `userScore.score = null` —
   платформы и userscore берутся из деталки и stats-эндпоинта.
4. `limit` finder-а ограничен 50 (60+ → HTTP 400).
5. Ответы: `{"data": {...}, "links": {"next": {"href": ...}}, "meta": {...}}`.
   Деталка — в `data.item`, списки — в `data.items`, всего — `data.totalResults`.

**Публичный API модуля:**

```python
# app/metacritic/schemas.py — pydantic v2, extra="ignore"
class ScoreStats(BaseModel):
    score: float | None; review_count: int | None; sentiment: str | None

class PlatformInfo(BaseModel):
    mc_id: int | None; name: str; slug: str; is_lead: bool
    release_date: date | None; metascore: ScoreStats | None

class FinderItem(BaseModel):
    mc_id: int; slug: str; title: str; release_date: date | None
    description: str | None; cover_url: str | None
    metascore: int | None; genres: list[str]

class GameDetail(BaseModel):
    mc_id: int; slug: str; title: str; description: str | None
    developer: str | None; publisher: str | None; release_date: date | None
    esrb_rating: str | None; genres: list[str]; cover_url: str | None
    trailer_embed_url: str | None; trailer_title: str | None
    platforms: list[PlatformInfo]; lead_metascore: int | None

class ReviewItem(BaseModel):
    source_key: str      # user: item["id"]; critic: f"{publicationSlug}:{author}:{date}"
    kind: str            # 'critic' | 'user'
    score: int | None; quote: str | None; author: str | None
    publication: str | None; review_date: date | None
    external_url: str | None; platform_slug: str | None; spoiler: bool
```

```python
# app/metacritic/client.py
class MetacriticError(Exception): ...

class MetacriticClient:
    """Асинхронный клиент. Троттлинг settings.mc_rate_limit_rps, ретраи с backoff
    на 403/429/5xx, Chrome UA, опциональный settings.mc_proxy."""
    def __init__(self, client: httpx.AsyncClient | None = None) -> None: ...
    async def __aenter__(self) -> "MetacriticClient": ...
    async def __aexit__(self, *exc) -> None: ...

    async def list_new_releases(self, limit: int = 20) -> list[FinderItem]: ...
    async def list_browse(self, offset: int, limit: int = 50) -> tuple[list[FinderItem], int]:
        """Возвращает (items, total_results)."""
    async def get_game(self, slug: str) -> GameDetail: ...
    async def get_platform_userscore(self, slug: str, platform_slug: str) -> ScoreStats | None: ...
    async def get_critic_reviews(self, slug: str, max_items: int = 40) -> list[ReviewItem]: ...
    async def get_user_reviews(self, slug: str, max_items: int = 60) -> list[ReviewItem]: ...
```

`app/metacritic/parsers.py` — чистые функции `parse_finder_item(dict) -> FinderItem`,
`parse_game_detail(dict) -> GameDetail`, `parse_critic_review(dict)`, `parse_user_review(dict)`,
`build_image_url(image: dict) -> str | None`. Они и покрываются тестами на фикстурах.

## 2. `app/llm/` — LLM-контур

**Требование задания: вся переписка с нейросетью — сырыми JSONL.** Поэтому `LlmClient` —
единственная точка вызова Anthropic/Voyage во всём коде. Ни один другой модуль
не импортирует `anthropic` напрямую.

```python
# app/llm/client.py
PURPOSES = ("critic_summary", "user_summary", "letsplay_conclusion", "embedding")

class LlmDisabled(Exception): ...   # нет ключа / llm_enabled=false

class LlmClient:
    async def complete_structured(
        self, *, purpose: str, system: str, user: str,
        schema: dict, game_id: int | None = None, max_tokens: int | None = None,
    ) -> dict:
        """Anthropic messages.create со structured outputs:
        output_config={"format": {"type": "json_schema", "schema": schema}}.
        Модель settings.llm_model. Возвращает распарсенный JSON-объект."""

    async def embed(
        self, texts: list[str], *, game_id: int | None = None, input_type: str = "document",
    ) -> list[list[float]]:
        """Voyage AI REST https://api.voyageai.com/v1/embeddings, модель settings.embedding_model."""
```

Каждый вызов (успех, ошибка, refusal, эмбеддинги) пишет **одну строку** в
`{settings.llm_log_dir}/YYYY-MM-DD.jsonl`:

```json
{"id":"<uuid4>","ts":"<iso8601>","provider":"anthropic|voyage","model":"...",
 "purpose":"...","game_id":123,"request":{...полные kwargs...},
 "response":{...полный model_dump()...},"usage":{...},"latency_ms":1234,
 "status":"ok|error","error":null}
```

`json.dumps(..., ensure_ascii=False)`, append + flush. Плюс строка в таблице `llm_calls`
(если передана сессия — через `record_llm_call(session, ...)`, иначе только файл).

```python
# app/llm/summarize.py
async def summarize_game(session: AsyncSession, game: Game, kind: str) -> Summary | None:
    """kind: 'critic' | 'user'. Читает reviews из БД, считает input_hash (sha256
    отсортированных source_key+quote). Если хэш совпал с summaries.input_hash — LLM
    не вызывается, возвращается существующая строка. Иначе — вызов LLM и UPSERT.
    Результат схемы: {"likes": [str, ...], "dislikes": [str, ...], "tl_dr": str} на русском."""

# app/llm/embeddings.py
async def ensure_embedding(session: AsyncSession, game: Game) -> bool:
    """Эмбеддинг title + жанры + описание; пропуск, если embedding_hash не изменился."""
async def recompute_similar(session: AsyncSession, game_ids: list[int] | None = None) -> int:
    """Пересчёт similar_games косинусом (pgvector <=>), settings.similar_games_count строк на игру.
    Без ключа Voyage — фолбэк на лексическое сходство (жанры + pg_trgm по названию)."""
```

## 3. `app/youtube/` — доп. часть 1

```python
# app/youtube/service.py
async def process_letsplay(session: AsyncSession, game: Game) -> LetsPlay:
    """1) YouTube Data API v3: search.list(q=f"{title} let's play", type=video,
       order=viewCount, maxResults=10) + videos.list(part=statistics,contentDetails)
       -> максимум viewCount с фильтром по длительности (>= 8 мин).
       2) Транскрипт: youtube_transcript_api (прокси settings.youtube_proxy) ->
       фолбэк yt-dlp --write-auto-subs -> иначе transcript_source='none'.
       3) Заключение: LlmClient.complete_structured(purpose='letsplay_conclusion').
       Любой сбой -> letsplays.error, поле conclusion=None, исключение НЕ пробрасывается."""
```

Кэш: если `letsplays.fetched_at` моложе `settings.letsplay_ttl_days` — ничего не делаем.

## 4. `app/pipeline/` — воркер и оркестрация

```python
# app/pipeline/selection.py  (сердце требования «каждый день заново»)
@dataclass
class Selection:
    items: list[FinderItem]; phase: str; next_offset: int; pages_scanned: int

async def select_batch(session: AsyncSession, client: MetacriticClient,
                       day: date, size: int = 20) -> Selection:
    """Нет строки crawl_state на day -> создаём {phase:'carousel', next_offset:0}.
    phase='carousel': берём list_new_releases(20), фильтруем анти-join'ом по daily_seen,
    переводим состояние в 'browse'. phase='browse': листаем list_browse(next_offset, 50),
    отбрасывая уже виденных сегодня, пока не наберём size или не кончится выдача.
    Возвращает отобранные элементы; запись в daily_seen делает вызывающий код
    ПОСЛЕ успешной обработки игры (идемпотентность при падении)."""

# app/pipeline/events.py
class EventBus:
    async def publish(self, session, run_id, *, stage, message, level="info",
                      game_id=None, payload=None) -> None:
        """Пишет в task_events И публикует JSON в Redis-канал 'metapulse:events'."""
    async def heartbeat(self, worker: str, state: dict) -> None:
        """SETEX metapulse:worker:{worker} 15 <json>."""
    async def workers(self) -> list[dict]:  # чтение живых воркеров для /monitor

# app/pipeline/tasks.py
async def crawl_batch(ctx, trigger: str = "cron") -> dict:
    """Полный цикл: lock -> pipeline_runs(running) -> select_batch -> для каждой игры
    enrich+reviews+summaries+embedding -> daily_seen -> similar -> youtube ->
    финализация run со stats. Redis-lock 'metapulse:lock:crawl' (SET NX EX 3600)."""
async def startup(ctx) -> None    # catch-up при старте, если слот пропущен
async def shutdown(ctx) -> None

# app/pipeline/worker.py
class WorkerSettings:  # arq
    functions = [crawl_batch]; cron_jobs = [cron(crawl_batch, minute=settings.schedule_cron_minute)]
    on_startup = startup; on_shutdown = shutdown; redis_settings = ...
```

## 5. `app/web/` — веб-интерфейс

FastAPI + Jinja2 + HTMX (CDN нельзя — htmx кладём в `app/web/static/`).

| Маршрут | Назначение |
|---|---|
| `GET /` | список игр: `q` (поиск по названию), `platform` (фильтр), `sort` (`metascore`/`userscore`/`date`), `page` |
| `GET /game/{slug}` | карточка: обложка, описание, разработчик, таблица платформ со скорами, трейлер, оба резюме, летсплей, похожие игры |
| `GET /monitor` | live-мониторинг + кнопка принудительного запуска |
| `GET /api/events` | SSE-поток (sse-starlette): события из Redis-канала + heartbeat воркеров + счётчики |
| `POST /api/admin/run` | ставит задачу `crawl_batch(trigger='manual')` в arq; требует заголовок `X-Admin-Token` |
| `GET /api/stats` | счётчики для первичной отрисовки монитора |
| `GET /healthz` | healthcheck для Docker |
| `GET /llm-logs` | список JSONL-файлов + скачивание (deliverable задания) |

Сортировка по рейтингу — `lead_metascore`/`lead_userscore` с `NULLS LAST`.
Поиск — `title ILIKE '%q%'` (индекс gin_trgm_ops). Фильтр — JOIN `game_platforms`.
Похожие игры — из `similar_games`, каждая ссылкой на свою карточку.
