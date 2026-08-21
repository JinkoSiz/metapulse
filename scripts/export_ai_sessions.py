"""Выгрузка «переписки с нейросетью» из сессий Claude Code в docs/ai-sessions/.

Задание требует приложить всю переписку с нейросетью raw JSONL-файлами. У этого
требования две стороны, и закрываем обе:

* runtime-логи вызовов LLM самим сервисом — их пишет app/llm/client.py в logs/llm/*.jsonl;
* переписка времён разработки — сессии Claude Code, которые и выгружает этот скрипт.

Файлы копируются как есть (raw JSONL), но с обязательной чисткой секретов: в транскрипт
сессии могли попасть содержимое .env, ключи API и токены.

Запуск:  python scripts/export_ai_sessions.py [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "docs" / "ai-sessions"
CLAUDE_PROJECTS = Path.home() / ".claude" / "projects"

# Директории сессий Claude Code для этого проекта (кодируются из пути рабочей папки).
PROJECT_DIR_PATTERNS = ("*SomeTest", "*SomeTest-metapulse")

SECRET_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # --- ключи и пароли ---
    (re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}"), "sk-ant-REDACTED"),
    (re.compile(r"\bsk-[A-Za-z0-9]{32,}\b"), "sk-REDACTED"),
    (re.compile(r"\bpa-[A-Za-z0-9_\-]{30,}\b"), "pa-REDACTED"),  # Voyage AI
    (re.compile(r"\bAIza[0-9A-Za-z_\-]{30,}\b"), "AIza-REDACTED"),  # Google/YouTube
    (re.compile(r"(?i)(authorization\"?\s*[:=]\s*\"?bearer\s+)[^\s\"']+"), r"\1REDACTED"),
    (re.compile(r"(?i)(ADMIN_TOKEN\s*[=:]\s*\"?)[^\s\"',]+"), r"\1REDACTED"),
    (re.compile(r"(?i)(POSTGRES_PASSWORD\s*[=:]\s*\"?)[^\s\"',]+"), r"\1REDACTED"),
    (re.compile(r"(?i)(proxy\s*[:=]\s*\"?https?://)[^@\s\"']+@"), r"\1REDACTED@"),
    # 64-символьный hex — типовой формат токенов у самохостящихся сервисов
    (re.compile(r"\b[0-9a-f]{64}\b"), "REDACTED-TOKEN"),
    # --- инфраструктура: транскрипт не должен превращаться в карту чужого сервера ---
    # Границы слова здесь бесполезны: в JSONL переводы строк экранированы, и перед
    # адресом оказывается буква «n» из «\n» — \b такую пару словесных символов не видит.
    (re.compile(r"(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)"), "x.x.x.x"),
    (re.compile(r"(?i)[\w-]*jqubik\.pro"), "service.example.com"),
    (re.compile(r"(?i)giggleshitter"), "server"),
    # Имена учётных записей сервера. Слева границу не проверяем по той же причине,
    # что и у адресов; справа — чтобы не рвать слова, начинающиеся с этих имён.
    (re.compile(r"(?i)(?:jinkosiz|jqubiik|ligma)(?![\w-])"), "user"),
]

# Пути, содержимое которых вырезается целиком: это личные записи, не относящиеся
# к разрабатываемому сервису.
PRIVATE_PATH_MARKERS = ("obsidian", "vault")
PRIVATE_PLACEHOLDER = "[содержимое личных заметок вырезано при выгрузке]"


def redact(text: str) -> tuple[str, int]:
    """Возвращает (очищенный текст, число замен)."""
    total = 0
    for pattern, replacement in SECRET_PATTERNS:
        text, n = pattern.subn(replacement, text)
        total += n
    return text, total


def strip_private_reads(line: str) -> tuple[str, bool]:
    """Вырезает из строки транскрипта содержимое личных файлов.

    Чистка по шаблонам ловит ключи и адреса, но не поможет против целиком прочитанной
    заметки. Поэтому у записей, где фигурирует путь к личному хранилищу, текст
    результата заменяется заглушкой — сам факт обращения остаётся видимым.
    """
    lowered = line.lower()
    if not any(marker in lowered for marker in PRIVATE_PATH_MARKERS):
        return line, False
    try:
        record = json.loads(line)
    except json.JSONDecodeError:
        return line, False

    changed = False

    def walk(node: object) -> object:
        nonlocal changed
        if isinstance(node, dict):
            return {key: walk(value) for key, value in node.items()}
        if isinstance(node, list):
            return [walk(item) for item in node]
        if isinstance(node, str) and len(node) > 400 and any(
            marker in node.lower() for marker in PRIVATE_PATH_MARKERS
        ):
            changed = True
            return PRIVATE_PLACEHOLDER
        return node

    cleaned = walk(record)
    if not changed:
        return line, False
    return json.dumps(cleaned, ensure_ascii=False), True


def find_session_files() -> list[Path]:
    if not CLAUDE_PROJECTS.exists():
        return []
    files: list[Path] = []
    for pattern in PROJECT_DIR_PATTERNS:
        for project_dir in CLAUDE_PROJECTS.glob(pattern):
            files.extend(sorted(project_dir.rglob("*.jsonl")))
    # уникальные пути, стабильный порядок
    return sorted(set(files), key=lambda p: (p.stat().st_mtime, str(p)))


def describe(path: Path) -> tuple[int, str | None]:
    """(число строк, время первой записи) — для манифеста."""
    lines = 0
    first_ts: str | None = None
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            lines += 1
            if first_ts is None:
                try:
                    first_ts = json.loads(line).get("timestamp")
                except (json.JSONDecodeError, AttributeError):
                    pass
    return lines, first_ts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true", help="только показать, что будет выгружено"
    )
    args = parser.parse_args()

    sources = find_session_files()
    if not sources:
        print(f"Сессии не найдены в {CLAUDE_PROJECTS}")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest: list[str] = [
        "# Переписка с нейросетью: сессии разработки",
        "",
        "Сырые JSONL-транскрипты сессий Claude Code, в которых разрабатывался сервис.",
        "",
        "При выгрузке автоматически вычищены: ключи API, пароли и токены; адреса, домены",
        "и имена машин, по которым можно опознать инфраструктуру; содержимое личных",
        "заметок, которые попадали в сессию и к сервису отношения не имеют.",
        "",
        f"Выгружено: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "| Файл | Строк | Начало | Источник |",
        "|---|---:|---|---|",
    ]

    total_redactions = 0
    total_stripped = 0
    for src in sources:
        # сохраняем структуру: главная сессия и подагенты различимы по имени
        rel = src.relative_to(CLAUDE_PROJECTS)
        flat_name = "__".join(rel.parts).replace(".jsonl", "") + ".jsonl"
        dst = OUT_DIR / flat_name

        lines, first_ts = describe(src)
        if args.dry_run:
            print(f"[dry-run] {src} -> {dst.name} ({lines} строк)")
            continue

        cleaned_lines: list[str] = []
        stripped = 0
        replaced = 0
        for line in src.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            line, was_stripped = strip_private_reads(line)
            stripped += int(was_stripped)
            line, n = redact(line)
            replaced += n
            cleaned_lines.append(line)

        total_redactions += replaced
        total_stripped += stripped
        dst.write_text("\n".join(cleaned_lines) + "\n", encoding="utf-8")

        manifest.append(f"| `{dst.name}` | {lines} | {first_ts or '—'} | `{rel.as_posix()}` |")
        print(f"{dst.name}: {lines} строк, замен {replaced}, вырезано личных записей {stripped}")

    if not args.dry_run:
        (OUT_DIR / "README.md").write_text("\n".join(manifest) + "\n", encoding="utf-8")
        print(
            f"\nГотово: {len(sources)} файлов в {OUT_DIR}, "
            f"замен секретов и адресов: {total_redactions}, "
            f"вырезано личных записей: {total_stripped}"
        )


if __name__ == "__main__":
    main()
