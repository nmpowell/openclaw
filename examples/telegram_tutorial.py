#!/usr/bin/env python3
"""
========================================================================
  Building a Telegram Bot Integration in Python — From MVP to Full
========================================================================

A progressive tutorial that builds a complete Telegram bot integration
from scratch, modelled on OpenClaw's production TypeScript implementation.
Companion to heartbeat_tutorial.py — this covers the *channel* layer.

Each chapter adds one layer of capability.  Every chapter is runnable.
No external dependencies required — Telegram API calls are mocked so
the tutorial runs standalone.  Comments show exactly where aiogram or
python-telegram-bot would plug in.

    python examples/telegram_tutorial.py

Source of truth (TypeScript):
    src/telegram/monitor.ts           — polling transport
    src/telegram/webhook.ts           — webhook transport
    src/telegram/send.ts              — outbound send + retry + media
    src/telegram/bot-handlers.ts      — inbound handler chain
    src/telegram/bot-message-context.ts — message context building
    src/telegram/bot-message-dispatch.ts — agent dispatch
    src/telegram/bot-access.ts        — access control
    src/telegram/format.ts            — markdown → Telegram HTML
    src/telegram/draft-stream.ts      — streaming draft edits
    src/telegram/targets.ts           — target parsing
    src/telegram/caption.ts           — caption splitting
    src/auto-reply/chunk.ts           — text chunking
    src/auto-reply/inbound-debounce.ts — message debouncing

Table of contents:
    Chapter 1  — MVP: Receive and Reply                        (~50 lines)
    Chapter 2  — Access Control                                (~60 lines)
    Chapter 3  — Message Context Building                      (~60 lines)
    Chapter 4  — Markdown to Telegram HTML                     (~50 lines)
    Chapter 5  — Text Chunking                                 (~60 lines)
    Chapter 6  — Target Parsing & Threading                    (~40 lines)
    Chapter 7  — Inbound Debouncing                            (~50 lines)
    Chapter 8  — Streaming Draft Edits                         (~60 lines)
    Chapter 9  — Delivery Pipeline                             (~80 lines)
    Chapter 10 — Webhook Transport                             (~50 lines)
    Chapter 11 — Heartbeat Integration                         (~50 lines)
    Chapter 12 — Full Bot                                      (~120 lines)
    Appendix A — Use-case gallery                              (examples)
"""

from __future__ import annotations

import asyncio
import html
import json
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable

log = logging.getLogger("telegram")


# ╔════════════════════════════════════════════════════════════╗
# ║  CHAPTER 1 — MVP: Receive and Reply                      ║
# ╚════════════════════════════════════════════════════════════╝
#
# Telegram bots work via the Bot API (https://core.telegram.org/bots/api).
# It's free, no payment required.  You create a bot by messaging
# @BotFather, who gives you a token like "123456:ABC-DEF".
#
# There are two ways to receive messages:
#
#   1. POLLING (getUpdates) — your bot calls Telegram's API in a loop.
#      Telegram holds the connection open until a message arrives (long-poll).
#      No public IP needed.  This is what OpenClaw uses by default via
#      @grammyjs/runner (grammy library).
#
#   2. WEBHOOK — Telegram POSTs updates to your HTTPS endpoint.
#      Requires a public URL.  Covered in Chapter 10.
#
# The response path is simple: call sendMessage with the chat_id.
#
# Mirrors: src/telegram/monitor.ts (polling), src/telegram/send.ts (send)

@dataclass
class TelegramUpdate:
    """One update from getUpdates.  Simplified from Telegram's full schema."""
    update_id: int
    chat_id: int
    chat_type: str           # "private", "group", "supergroup"
    from_id: int
    from_username: str | None
    from_first_name: str
    text: str
    message_id: int
    # Real Telegram updates also carry: media, location, stickers,
    # reply_to_message, forward_from, etc.


@dataclass
class SendResult:
    """Result of sendMessage."""
    message_id: int
    chat_id: int


class MockTelegramAPI:
    """
    Stands in for the real Telegram Bot API.
    In production you'd use:
      - aiogram:  Bot(token).send_message(chat_id, text)
      - python-telegram-bot:  bot.send_message(chat_id, text)
      - raw HTTP:  POST https://api.telegram.org/bot{token}/sendMessage

    All three ultimately hit the same HTTPS endpoint.
    """

    def __init__(self):
        self.sent: list[dict] = []
        self._update_queue: list[TelegramUpdate] = []
        self._next_msg_id = 1000

    def enqueue_update(self, update: TelegramUpdate):
        """Simulate an incoming message from a user."""
        self._update_queue.append(update)

    async def get_updates(self, offset: int = 0, timeout: int = 30) -> list[TelegramUpdate]:
        """
        Long-poll for updates.  In the real API this blocks until
        a message arrives or timeout elapses.

        Mirrors: grammy's runner calling getUpdates in a loop.
        """
        # Simulate: wait briefly, then return queued updates
        await asyncio.sleep(0.05)
        results = [u for u in self._update_queue if u.update_id >= offset]
        self._update_queue = [u for u in self._update_queue if u.update_id < offset]
        return results

    async def send_message(
        self,
        chat_id: int,
        text: str,
        parse_mode: str | None = None,
        reply_to_message_id: int | None = None,
        reply_markup: dict | None = None,
        **kwargs,
    ) -> SendResult:
        """
        Send a text message.

        Mirrors: sendMessageTelegram() in src/telegram/send.ts
        which calls bot.api.sendMessage().
        """
        self._next_msg_id += 1
        self.sent.append({
            "chat_id": chat_id,
            "text": text,
            "parse_mode": parse_mode,
            "reply_to_message_id": reply_to_message_id,
            "reply_markup": reply_markup,
            **kwargs,
        })
        return SendResult(message_id=self._next_msg_id, chat_id=chat_id)

    async def edit_message_text(
        self,
        chat_id: int,
        message_id: int,
        text: str,
        parse_mode: str | None = None,
    ) -> SendResult:
        """Edit an existing message (used for streaming drafts)."""
        self.sent.append({
            "action": "edit",
            "chat_id": chat_id,
            "message_id": message_id,
            "text": text,
            "parse_mode": parse_mode,
        })
        return SendResult(message_id=message_id, chat_id=chat_id)

    async def send_photo(self, chat_id: int, photo: str, caption: str = "", **kw) -> SendResult:
        self._next_msg_id += 1
        self.sent.append({"action": "photo", "chat_id": chat_id, "photo": photo, "caption": caption, **kw})
        return SendResult(message_id=self._next_msg_id, chat_id=chat_id)

    async def send_document(self, chat_id: int, document: str, caption: str = "", **kw) -> SendResult:
        self._next_msg_id += 1
        self.sent.append({"action": "document", "chat_id": chat_id, "document": document, "caption": caption, **kw})
        return SendResult(message_id=self._next_msg_id, chat_id=chat_id)

    async def send_voice(self, chat_id: int, voice: str, caption: str = "", **kw) -> SendResult:
        self._next_msg_id += 1
        self.sent.append({"action": "voice", "chat_id": chat_id, "voice": voice, "caption": caption, **kw})
        return SendResult(message_id=self._next_msg_id, chat_id=chat_id)


def _make_update(uid: int, chat_id: int, text: str, **kw) -> TelegramUpdate:
    return TelegramUpdate(
        update_id=uid,
        chat_id=chat_id,
        chat_type=kw.get("chat_type", "private"),
        from_id=kw.get("from_id", 42),
        from_username=kw.get("from_username", "alice"),
        from_first_name=kw.get("from_first_name", "Alice"),
        text=text,
        message_id=kw.get("message_id", uid * 10),
    )


async def chapter_1():
    """MVP: receive a message, send a reply."""

    api = MockTelegramAPI()

    # Simulate Alice sending "Hello"
    api.enqueue_update(_make_update(1, chat_id=12345, text="Hello bot!"))

    # Polling loop (one iteration for demo)
    updates = await api.get_updates(offset=0)
    for update in updates:
        # In the real code, this is where grammy's bot.on("message") fires,
        # which eventually calls dispatchTelegramMessage() → agent → reply.
        reply_text = f"You said: {update.text}"
        result = await api.send_message(update.chat_id, reply_text)
        print(f"  Received: {update.text!r} from @{update.from_username}")
        print(f"  Replied:  {reply_text!r} (msg_id={result.message_id})")

    assert len(api.sent) == 1
    assert api.sent[0]["chat_id"] == 12345


# ╔════════════════════════════════════════════════════════════╗
# ║  CHAPTER 2 — Access Control                              ║
# ╚════════════════════════════════════════════════════════════╝
#
# Not everyone should be able to talk to your bot.  OpenClaw has
# a layered access control system:
#
#   DM policy:    allowlist | open | pairing | disabled
#   Group policy: allowlist | open | disabled
#   allowFrom:    list of user IDs and/or @usernames
#
# The allowFrom list supports:
#   - Numeric Telegram user IDs:  12345678
#   - @usernames:                 "@alice"
#   - Wildcard:                   "*" (allow everyone)
#
# "pairing" mode is interesting: the first user to message the bot
# gets auto-added to the allowlist (like Bluetooth pairing).
#
# Mirrors: src/telegram/bot-access.ts

class DmPolicy(Enum):
    ALLOWLIST = "allowlist"    # only allowFrom users
    OPEN = "open"             # anyone can DM
    PAIRING = "pairing"       # first user auto-paired
    DISABLED = "disabled"     # reject all DMs


class GroupPolicy(Enum):
    ALLOWLIST = "allowlist"   # only listed groups
    OPEN = "open"             # any group
    DISABLED = "disabled"     # reject all groups


@dataclass
class AccessConfig:
    dm_policy: DmPolicy = DmPolicy.ALLOWLIST
    group_policy: GroupPolicy = GroupPolicy.ALLOWLIST
    allow_from: list[str | int] = field(default_factory=list)
    group_allow_from: list[str | int] = field(default_factory=list)


def normalize_allow_entry(entry: str | int) -> str:
    """Normalize an allowFrom entry for comparison."""
    s = str(entry).strip().lower()
    if s.startswith("@"):
        s = s[1:]
    return s


def is_sender_allowed(
    update: TelegramUpdate,
    config: AccessConfig,
) -> tuple[bool, str]:
    """
    Check if a sender is allowed to interact with the bot.
    Returns (allowed, reason).

    Mirrors: isSenderAllowed() in src/telegram/bot-access.ts
    """
    is_group = update.chat_type in ("group", "supergroup")

    if is_group:
        if config.group_policy == GroupPolicy.DISABLED:
            return False, "groups-disabled"
        if config.group_policy == GroupPolicy.OPEN:
            return True, "group-open"
        # allowlist mode: check group chat_id
        normalized_group = [normalize_allow_entry(e) for e in config.group_allow_from]
        if str(update.chat_id) in normalized_group:
            return True, "group-allowlisted"
        return False, "group-not-allowed"

    # DM
    if config.dm_policy == DmPolicy.DISABLED:
        return False, "dms-disabled"
    if config.dm_policy == DmPolicy.OPEN:
        return True, "dm-open"
    if config.dm_policy == DmPolicy.PAIRING:
        # In real code: check if anyone has paired yet.
        # If not, auto-pair this user.  If yes, check the stored list.
        return True, "dm-pairing"

    # allowlist mode
    normalized = [normalize_allow_entry(e) for e in config.allow_from]
    if "*" in normalized:
        return True, "wildcard"

    sender_id = str(update.from_id)
    sender_username = normalize_allow_entry(update.from_username or "")

    if sender_id in normalized:
        return True, "id-match"
    if sender_username and sender_username in normalized:
        return True, "username-match"

    return False, "not-in-allowlist"


async def chapter_2():
    """Access control: who can talk to the bot?"""

    config = AccessConfig(
        dm_policy=DmPolicy.ALLOWLIST,
        allow_from=["@alice", 99999],
        group_policy=GroupPolicy.ALLOWLIST,
        group_allow_from=[-100123456],
    )

    cases = [
        ("Alice DM (by username)",  _make_update(1, 42, "hi", from_username="alice")),
        ("Bob DM (not listed)",     _make_update(2, 43, "hi", from_id=88888, from_username="bob")),
        ("User 99999 DM (by ID)",   _make_update(3, 44, "hi", from_id=99999, from_username="charlie")),
        ("Allowed group",           _make_update(4, -100123456, "hi", chat_type="supergroup")),
        ("Unknown group",           _make_update(5, -100999999, "hi", chat_type="supergroup")),
    ]

    for label, update in cases:
        allowed, reason = is_sender_allowed(update, config)
        status = "ALLOWED" if allowed else "DENIED"
        print(f"  {label:30s} → {status:7s} ({reason})")


# ╔════════════════════════════════════════════════════════════╗
# ║  CHAPTER 3 — Message Context Building                    ║
# ╚════════════════════════════════════════════════════════════╝
#
# Before sending a message to the LLM, OpenClaw wraps it in a
# structured envelope that gives the model context about who
# sent it, when, from which channel, etc.
#
# Format:  [Telegram @alice +2m] How's the deploy?
#
# For groups:  [Telegram #dev-chat @alice +5m] Check the logs
#
# The +elapsed tells the LLM how long ago the message arrived
# relative to the last activity, helping it understand conversation
# flow and urgency.
#
# Mirrors: src/auto-reply/envelope.ts, src/telegram/bot-message-context.ts

@dataclass
class InboundEnvelope:
    """Structured wrapper around an incoming message for the LLM."""
    provider: str                  # "telegram", "whatsapp", etc.
    sender_label: str              # "@alice" or "Alice (12345)"
    group_label: str | None        # "#dev-chat" or None for DMs
    body: str                      # the actual message text
    elapsed_label: str | None      # "+2m", "+1h", None if first message
    message_id: str | None = None
    media_urls: list[str] = field(default_factory=list)


def build_sender_label(update: TelegramUpdate) -> str:
    """
    Build a human-readable sender label.
    Prefer @username, fall back to "FirstName (id)".

    Mirrors: buildSenderLabel() in src/telegram/bot/helpers.ts
    """
    if update.from_username:
        return f"@{update.from_username}"
    return f"{update.from_first_name} ({update.from_id})"


def build_group_label(update: TelegramUpdate) -> str | None:
    """Build a group label like '#chat-name'.  None for DMs."""
    if update.chat_type in ("group", "supergroup"):
        return f"#{update.chat_id}"
    return None


def format_elapsed(seconds: float | None) -> str | None:
    """Format seconds into a human-readable elapsed label."""
    if seconds is None or seconds < 0:
        return None
    if seconds < 60:
        return f"+{int(seconds)}s"
    if seconds < 3600:
        return f"+{int(seconds / 60)}m"
    if seconds < 86400:
        return f"+{seconds / 3600:.1f}h"
    return f"+{seconds / 86400:.1f}d"


def format_envelope(envelope: InboundEnvelope) -> str:
    """
    Format the envelope as a single string for the LLM.
    This becomes the "user" message in the chat history.

    Example output:
        [Telegram @alice +2m] How's the deploy?
        [Telegram #dev-chat @bob +5m] Check the logs

    Mirrors: formatInboundEnvelope() in src/auto-reply/envelope.ts
    """
    parts = [envelope.provider]
    if envelope.group_label:
        parts.append(envelope.group_label)
    parts.append(envelope.sender_label)
    if envelope.elapsed_label:
        parts.append(envelope.elapsed_label)
    header = " ".join(parts)
    return f"[{header}] {envelope.body}"


def build_message_context(
    update: TelegramUpdate,
    last_activity_ts: float | None = None,
) -> InboundEnvelope:
    """
    Build the full message context from a Telegram update.

    Mirrors: buildTelegramMessageContext() in src/telegram/bot-message-context.ts
    """
    now = time.time()
    elapsed = (now - last_activity_ts) if last_activity_ts else None

    return InboundEnvelope(
        provider="Telegram",
        sender_label=build_sender_label(update),
        group_label=build_group_label(update),
        body=update.text,
        elapsed_label=format_elapsed(elapsed),
        message_id=str(update.message_id),
    )


async def chapter_3():
    """Message context building."""

    t0 = time.time()

    # DM from Alice, first message (no elapsed)
    dm = _make_update(1, 42, "How's the deploy?", from_username="alice")
    env1 = build_message_context(dm, last_activity_ts=None)
    print(f"  DM:    {format_envelope(env1)}")

    # Group message from Bob, 5 minutes after last activity
    grp = _make_update(2, -100555, "Check the logs", chat_type="supergroup",
                        from_id=99, from_username="bob")
    env2 = build_message_context(grp, last_activity_ts=t0 - 300)
    print(f"  Group: {format_envelope(env2)}")

    # DM from user without username, 2 hours after last activity
    no_user = _make_update(3, 77, "Hello", from_id=77, from_username=None,
                           from_first_name="Charlie")
    env3 = build_message_context(no_user, last_activity_ts=t0 - 7200)
    print(f"  No @:  {format_envelope(env3)}")


# ╔════════════════════════════════════════════════════════════╗
# ║  CHAPTER 4 — Markdown to Telegram HTML                   ║
# ╚════════════════════════════════════════════════════════════╝
#
# Telegram supports a subset of HTML for message formatting:
#   <b>bold</b>  <i>italic</i>  <code>mono</code>
#   <pre>code block</pre>  <a href="url">link</a>
#
# LLMs typically produce Markdown.  We need to convert:
#   **bold** → <b>bold</b>
#   *italic* → <i>italic</i>
#   `code`   → <code>code</code>
#   ```block``` → <pre>block</pre>
#   [text](url) → <a href="url">text</a>
#
# Critical: HTML entities (&, <, >) must be escaped FIRST,
# before any tag insertion.
#
# If the conversion produces invalid HTML that Telegram rejects
# (error 400: "can't parse entities"), fall back to plain text.
#
# Mirrors: src/telegram/format.ts (renderTelegramHtmlText)

def escape_telegram_html(text: str) -> str:
    """Escape HTML entities for Telegram.  Must run before tag insertion."""
    return html.escape(text, quote=False)


def markdown_to_telegram_html(text: str) -> str:
    """
    Convert common Markdown to Telegram-compatible HTML.

    This is a simplified version.  The real OpenClaw implementation
    (renderTelegramHtmlText) also handles:
      - Nested formatting
      - Markdown tables → <pre> blocks or compact text
      - Strikethrough ~~text~~ → <s>text</s>
      - Spoiler ||text|| → <tg-spoiler>text</tg-spoiler>
      - Block quotes > text → <blockquote>text</blockquote>

    Mirrors: src/telegram/format.ts
    """
    # Step 1: Protect code blocks from entity escaping.
    # Extract fenced blocks and inline code first.
    code_blocks: list[str] = []
    inline_codes: list[str] = []

    def stash_fenced(m: re.Match) -> str:
        code_blocks.append(m.group(1))
        return f"\x00FENCED{len(code_blocks) - 1}\x00"

    def stash_inline(m: re.Match) -> str:
        inline_codes.append(m.group(1))
        return f"\x00INLINE{len(inline_codes) - 1}\x00"

    result = re.sub(r"```(?:\w*\n)?(.*?)```", stash_fenced, text, flags=re.DOTALL)
    result = re.sub(r"`([^`]+)`", stash_inline, result)

    # Step 2: Escape HTML entities in the remaining text.
    result = escape_telegram_html(result)

    # Step 3: Convert Markdown formatting to HTML tags.
    # Bold: **text** or __text__
    result = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", result)
    result = re.sub(r"__(.+?)__", r"<b>\1</b>", result)
    # Italic: *text* or _text_ (but not inside words like file_name)
    result = re.sub(r"(?<!\w)\*([^*]+?)\*(?!\w)", r"<i>\1</i>", result)
    result = re.sub(r"(?<!\w)_([^_]+?)_(?!\w)", r"<i>\1</i>", result)
    # Links: [text](url)
    result = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', result)

    # Step 4: Restore code blocks with HTML tags.
    for i, block in enumerate(code_blocks):
        escaped_block = escape_telegram_html(block)
        result = result.replace(f"\x00FENCED{i}\x00", f"<pre>{escaped_block}</pre>")
    for i, code in enumerate(inline_codes):
        escaped_code = escape_telegram_html(code)
        result = result.replace(f"\x00INLINE{i}\x00", f"<code>{escaped_code}</code>")

    return result


async def send_with_html_fallback(
    api: MockTelegramAPI,
    chat_id: int,
    markdown_text: str,
) -> SendResult:
    """
    Try sending as HTML; if Telegram rejects the formatting,
    fall back to plain text.

    Mirrors: the catch(PARSE_ERR_RE) block in src/telegram/send.ts
    """
    html_text = markdown_to_telegram_html(markdown_text)
    try:
        return await api.send_message(chat_id, html_text, parse_mode="HTML")
    except Exception:
        # Telegram returned 400: "can't parse entities"
        # Fall back to plain text (strip all tags)
        plain = re.sub(r"<[^>]+>", "", html_text)
        return await api.send_message(chat_id, plain)


async def chapter_4():
    """Markdown → Telegram HTML conversion."""

    samples = [
        "**Bold** and *italic* text",
        "Use `inline code` here",
        "```python\nprint('hello')\n```",
        "[Click here](https://example.com)",
        "Escape <html> & entities",
        "**Bold with `code` inside**",
    ]

    for md in samples:
        converted = markdown_to_telegram_html(md)
        print(f"  MD:   {md!r}")
        print(f"  HTML: {converted!r}")
        print()


# ╔════════════════════════════════════════════════════════════╗
# ║  CHAPTER 5 — Text Chunking                               ║
# ╚════════════════════════════════════════════════════════════╝
#
# Telegram messages are limited to 4096 characters.
# Captions on media are limited to 1024 characters.
#
# When the LLM produces a long reply, we need to split it into
# chunks that fit within these limits.
#
# OpenClaw has three chunking strategies:
#
#   1. Length-based:     Find last newline or space within limit
#   2. Paragraph-based:  Split on blank lines, pack into chunks
#   3. Markdown-aware:   Respect code fences (close + reopen)
#
# Mirrors: src/auto-reply/chunk.ts, src/telegram/caption.ts

TELEGRAM_TEXT_LIMIT = 4096
TELEGRAM_CAPTION_LIMIT = 1024


def chunk_text(text: str, limit: int = TELEGRAM_TEXT_LIMIT) -> list[str]:
    """
    Split text into chunks that fit within `limit` characters.
    Prefers breaking at newlines, then spaces, then hard-breaks.

    Mirrors: chunkText() in src/auto-reply/chunk.ts
    """
    if len(text) <= limit:
        return [text] if text.strip() else []

    chunks: list[str] = []
    remaining = text

    while remaining:
        if len(remaining) <= limit:
            if remaining.strip():
                chunks.append(remaining)
            break

        window = remaining[:limit]

        # Prefer breaking at last newline
        nl_pos = window.rfind("\n")
        if nl_pos > limit // 4:  # don't break too early
            chunks.append(remaining[:nl_pos].rstrip())
            remaining = remaining[nl_pos + 1:]
            continue

        # Fall back to last space (word boundary)
        sp_pos = window.rfind(" ")
        if sp_pos > limit // 4:
            chunks.append(remaining[:sp_pos].rstrip())
            remaining = remaining[sp_pos + 1:]
            continue

        # Hard break at limit
        chunks.append(remaining[:limit])
        remaining = remaining[limit:]

    return chunks


def chunk_by_paragraph(text: str, limit: int = TELEGRAM_TEXT_LIMIT) -> list[str]:
    """
    Split on paragraph boundaries (blank lines), packing multiple
    paragraphs into chunks up to `limit`.

    Falls back to length-based splitting if a single paragraph
    exceeds the limit.

    Mirrors: chunkByParagraph() in src/auto-reply/chunk.ts
    """
    paragraphs = re.split(r"\n\s*\n", text)
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if not current:
            current = para
        elif len(current) + 2 + len(para) <= limit:
            current += "\n\n" + para
        else:
            chunks.append(current)
            current = para

    if current:
        chunks.append(current)

    # Split any oversized chunks
    final: list[str] = []
    for chunk in chunks:
        if len(chunk) <= limit:
            final.append(chunk)
        else:
            final.extend(chunk_text(chunk, limit))

    return final


def split_caption(text: str) -> tuple[str, str | None]:
    """
    Split text for media messages: caption (≤1024) + overflow.
    If text fits in 1024, returns (text, None).
    Otherwise finds a good break point and returns (caption, follow_up).

    Mirrors: splitTelegramCaption() in src/telegram/caption.ts
    """
    if len(text) <= TELEGRAM_CAPTION_LIMIT:
        return text, None

    # Find break point near the limit
    window = text[:TELEGRAM_CAPTION_LIMIT]
    nl_pos = window.rfind("\n")
    if nl_pos > TELEGRAM_CAPTION_LIMIT // 2:
        return text[:nl_pos].rstrip(), text[nl_pos + 1:]

    sp_pos = window.rfind(" ")
    if sp_pos > TELEGRAM_CAPTION_LIMIT // 2:
        return text[:sp_pos].rstrip(), text[sp_pos + 1:]

    return text[:TELEGRAM_CAPTION_LIMIT], text[TELEGRAM_CAPTION_LIMIT:]


async def chapter_5():
    """Text chunking for Telegram's message limits."""

    # Short text — no chunking needed
    short = "Hello world"
    chunks = chunk_text(short)
    print(f"  Short ({len(short)} chars): {len(chunks)} chunk(s)")

    # Long text — split at paragraph boundaries
    paras = "\n\n".join([f"Paragraph {i}: " + "x" * 200 for i in range(25)])
    chunks = chunk_by_paragraph(paras, limit=4096)
    print(f"  Long  ({len(paras)} chars): {len(chunks)} chunk(s), "
          f"sizes: {[len(c) for c in chunks]}")

    # Caption splitting
    long_caption = "A" * 800 + "\n\nMore text here: " + "B" * 400
    caption, follow_up = split_caption(long_caption)
    print(f"  Caption split: caption={len(caption)} chars, "
          f"follow_up={len(follow_up) if follow_up else 0} chars")

    # Verify no chunk exceeds the limit
    huge = "word " * 2000
    chunks = chunk_text(huge, limit=4096)
    max_len = max(len(c) for c in chunks)
    print(f"  Huge  ({len(huge)} chars): {len(chunks)} chunks, max={max_len} (limit=4096)")
    assert all(len(c) <= 4096 for c in chunks)


# ╔════════════════════════════════════════════════════════════╗
# ║  CHAPTER 6 — Target Parsing & Threading                  ║
# ╚════════════════════════════════════════════════════════════╝
#
# When sending a message, the "to" target can come in many formats:
#   "12345"                  — numeric chat ID
#   "@username"              — public username
#   "telegram:12345"         — prefixed (from internal routing)
#   "tg:group:-100555"       — prefixed group
#   "12345:topic:99"         — chat with forum topic thread ID
#
# Telegram supergroups can be "forum-enabled" (topics).  Each topic
# has a message_thread_id.  The "General" topic has id=1, but the
# Telegram API silently rejects it — you must omit the param.
#
# Reply threading uses reply_to_message_id to create reply chains.
#
# Mirrors: src/telegram/targets.ts, src/telegram/bot/helpers.ts

@dataclass
class ParsedTarget:
    """Result of parsing a send target."""
    chat_id: str
    message_thread_id: int | None = None


def parse_telegram_target(raw: str) -> ParsedTarget:
    """
    Parse a target string into chat_id + optional thread_id.

    Mirrors: parseTelegramTarget() in src/telegram/targets.ts
    """
    # Strip known prefixes
    cleaned = raw.strip()
    for prefix in ("telegram:group:", "telegram:", "tg:group:", "tg:"):
        if cleaned.lower().startswith(prefix):
            cleaned = cleaned[len(prefix):]
            break

    # Check for :topic:N suffix
    topic_match = re.match(r"^(.+?):topic:(\d+)$", cleaned)
    if topic_match:
        chat_id = topic_match.group(1)
        thread_id = int(topic_match.group(2))
        # General topic (id=1) is rejected by Telegram API — omit it.
        if thread_id == 1:
            thread_id = None
        return ParsedTarget(chat_id=chat_id, message_thread_id=thread_id)

    # Strip t.me/ links → @username
    tme_match = re.match(r"^(?:https?://)?t\.me/([^/?]+)", cleaned)
    if tme_match:
        return ParsedTarget(chat_id=f"@{tme_match.group(1)}")

    return ParsedTarget(chat_id=cleaned)


async def chapter_6():
    """Target parsing and threading."""

    targets = [
        "12345",
        "@mybot",
        "telegram:12345",
        "tg:group:-100555",
        "-100555:topic:42",
        "-100555:topic:1",       # General topic → thread_id omitted
        "https://t.me/mybot",
    ]

    for raw in targets:
        parsed = parse_telegram_target(raw)
        thread = f", thread={parsed.message_thread_id}" if parsed.message_thread_id else ""
        print(f"  {raw:30s} → chat_id={parsed.chat_id}{thread}")


# ╔════════════════════════════════════════════════════════════╗
# ║  CHAPTER 7 — Inbound Debouncing                          ║
# ╚════════════════════════════════════════════════════════════╝
#
# Users often send multiple messages in quick succession:
#   "hey"
#   "can you check"
#   "the deploy status?"
#
# Without debouncing, the bot would process each as a separate
# request, generating 3 independent LLM calls.  With debouncing,
# we buffer messages for a short window (e.g. 500ms) and batch
# them into a single prompt.
#
# Additionally, Telegram splits long pastes at 4096 characters.
# The text-fragment buffer detects these splits (consecutive
# message IDs, each near 4096 chars) and reassembles them.
#
# Mirrors: src/auto-reply/inbound-debounce.ts, src/telegram/bot-handlers.ts

class InboundDebouncer:
    """
    Buffer rapid messages from the same chat, flush after a delay.

    Per-key buffering: different chats debounce independently.
    Control commands (like /start) bypass the debounce and flush
    any pending buffer immediately.

    Mirrors: createInboundDebouncer() in src/auto-reply/inbound-debounce.ts
    """

    def __init__(
        self,
        debounce_ms: float = 500,
        on_flush: Callable[[list[TelegramUpdate]], Awaitable[None]] | None = None,
        is_control_command: Callable[[str], bool] | None = None,
    ):
        self._debounce_secs = debounce_ms / 1000.0
        self._on_flush = on_flush or (lambda msgs: asyncio.sleep(0))
        self._is_control = is_control_command or (lambda t: t.startswith("/"))
        self._buffers: dict[int, list[TelegramUpdate]] = {}
        self._timers: dict[int, asyncio.TimerHandle] = {}

    async def enqueue(self, update: TelegramUpdate):
        """Add an incoming message to the debounce buffer."""
        chat_id = update.chat_id
        text = update.text or ""

        # Control commands bypass debouncing and flush current buffer
        if self._is_control(text):
            await self._flush_key(chat_id)
            await self._on_flush([update])
            return

        # Add to buffer
        if chat_id not in self._buffers:
            self._buffers[chat_id] = []
        self._buffers[chat_id].append(update)

        # Reset the flush timer
        if chat_id in self._timers:
            self._timers[chat_id].cancel()

        loop = asyncio.get_running_loop()
        self._timers[chat_id] = loop.call_later(
            self._debounce_secs,
            lambda cid=chat_id: asyncio.ensure_future(self._flush_key(cid)),
        )

    async def _flush_key(self, chat_id: int):
        """Flush all buffered messages for a chat."""
        timer = self._timers.pop(chat_id, None)
        if timer:
            timer.cancel()
        messages = self._buffers.pop(chat_id, [])
        if messages:
            await self._on_flush(messages)


def reassemble_text_fragments(
    messages: list[TelegramUpdate],
    threshold: int = 4000,
) -> str:
    """
    Detect and reassemble Telegram's text-fragment splits.

    Telegram splits messages at 4096 chars.  If consecutive messages
    from the same sender each start near that limit, they're likely
    fragments of a single paste.

    Mirrors: the text fragment buffering in src/telegram/bot-handlers.ts
    """
    if len(messages) == 1:
        return messages[0].text

    parts: list[str] = []
    for i, msg in enumerate(messages):
        text = msg.text or ""
        if i > 0 and len(messages[i - 1].text) >= threshold:
            # Previous message was near the limit — this is a continuation
            parts.append(text)
        else:
            if parts:
                parts.append("\n")
            parts.append(text)

    return "".join(parts)


async def chapter_7():
    """Inbound debouncing."""

    flushed: list[list[TelegramUpdate]] = []

    async def on_flush(msgs: list[TelegramUpdate]):
        flushed.append(msgs)
        combined = reassemble_text_fragments(msgs)
        print(f"    Flushed {len(msgs)} msg(s): {combined!r}")

    debouncer = InboundDebouncer(debounce_ms=200, on_flush=on_flush)

    # Rapid-fire messages from the same chat
    print("  Sending 3 rapid messages from chat 42...")
    await debouncer.enqueue(_make_update(1, 42, "hey"))
    await debouncer.enqueue(_make_update(2, 42, "can you check"))
    await debouncer.enqueue(_make_update(3, 42, "the deploy?"))
    await asyncio.sleep(0.4)
    print(f"  Flush count: {len(flushed)} (expected 1 — all 3 batched)")
    assert len(flushed) == 1

    # Control command bypasses debounce
    flushed.clear()
    print("\n  Sending normal msg then /status command...")
    await debouncer.enqueue(_make_update(4, 42, "thinking about it"))
    await debouncer.enqueue(_make_update(5, 42, "/status"))
    await asyncio.sleep(0.4)
    print(f"  Flush count: {len(flushed)} (expected 2 — buffer flushed + command)")


# ╔════════════════════════════════════════════════════════════╗
# ║  CHAPTER 8 — Streaming Draft Edits                       ║
# ╚════════════════════════════════════════════════════════════╝
#
# LLMs stream tokens one at a time.  Instead of making the user
# wait for the full response, we can send a "draft" message and
# keep editing it as more tokens arrive.
#
# Telegram's editMessageText API supports this.  The stream is:
#   1. Send initial message with first few tokens
#   2. As more tokens arrive, call editMessageText
#   3. Throttle edits to avoid rate limits (300ms default)
#   4. Stop editing if text exceeds 4096 chars (wait for chunked delivery)
#   5. On completion, delete the draft and send the final chunked reply
#
# Key optimisation: don't send an edit if the text hasn't changed.
#
# Mirrors: src/telegram/draft-stream.ts

class DraftStream:
    """
    Stream partial LLM responses as editable Telegram messages.

    Mirrors: createTelegramDraftStream() in src/telegram/draft-stream.ts
    """

    THROTTLE_SECS = 0.3         # 300ms between edits
    MAX_CHARS = 4096            # Telegram message limit

    def __init__(self, api: MockTelegramAPI, chat_id: int):
        self._api = api
        self._chat_id = chat_id
        self._draft_msg_id: int | None = None
        self._last_sent_text = ""
        self._last_sent_at = 0.0
        self._pending_text = ""
        self._in_flight = False
        self._stopped = False
        self._timer: asyncio.TimerHandle | None = None

    async def update(self, text: str):
        """
        Queue new text for the draft.  Throttled to avoid API spam.

        Called on every LLM token / chunk.
        """
        if self._stopped:
            return
        if len(text) > self.MAX_CHARS:
            # Text too long for a single message — stop streaming,
            # let the final delivery handle chunking.
            self._stopped = True
            return

        self._pending_text = text
        await self._maybe_flush()

    async def _maybe_flush(self):
        if self._in_flight or self._stopped:
            return

        text = self._pending_text.rstrip()
        if not text or text == self._last_sent_text:
            return

        # Throttle: wait if we sent recently
        now = time.time()
        elapsed = now - self._last_sent_at
        if elapsed < self.THROTTLE_SECS:
            delay = self.THROTTLE_SECS - elapsed
            if self._timer is None:
                loop = asyncio.get_running_loop()
                self._timer = loop.call_later(
                    delay,
                    lambda: asyncio.ensure_future(self._do_flush()),
                )
            return

        await self._do_flush()

    async def _do_flush(self):
        if self._timer:
            self._timer.cancel()
            self._timer = None
        if self._stopped:
            return

        text = self._pending_text.rstrip()
        if not text or text == self._last_sent_text:
            return

        self._in_flight = True
        try:
            if self._draft_msg_id is None:
                # First chunk: send a new message
                result = await self._api.send_message(self._chat_id, text)
                self._draft_msg_id = result.message_id
            else:
                # Subsequent chunks: edit the existing message
                await self._api.edit_message_text(
                    self._chat_id, self._draft_msg_id, text,
                )
            self._last_sent_text = text
            self._last_sent_at = time.time()
        except Exception:
            # On failure, stop streaming to avoid error loops.
            self._stopped = True
        finally:
            self._in_flight = False

        # If more text arrived while we were sending, flush again
        if self._pending_text.rstrip() != self._last_sent_text:
            await self._maybe_flush()

    def stop(self):
        self._stopped = True
        if self._timer:
            self._timer.cancel()

    @property
    def draft_message_id(self) -> int | None:
        return self._draft_msg_id


async def chapter_8():
    """Streaming draft edits."""

    api = MockTelegramAPI()
    draft = DraftStream(api, chat_id=42)

    # Simulate streaming tokens
    tokens = ["The ", "deploy ", "looks ", "good. ", "All ", "142 ", "tests ", "passed."]
    accumulated = ""
    for token in tokens:
        accumulated += token
        await draft.update(accumulated)
        await asyncio.sleep(0.1)  # simulate token arrival rate

    # Allow final flush
    await asyncio.sleep(0.5)
    draft.stop()

    # Count API calls: should be throttled (not 8 calls for 8 tokens)
    edits = [s for s in api.sent if s.get("action") == "edit"]
    sends = [s for s in api.sent if "action" not in s]
    print(f"  Tokens streamed:  {len(tokens)}")
    print(f"  Initial sends:    {len(sends)} (first draft message)")
    print(f"  Edit calls:       {len(edits)} (throttled updates)")
    print(f"  Total API calls:  {len(api.sent)} (vs {len(tokens)} tokens)")
    if api.sent:
        last = api.sent[-1]
        last_text = last.get("text", "")
        print(f"  Final draft text: {last_text!r}")


# ╔════════════════════════════════════════════════════════════╗
# ║  CHAPTER 9 — Delivery Pipeline                           ║
# ╚════════════════════════════════════════════════════════════╝
#
# The full outbound pipeline handles:
#   1. Chunk the text (Ch.5)
#   2. Render each chunk as HTML (Ch.4)
#   3. Send with retry on transient errors
#   4. Handle media: detect type, caption split, send media API
#   5. Attach inline keyboard buttons
#   6. Reply threading (reply_to modes: off / first / all)
#   7. Thread fallback (retry without thread if topic not found)
#
# Mirrors: src/telegram/bot/delivery.ts, src/telegram/send.ts

class ReplyToMode(Enum):
    OFF = "off"        # never reply to the original message
    FIRST = "first"    # reply only on first outbound message
    ALL = "all"        # reply on every chunk


@dataclass
class ReplyPayload:
    """One unit of outbound content."""
    text: str = ""
    media_url: str | None = None
    media_type: str | None = None       # "photo", "voice", "document"
    buttons: list[list[dict]] | None = None  # inline keyboard grid


@dataclass
class DeliveryConfig:
    """How to deliver replies to Telegram."""
    chat_id: int
    reply_to_mode: ReplyToMode = ReplyToMode.FIRST
    reply_to_message_id: int | None = None
    message_thread_id: int | None = None
    text_limit: int = TELEGRAM_TEXT_LIMIT
    chunk_mode: str = "paragraph"       # "length" or "paragraph"


async def deliver_reply(
    api: MockTelegramAPI,
    config: DeliveryConfig,
    payload: ReplyPayload,
) -> list[SendResult]:
    """
    Deliver a single ReplyPayload, handling chunking, media, and threading.

    Mirrors: deliverReplies() in src/telegram/bot/delivery.ts
    """
    results: list[SendResult] = []
    chunk_idx = 0

    def should_reply_to(idx: int) -> int | None:
        if config.reply_to_message_id is None:
            return None
        if config.reply_to_mode == ReplyToMode.OFF:
            return None
        if config.reply_to_mode == ReplyToMode.FIRST and idx == 0:
            return config.reply_to_message_id
        if config.reply_to_mode == ReplyToMode.ALL:
            return config.reply_to_message_id
        return None

    thread_kwargs: dict[str, Any] = {}
    if config.message_thread_id:
        thread_kwargs["message_thread_id"] = config.message_thread_id

    # Handle media
    if payload.media_url:
        caption, follow_up = split_caption(payload.text)
        html_caption = markdown_to_telegram_html(caption) if caption else ""

        send_fn = {
            "photo": api.send_photo,
            "voice": api.send_voice,
            "document": api.send_document,
        }.get(payload.media_type or "document", api.send_document)

        try:
            result = await send_fn(
                config.chat_id,
                payload.media_url,
                caption=html_caption,
                reply_to_message_id=should_reply_to(0),
                **thread_kwargs,
            )
            results.append(result)
            chunk_idx = 1
        except Exception:
            # Thread not found → retry without thread
            if thread_kwargs:
                result = await send_fn(
                    config.chat_id, payload.media_url, caption=html_caption,
                )
                results.append(result)
                chunk_idx = 1

        # Send overflow text as separate message
        if follow_up:
            payload = ReplyPayload(text=follow_up, buttons=payload.buttons)
        else:
            return results

    # Handle text (with chunking)
    text = payload.text
    if not text.strip():
        return results

    if config.chunk_mode == "paragraph":
        chunks = chunk_by_paragraph(text, config.text_limit)
    else:
        chunks = chunk_text(text, config.text_limit)

    for i, chunk in enumerate(chunks):
        idx = chunk_idx + i
        html_text = markdown_to_telegram_html(chunk)
        reply_markup = None
        # Buttons only on the first chunk
        if idx == 0 and payload.buttons:
            reply_markup = {"inline_keyboard": payload.buttons}

        try:
            result = await api.send_message(
                config.chat_id,
                html_text,
                parse_mode="HTML",
                reply_to_message_id=should_reply_to(idx),
                reply_markup=reply_markup,
                **thread_kwargs,
            )
        except Exception:
            # Thread fallback: retry without thread params
            if thread_kwargs:
                result = await api.send_message(
                    config.chat_id, html_text, parse_mode="HTML",
                )
            else:
                raise
        results.append(result)

    return results


async def chapter_9():
    """Full delivery pipeline."""

    api = MockTelegramAPI()
    config = DeliveryConfig(
        chat_id=42,
        reply_to_mode=ReplyToMode.FIRST,
        reply_to_message_id=100,
    )

    # Simple text reply
    print("  --- Simple text reply ---")
    results = await deliver_reply(api, config, ReplyPayload(
        text="**Deploy succeeded!** All tests passed.",
    ))
    print(f"  Sent {len(results)} message(s)")
    print(f"  HTML: {api.sent[-1]['text']!r}")
    print(f"  Reply-to: {api.sent[-1].get('reply_to_message_id')}")

    # Long text (chunked)
    api.sent.clear()
    print("\n  --- Long reply (chunked) ---")
    long_text = "\n\n".join([f"Section {i}: " + "x" * 500 for i in range(12)])
    results = await deliver_reply(api, config, ReplyPayload(text=long_text))
    print(f"  Sent {len(results)} chunk(s) for {len(long_text)} chars")
    # Only first chunk should have reply_to
    print(f"  Chunk 0 reply_to: {api.sent[0].get('reply_to_message_id')}")
    if len(api.sent) > 1:
        print(f"  Chunk 1 reply_to: {api.sent[1].get('reply_to_message_id')}")

    # Media with caption
    api.sent.clear()
    print("\n  --- Photo with caption ---")
    results = await deliver_reply(api, config, ReplyPayload(
        text="Here's the dashboard screenshot",
        media_url="https://example.com/dashboard.png",
        media_type="photo",
    ))
    print(f"  Sent {len(results)} message(s)")
    print(f"  Action: {api.sent[0].get('action')}, caption: {api.sent[0].get('caption')!r}")

    # With inline buttons
    api.sent.clear()
    print("\n  --- Reply with buttons ---")
    results = await deliver_reply(api, config, ReplyPayload(
        text="Choose a model:",
        buttons=[[
            {"text": "GPT-4", "callback_data": "model:gpt4"},
            {"text": "Claude", "callback_data": "model:claude"},
        ]],
    ))
    print(f"  Has keyboard: {api.sent[0].get('reply_markup') is not None}")


# ╔════════════════════════════════════════════════════════════╗
# ║  CHAPTER 10 — Webhook Transport                          ║
# ╚════════════════════════════════════════════════════════════╝
#
# An alternative to polling: Telegram POSTs updates to your
# HTTPS endpoint.  Advantages:
#   - Lower latency (no polling interval)
#   - Scales better (no per-bot polling connection)
#   - Works behind load balancers
#
# Disadvantages:
#   - Requires public HTTPS URL (or a tunnel like ngrok)
#   - More infrastructure to manage
#
# OpenClaw's webhook server:
#   - Creates a Node.js HTTP server on configurable port (default 8787)
#   - Registers the webhook URL via bot.api.setWebhook()
#   - Validates a secret_token header on each request
#   - Has a /healthz endpoint for monitoring
#
# Mirrors: src/telegram/webhook.ts

class WebhookServer:
    """
    Minimal webhook server using asyncio + aiohttp patterns.

    In production you'd use:
      - aiohttp:    web.Application() + web.post("/webhook", handler)
      - FastAPI:    @app.post("/webhook")
      - grammy:     webhookCallback(bot, "http")
      - aiogram:    SimpleRequestHandler(dispatcher, bot)

    Mirrors: startTelegramWebhook() in src/telegram/webhook.ts
    """

    def __init__(
        self,
        path: str = "/telegram-webhook",
        health_path: str = "/healthz",
        secret: str | None = None,
        port: int = 8787,
    ):
        self.path = path
        self.health_path = health_path
        self.secret = secret
        self.port = port
        self._handler: Callable[[TelegramUpdate], Awaitable[None]] | None = None

    def set_handler(self, handler: Callable[[TelegramUpdate], Awaitable[None]]):
        self._handler = handler

    async def handle_request(self, method: str, path: str, headers: dict, body: bytes) -> tuple[int, str]:
        """
        Process an incoming HTTP request.

        Returns (status_code, body).
        """
        # Health check
        if path == self.health_path:
            return 200, "ok"

        # Reject non-POST and wrong paths
        if path != self.path or method != "POST":
            return 404, ""

        # Validate secret token
        if self.secret:
            token = headers.get("x-telegram-bot-api-secret-token", "")
            if token != self.secret:
                return 403, "invalid secret"

        # Parse update
        try:
            data = json.loads(body)
            msg = data.get("message", {})
            update = TelegramUpdate(
                update_id=data.get("update_id", 0),
                chat_id=msg.get("chat", {}).get("id", 0),
                chat_type=msg.get("chat", {}).get("type", "private"),
                from_id=msg.get("from", {}).get("id", 0),
                from_username=msg.get("from", {}).get("username"),
                from_first_name=msg.get("from", {}).get("first_name", ""),
                text=msg.get("text", ""),
                message_id=msg.get("message_id", 0),
            )
            if self._handler:
                await self._handler(update)
            return 200, "ok"
        except Exception as e:
            return 500, str(e)


async def chapter_10():
    """Webhook transport."""

    received: list[TelegramUpdate] = []

    async def on_update(update: TelegramUpdate):
        received.append(update)
        print(f"    Webhook received: {update.text!r} from @{update.from_username}")

    server = WebhookServer(secret="my-secret-token")
    server.set_handler(on_update)

    # Simulate Telegram POSTing an update
    webhook_body = json.dumps({
        "update_id": 1,
        "message": {
            "message_id": 10,
            "from": {"id": 42, "username": "alice", "first_name": "Alice"},
            "chat": {"id": 42, "type": "private"},
            "text": "Hello via webhook!",
        },
    }).encode()

    # Good request with correct secret
    status, body = await server.handle_request(
        "POST", "/telegram-webhook",
        {"x-telegram-bot-api-secret-token": "my-secret-token"},
        webhook_body,
    )
    print(f"  Valid request:   status={status}")

    # Bad secret
    status, body = await server.handle_request(
        "POST", "/telegram-webhook",
        {"x-telegram-bot-api-secret-token": "wrong"},
        webhook_body,
    )
    print(f"  Bad secret:      status={status}")

    # Health check
    status, body = await server.handle_request("GET", "/healthz", {}, b"")
    print(f"  Health check:    status={status}, body={body!r}")

    # Wrong path
    status, body = await server.handle_request("POST", "/other", {}, b"")
    print(f"  Wrong path:      status={status}")

    assert len(received) == 1


# ╔════════════════════════════════════════════════════════════╗
# ║  CHAPTER 11 — Heartbeat Integration                      ║
# ╚════════════════════════════════════════════════════════════╝
#
# This connects the Telegram channel to the heartbeat system
# from heartbeat_tutorial.py.  Two integration points:
#
#   1. DELIVERY CALLBACK: Heartbeat alerts are delivered to Telegram
#      via the same send pipeline (chunk → HTML → sendMessage).
#
#   2. INCOMING TRIGGER: When a user messages the bot, it can
#      wake the heartbeat system (e.g. to relay an exec result
#      that just completed).
#
# The heartbeat system doesn't know about Telegram — it just
# calls deliver_callback(target, text).  The Telegram layer
# provides that callback.
#
# Mirrors: src/infra/heartbeat-runner.ts, src/infra/outbound/targets.ts

@dataclass
class HeartbeatDeliveryTarget:
    """Where heartbeat alerts should be delivered."""
    channel: str            # "telegram", "whatsapp", etc.
    chat_id: int
    account_id: str | None = None


async def chapter_11():
    """Heartbeat ↔ Telegram integration."""

    api = MockTelegramAPI()
    deliveries: list[dict] = []

    # 1. The delivery callback — called by the heartbeat system
    async def telegram_deliver(target: HeartbeatDeliveryTarget, text: str):
        """
        Bridge function: heartbeat system → Telegram send pipeline.

        In the real code, this is resolved by resolveHeartbeatDeliveryTarget()
        which picks the channel from the session's last activity, then
        deliverOutboundPayloads() calls the channel plugin's sendText().
        """
        config = DeliveryConfig(chat_id=target.chat_id)
        payload = ReplyPayload(text=text)
        results = await deliver_reply(api, config, payload)
        deliveries.append({"target": target, "text": text, "results": results})
        print(f"    Heartbeat → Telegram {target.chat_id}: {text!r}")

    # 2. Simulate heartbeat alerts being delivered
    target = HeartbeatDeliveryTarget(channel="telegram", chat_id=42)

    await telegram_deliver(target, "Your staging deploy has been stuck for 45 minutes.")
    await telegram_deliver(target, "**Build passed.** All 142 tests green.")

    print(f"  Total deliveries: {len(deliveries)}")
    print(f"  API calls: {len(api.sent)}")

    # 3. Simulate an incoming message triggering a heartbeat wake
    print("\n  --- Incoming message triggers heartbeat wake ---")

    wake_reasons: list[str] = []

    def request_heartbeat_wake(reason: str):
        """Called when incoming activity should wake the heartbeat."""
        wake_reasons.append(reason)
        print(f"    Wake requested: reason={reason!r}")

    # Simulate: user sends a message, and there's a pending exec result
    update = _make_update(10, 42, "What happened with the build?")
    print(f"  User says: {update.text!r}")

    # The message handler might detect pending system events and wake
    has_pending_exec = True  # simulated
    if has_pending_exec:
        request_heartbeat_wake("exec-event")

    print(f"  Wake reasons queued: {wake_reasons}")


# ╔════════════════════════════════════════════════════════════╗
# ║  CHAPTER 12 — Full Bot                                   ║
# ╚════════════════════════════════════════════════════════════╝
#
# This composes all layers into a single TelegramBot class that
# you could adapt for any Python project.
#
# Architecture:
#   TelegramBot
#     ├─ MockTelegramAPI (or aiogram.Bot in production)
#     ├─ AccessConfig (Ch.2)
#     ├─ InboundDebouncer (Ch.7)
#     ├─ DraftStream pool (Ch.8)
#     ├─ DeliveryConfig defaults (Ch.9)
#     └─ HeartbeatDeliveryTarget (Ch.11)
#
# The polling loop:
#   1. getUpdates → access check → debounce → build context
#   2. Send context to LLM → stream draft → final reply
#   3. Deliver via chunked HTML pipeline

@dataclass
class BotConfig:
    """
    Full bot configuration, mirroring OpenClaw's Telegram config.

    Mirrors: TelegramAccountConfig in src/config/types.telegram.ts
    """
    token: str = "MOCK_TOKEN"
    dm_policy: DmPolicy = DmPolicy.ALLOWLIST
    allow_from: list[str | int] = field(default_factory=list)
    group_policy: GroupPolicy = GroupPolicy.DISABLED
    group_allow_from: list[str | int] = field(default_factory=list)
    debounce_ms: float = 500
    text_chunk_limit: int = TELEGRAM_TEXT_LIMIT
    chunk_mode: str = "paragraph"
    stream_mode: str = "partial"       # "off", "partial", "block"
    reply_to_mode: ReplyToMode = ReplyToMode.FIRST
    heartbeat_chat_id: int | None = None


# Type for the LLM callback.
AgentCallback = Callable[[str, str], Awaitable[str]]
# agent_callback(agent_id, formatted_prompt) → reply text


class TelegramBot:
    """
    Complete Telegram bot composing all tutorial layers.

    Usage:
        bot = TelegramBot(
            config=BotConfig(token="123:ABC", allow_from=["@you"]),
            agent_callback=my_llm_function,
        )
        await bot.start()
        # ... bot runs until stopped ...
        await bot.stop()
    """

    def __init__(
        self,
        config: BotConfig,
        agent_callback: AgentCallback,
        api: MockTelegramAPI | None = None,  # inject for testing
    ):
        self.config = config
        self._agent = agent_callback
        self.api = api or MockTelegramAPI()
        self._access = AccessConfig(
            dm_policy=config.dm_policy,
            allow_from=config.allow_from,
            group_policy=config.group_policy,
            group_allow_from=config.group_allow_from,
        )
        self._debouncer = InboundDebouncer(
            debounce_ms=config.debounce_ms,
            on_flush=self._process_batch,
        )
        self._running = False
        self._poll_task: asyncio.Task | None = None
        self._last_activity: dict[int, float] = {}
        self._stats = {"received": 0, "processed": 0, "denied": 0, "sent": 0}

    async def start(self):
        """Start the polling loop."""
        self._running = True
        self._poll_task = asyncio.create_task(self._poll_loop())

    async def stop(self):
        """Stop the bot gracefully."""
        self._running = False
        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass

    async def _poll_loop(self):
        """
        Long-polling loop.

        In production with aiogram:
            dp = Dispatcher()
            await dp.start_polling(bot)

        Mirrors: monitorTelegramProvider() in src/telegram/monitor.ts
        with @grammyjs/runner for concurrent processing.
        """
        offset = 0
        while self._running:
            try:
                updates = await self.api.get_updates(offset=offset)
                for update in updates:
                    offset = update.update_id + 1
                    self._stats["received"] += 1
                    await self._handle_update(update)
            except asyncio.CancelledError:
                break
            except Exception:
                # Backoff on error (real code uses exponential backoff)
                await asyncio.sleep(1.0)

    async def _handle_update(self, update: TelegramUpdate):
        """Handle a single incoming update."""
        # Access control (Ch.2)
        allowed, reason = is_sender_allowed(update, self._access)
        if not allowed:
            self._stats["denied"] += 1
            return

        # Debounce (Ch.7)
        await self._debouncer.enqueue(update)

    async def _process_batch(self, messages: list[TelegramUpdate]):
        """Process a debounced batch of messages."""
        if not messages:
            return

        last_msg = messages[-1]
        chat_id = last_msg.chat_id

        # Build context (Ch.3)
        last_ts = self._last_activity.get(chat_id)
        combined_text = reassemble_text_fragments(messages)
        envelope = InboundEnvelope(
            provider="Telegram",
            sender_label=build_sender_label(last_msg),
            group_label=build_group_label(last_msg),
            body=combined_text,
            elapsed_label=format_elapsed(
                (time.time() - last_ts) if last_ts else None,
            ),
            message_id=str(last_msg.message_id),
        )
        prompt = format_envelope(envelope)

        # Call agent
        try:
            reply_text = await self._agent("default", prompt)
        except Exception:
            reply_text = "Sorry, something went wrong."

        self._stats["processed"] += 1
        self._last_activity[chat_id] = time.time()

        # Deliver reply (Ch.9)
        config = DeliveryConfig(
            chat_id=chat_id,
            reply_to_mode=self.config.reply_to_mode,
            reply_to_message_id=last_msg.message_id,
            text_limit=self.config.text_chunk_limit,
            chunk_mode=self.config.chunk_mode,
        )
        results = await deliver_reply(self.api, config, ReplyPayload(text=reply_text))
        self._stats["sent"] += len(results)

    # ── Heartbeat delivery interface ──

    async def deliver_heartbeat(self, text: str, chat_id: int | None = None):
        """
        Deliver a heartbeat alert via this bot.
        Uses the configured heartbeat_chat_id or explicit chat_id.
        """
        target_id = chat_id or self.config.heartbeat_chat_id
        if not target_id:
            return
        config = DeliveryConfig(chat_id=target_id)
        await deliver_reply(self.api, config, ReplyPayload(text=text))
        self._stats["sent"] += 1


async def chapter_12():
    """Full bot integration demo."""

    banner = "=" * 64

    # Mock LLM
    async def mock_agent(agent_id: str, prompt: str) -> str:
        if "deploy" in prompt.lower():
            return "**Deploy status:** All services healthy. Last deploy 2h ago."
        if "help" in prompt.lower():
            return "I can help with:\n\n- Deploy status\n- Build monitoring\n- Alert management"
        return f"Echo: {prompt.split('] ')[-1] if '] ' in prompt else prompt}"

    # Build bot
    api = MockTelegramAPI()
    bot = TelegramBot(
        config=BotConfig(
            allow_from=["@alice", "@bob"],
            debounce_ms=100,
            heartbeat_chat_id=42,
        ),
        agent_callback=mock_agent,
        api=api,
    )

    print(f"\n{banner}")
    print("  Chapter 12: Full Bot Demo")
    print(banner)

    await bot.start()

    # Simulate messages
    api.enqueue_update(_make_update(1, 42, "How's the deploy?", from_username="alice"))
    await asyncio.sleep(0.3)

    api.enqueue_update(_make_update(2, 42, "Need help", from_username="alice"))
    await asyncio.sleep(0.3)

    # Denied user
    api.enqueue_update(_make_update(3, 42, "hack the planet",
                                     from_id=99999, from_username="mallory"))
    await asyncio.sleep(0.3)

    # Rapid-fire (debounced)
    api.enqueue_update(_make_update(4, 42, "hey", from_username="bob"))
    api.enqueue_update(_make_update(5, 42, "can you", from_username="bob"))
    api.enqueue_update(_make_update(6, 42, "check logs?", from_username="bob"))
    await asyncio.sleep(0.3)

    # Heartbeat delivery
    print("\n  --- Heartbeat alert ---")
    await bot.deliver_heartbeat("Disk usage at 94%. Consider cleaning /tmp.")

    await bot.stop()

    # Stats
    print(f"\n  Stats:")
    print(f"    Received:  {bot._stats['received']}")
    print(f"    Processed: {bot._stats['processed']}")
    print(f"    Denied:    {bot._stats['denied']}")
    print(f"    Sent:      {bot._stats['sent']}")
    print(f"    API calls: {len(api.sent)}")
    print(banner)


# ╔════════════════════════════════════════════════════════════╗
# ║  APPENDIX A — Use-Case Gallery                           ║
# ╚════════════════════════════════════════════════════════════╝
#
# ── Use Case 1: Personal AI Assistant ──
#
#   bot = TelegramBot(
#       config=BotConfig(
#           token=os.environ["TELEGRAM_BOT_TOKEN"],
#           dm_policy=DmPolicy.ALLOWLIST,
#           allow_from=["@yourname"],
#           heartbeat_chat_id=YOUR_CHAT_ID,
#       ),
#       agent_callback=openai_chat,
#   )
#
# ── Use Case 2: DevOps Alert Bot ──
#
#   # Heartbeat checks every 15 min, alerts to a Telegram group
#   heartbeat = HeartbeatSystem(
#       llm_callback=check_infrastructure,
#       deliver_callback=lambda t, text: bot.deliver_heartbeat(text, t.chat_id),
#   )
#   heartbeat.configure([HeartbeatAgentConfig(
#       agent_id="infra-monitor",
#       interval_secs=900,
#   )])
#
# ── Use Case 3: Group Moderator with Mention Gating ──
#
#   # Bot only responds when mentioned (@mybot) in the group
#   config = BotConfig(
#       group_policy=GroupPolicy.ALLOWLIST,
#       group_allow_from=[-100123456],
#       # In production, add require_mention=True to group config
#   )
#
# ── Use Case 4: Multi-Account (Work + Personal) ──
#
#   work_bot = TelegramBot(
#       config=BotConfig(token=WORK_TOKEN, allow_from=["@colleague"]),
#       agent_callback=work_agent,
#   )
#   personal_bot = TelegramBot(
#       config=BotConfig(token=PERSONAL_TOKEN, allow_from=["@me"]),
#       agent_callback=personal_agent,
#   )
#   # Both run concurrently in the same event loop
#   await asyncio.gather(work_bot.start(), personal_bot.start())
#
# ── Use Case 5: Webhook Behind Nginx ──
#
#   # nginx config:
#   #   location /telegram-webhook {
#   #       proxy_pass http://127.0.0.1:8787;
#   #   }
#
#   server = WebhookServer(
#       path="/telegram-webhook",
#       secret="your-secret-here",
#       port=8787,
#   )
#   server.set_handler(bot._handle_update)
#   # Register webhook:
#   # await bot.api.set_webhook("https://yourdomain.com/telegram-webhook",
#   #                           secret_token="your-secret-here")
#
# ── Use Case 6: Streaming Responses with Draft Edits ──
#
#   async def streaming_agent(agent_id, prompt):
#       draft = DraftStream(bot.api, chat_id=42)
#       accumulated = ""
#       async for token in llm.stream(prompt):
#           accumulated += token
#           await draft.update(accumulated)
#       draft.stop()
#       return accumulated
#
#   bot = TelegramBot(config=..., agent_callback=streaming_agent)


# ╔════════════════════════════════════════════════════════════╗
# ║  MAIN — Run individual chapters or all                    ║
# ╚════════════════════════════════════════════════════════════╝

async def run_all():
    banner = "=" * 64

    chapters = [
        ("Chapter 1:  MVP: Receive and Reply",       chapter_1),
        ("Chapter 2:  Access Control",                chapter_2),
        ("Chapter 3:  Message Context Building",      chapter_3),
        ("Chapter 4:  Markdown → Telegram HTML",      chapter_4),
        ("Chapter 5:  Text Chunking",                 chapter_5),
        ("Chapter 6:  Target Parsing & Threading",    chapter_6),
        ("Chapter 7:  Inbound Debouncing",            chapter_7),
        ("Chapter 8:  Streaming Draft Edits",         chapter_8),
        ("Chapter 9:  Delivery Pipeline",             chapter_9),
        ("Chapter 10: Webhook Transport",             chapter_10),
        ("Chapter 11: Heartbeat Integration",         chapter_11),
        ("Chapter 12: Full Bot",                      chapter_12),
    ]

    print(banner)
    print("  Telegram Bot Tutorial — Running All Chapters")
    print(banner)

    for title, fn in chapters:
        print(f"\n{'─' * 64}")
        print(f"  {title}")
        print(f"{'─' * 64}")
        await fn()

    print(f"\n{banner}")
    print("  All chapters complete.")
    print("  See Appendix A (in the source) for real-world use-case patterns.")
    print(banner)


def main():
    logging.basicConfig(level=logging.WARNING)
    asyncio.run(run_all())


if __name__ == "__main__":
    main()
