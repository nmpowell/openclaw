#!/usr/bin/env python3
"""
========================================================================
  Building an AI Agent Heartbeat System in Python — From MVP to Full
========================================================================

A progressive tutorial that builds a complete heartbeat system from
scratch, modelled on OpenClaw's production TypeScript implementation.

Each "chapter" adds one layer of capability.  Every chapter is runnable
on its own — just scroll to __main__ and uncomment the chapter you want.

    python examples/heartbeat_tutorial.py

Source of truth (TypeScript):
    src/infra/heartbeat-runner.ts     — runner + scheduler
    src/infra/heartbeat-wake.ts       — wake coordinator
    src/auto-reply/heartbeat.ts       — prompt, token stripping, file checks
    src/infra/heartbeat-active-hours.ts — quiet-hours gating
    src/infra/heartbeat-visibility.ts — per-channel visibility
    src/infra/heartbeat-events.ts     — observable event bus
    src/infra/system-events.ts        — ephemeral system event queue

Table of contents:
    Chapter 1 — MVP: one timer, one callback                      (~30 lines)
    Chapter 2 — Task file gating (skip when nothing to do)        (+25 lines)
    Chapter 3 — LLM integration and response post-processing      (+60 lines)
    Chapter 4 — Deduplication (don't nag)                         (+20 lines)
    Chapter 5 — Multi-source wake with priority coalescing         (+80 lines)
    Chapter 6 — Multi-agent scheduling                            (+50 lines)
    Chapter 7 — Active-hours / quiet-hours                        (+30 lines)
    Chapter 8 — Delivery targets and visibility                   (+40 lines)
    Chapter 9 — Observable events                                 (+30 lines)
    Chapter 10 — System event queue (exec/cron wake-ups)          (+40 lines)
    Chapter 11 — Full implementation (everything together)        (~400 lines)
    Appendix A — Use-case gallery                                 (examples)
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Awaitable, Callable, Optional

log = logging.getLogger("heartbeat")


# ╔════════════════════════════════════════════════════════════╗
# ║  CHAPTER 1 — MVP: One Timer, One Callback                ║
# ╚════════════════════════════════════════════════════════════╝
#
# The simplest possible heartbeat: a single asyncio timer that
# fires a callback every N seconds.  No threads, no blocking.
#
# This is the moral equivalent of:
#     setTimeout(fn, delay); timer.unref();
# in Node.js.  The event loop is idle between fires.
#
# WHY asyncio and not time.sleep()?
#   time.sleep() blocks the calling thread.  If your program does
#   anything else (handle HTTP requests, process messages, etc.)
#   you'd need a dedicated thread just for sleeping.  asyncio
#   registers the wakeup in the event loop's timer heap — the
#   same thread can serve requests in between.

async def chapter_1():
    """MVP heartbeat: fire a callback every N seconds."""

    async def on_heartbeat():
        print(f"  [{time.strftime('%H:%M:%S')}] heartbeat fired!")

    async def run_forever(interval: float):
        while True:
            await asyncio.sleep(interval)
            await on_heartbeat()

    # In production you'd use interval=1800 (30 min).
    # Here we use 1s for demo speed.
    task = asyncio.create_task(run_forever(interval=1.0))
    await asyncio.sleep(3.5)
    task.cancel()

    # That's it.  Between fires, zero threads consumed, zero CPU.
    # But this is too simple:
    #   - No way to skip when there's nothing to check
    #   - No way to wake early (e.g. a background job finished)
    #   - Can't handle multiple agents at different intervals


# ╔════════════════════════════════════════════════════════════╗
# ║  CHAPTER 2 — Task File Gating                            ║
# ╚════════════════════════════════════════════════════════════╝
#
# An LLM API call costs money and latency.  If the user hasn't
# configured any tasks in their HEARTBEAT.md, skip the call
# entirely.
#
# OpenClaw's rule: a file is "effectively empty" if every line
# is blank, an ATX heading (# ...), or an empty checkbox (- [ ]).
# A *missing* file is NOT considered empty — let the LLM decide.

def is_task_file_empty(content: str | None) -> bool:
    """
    Returns True if the task file exists but has no actionable content.
    Returns False if the file is missing (None) — the LLM should still run.

    Mirrors: isHeartbeatContentEffectivelyEmpty() in heartbeat.ts
    """
    if content is None:
        return False                      # missing file → run anyway
    for line in content.split("\n"):
        s = line.strip()
        if not s:
            continue                      # blank
        if re.match(r"^#+(\s|$)", s):
            continue                      # heading
        if re.match(r"^[-*+]\s*(\[[\sXx]?\]\s*)?$", s):
            continue                      # empty checkbox
        return False                      # real content found
    return True                           # all lines are structural


async def chapter_2():
    """Skip the LLM call when HEARTBEAT.md has no tasks."""

    cases = [
        ("missing file",     None),
        ("empty headings",   "# Tasks\n\n- [ ]\n"),
        ("has real task",    "# Tasks\n- Deploy v2.3 to prod\n"),
    ]

    for label, content in cases:
        if content is not None and is_task_file_empty(content):
            print(f"  {label:20s} → SKIP (no API call)")
        else:
            print(f"  {label:20s} → RUN  (call LLM)")


# ╔════════════════════════════════════════════════════════════╗
# ║  CHAPTER 3 — LLM Integration & Response Post-Processing  ║
# ╚════════════════════════════════════════════════════════════╝
#
# The heartbeat prompt is injected as a user message into the
# agent's existing chat session.  The system prompt already has
# HEARTBEAT.md in its workspace context.
#
# The LLM responds with one of:
#   - "HEARTBEAT_OK"              → nothing to do, swallow it
#   - "HEARTBEAT_OK. All clear."  → short ack, swallow it
#   - "Your build is broken..."   → real alert, deliver it
#   - "HEARTBEAT_OK — also, ..."  → strip token, deliver the rest
#
# The token-stripping logic handles markup wrapping too:
#   **HEARTBEAT_OK**, <b>HEARTBEAT_OK</b>, etc.

HEARTBEAT_TOKEN = "HEARTBEAT_OK"
DEFAULT_ACK_MAX_CHARS = 300

DEFAULT_PROMPT = (
    "Read HEARTBEAT.md if it exists (workspace context). "
    "Follow it strictly. Do not infer or repeat old tasks from prior chats. "
    "If nothing needs attention, reply HEARTBEAT_OK."
)


@dataclass
class StrippedReply:
    """Result of stripping the HEARTBEAT_OK token from an LLM reply."""
    should_skip: bool        # True → swallow the reply, don't deliver
    text: str                # remaining text after stripping
    did_strip: bool          # True → token was found and removed


def strip_heartbeat_token(
    raw: str | None,
    *,
    mode: str = "heartbeat",
    max_ack_chars: int = DEFAULT_ACK_MAX_CHARS,
) -> StrippedReply:
    """
    Post-process an LLM reply to decide: deliver or swallow?

    Two modes:
      "heartbeat" — strict: short tails after the token are swallowed
      "message"   — lenient: only strip token, keep any remaining text

    Mirrors: stripHeartbeatToken() in heartbeat.ts
    """
    if not raw or not raw.strip():
        return StrippedReply(should_skip=True, text="", did_strip=False)

    text = raw.strip()

    # Strip lightweight markup so **HEARTBEAT_OK** or <b>HEARTBEAT_OK</b>
    # still matches.
    normalized = re.sub(r"<[^>]*>", " ", text)
    normalized = re.sub(r"&nbsp;", " ", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"^[*`~_]+", "", normalized)
    normalized = re.sub(r"[*`~_]+$", "", normalized)

    if HEARTBEAT_TOKEN not in text and HEARTBEAT_TOKEN not in normalized:
        return StrippedReply(should_skip=False, text=text, did_strip=False)

    # Try stripping from both raw and normalized; prefer whichever succeeds.
    # The TS code does stripTokenAtEdges() on both and picks the better one.
    def _strip_edges(s: str) -> tuple[str, bool]:
        did = False
        changed = True
        while changed:
            changed = False
            if s.startswith(HEARTBEAT_TOKEN):
                s = s[len(HEARTBEAT_TOKEN):].lstrip()
                did = changed = True
            if s.endswith(HEARTBEAT_TOKEN):
                s = s[:-len(HEARTBEAT_TOKEN)].rstrip()
                did = changed = True
        return " ".join(s.split()), did

    raw_stripped, raw_did = _strip_edges(text)
    norm_stripped, norm_did = _strip_edges(normalized)
    # Pick: prefer the raw result if it stripped something AND has text
    if raw_did and raw_stripped:
        work, did_strip = raw_stripped, raw_did
    else:
        work, did_strip = norm_stripped, norm_did

    if not did_strip:
        return StrippedReply(should_skip=False, text=text, did_strip=False)
    if not work:
        return StrippedReply(should_skip=True, text="", did_strip=True)

    # In heartbeat mode, short tails are considered ack-only.
    if mode == "heartbeat" and len(work) <= max_ack_chars:
        return StrippedReply(should_skip=True, text="", did_strip=True)

    return StrippedReply(should_skip=False, text=work, did_strip=True)
    did_strip = False
    changed = True
    while changed:
        changed = False
        if work.startswith(HEARTBEAT_TOKEN):
            work = work[len(HEARTBEAT_TOKEN):].lstrip()
            did_strip = changed = True
        if work.endswith(HEARTBEAT_TOKEN):
            work = work[:-len(HEARTBEAT_TOKEN)].rstrip()
            did_strip = changed = True

    work = " ".join(work.split())  # collapse whitespace

    if not did_strip:
        return StrippedReply(should_skip=False, text=text, did_strip=False)
    if not work:
        return StrippedReply(should_skip=True, text="", did_strip=True)

    # In heartbeat mode, short tails are considered ack-only.
    if mode == "heartbeat" and len(work) <= max_ack_chars:
        return StrippedReply(should_skip=True, text="", did_strip=True)

    return StrippedReply(should_skip=False, text=work, did_strip=True)


async def chapter_3():
    """Strip HEARTBEAT_OK and decide whether to deliver."""

    replies = [
        "HEARTBEAT_OK",
        "HEARTBEAT_OK. All systems nominal.",
        "**HEARTBEAT_OK**",
        "Your staging deploy is stuck. The CI pipeline timed out after 45m.",
        "HEARTBEAT_OK — also, disk at 92%.",
    ]

    for reply in replies:
        result = strip_heartbeat_token(reply, mode="heartbeat")
        action = "SKIP" if result.should_skip else f"DELIVER: {result.text!r}"
        print(f"  LLM said: {reply!r:55s} → {action}")


# ╔════════════════════════════════════════════════════════════╗
# ║  CHAPTER 4 — Deduplication                               ║
# ╚════════════════════════════════════════════════════════════╝
#
# If the LLM keeps saying "your deploy is stuck" every 30 min,
# the user gets nagged.  Solution: remember the last delivered
# text and suppress exact duplicates within 24 hours.
#
# OpenClaw stores this in the session entry:
#   lastHeartbeatText     — the text of the last sent alert
#   lastHeartbeatSentAt   — when it was sent (unix ms)

@dataclass
class DeduplicationState:
    """Tracks the last delivered heartbeat for duplicate suppression."""
    last_text: str = ""
    last_sent_at: float | None = None
    window_secs: float = 86400  # 24 hours

    def is_duplicate(self, text: str) -> bool:
        if not self.last_text.strip():
            return False
        if text.strip() != self.last_text.strip():
            return False
        if self.last_sent_at is None:
            return False
        return (time.time() - self.last_sent_at) < self.window_secs

    def record(self, text: str):
        self.last_text = text
        self.last_sent_at = time.time()


async def chapter_4():
    """Suppress duplicate alerts within 24h."""

    dedup = DeduplicationState()
    alerts = [
        "Deploy stuck on staging.",
        "Deploy stuck on staging.",       # duplicate — suppress
        "Build failed: test_auth.py",     # new alert — deliver
        "Build failed: test_auth.py",     # duplicate — suppress
    ]

    for alert in alerts:
        if dedup.is_duplicate(alert):
            print(f"  {alert!r:40s} → SUPPRESSED (duplicate)")
        else:
            print(f"  {alert!r:40s} → DELIVERED")
            dedup.record(alert)


# ╔════════════════════════════════════════════════════════════╗
# ║  CHAPTER 5 — Wake Coordinator (Priority Coalescing)      ║
# ╚════════════════════════════════════════════════════════════╝
#
# Multiple things can trigger a heartbeat:
#   - The interval timer (every 30m)
#   - An async exec finishing (your build completed)
#   - A cron reminder firing
#   - The user typing /heartbeat manually
#
# We need a single entry point — requestHeartbeatNow() — that:
#   1. Debounces: 3 requests in 250ms = 1 run
#   2. Prioritises: manual > exec/cron > interval > retry
#   3. Serialises: at most 1 heartbeat running at a time
#   4. Retries on failure with backoff
#
# This is the "mailbox" pattern.  It's safe without locks because
# asyncio is single-threaded within one event loop.
#
# Mirrors: heartbeat-wake.ts

REASON_PRIORITY: dict[str, int] = {
    "retry":      0,    # lowest — back-off re-attempt
    "interval":   1,    # normal timer fire
    "default":    2,    # fallback for unknown reasons
    "manual":     3,    # user typed /heartbeat
    "exec-event": 3,    # async command completed
    "cron":       3,    # scheduled reminder
}

DEFAULT_COALESCE_SECS = 0.25   # 250ms debounce window
DEFAULT_RETRY_SECS = 1.0       # retry backoff on failure


@dataclass
class PendingWake:
    """A queued wake request waiting for the debounce window to pass."""
    reason: str
    priority: int
    requested_at: float


class WakeCoordinator:
    """
    Singleton mailbox that serialises and coalesces heartbeat wake requests.

    Any part of the system calls request_now() to trigger a heartbeat.
    The coordinator debounces, picks the highest-priority reason, and
    calls the registered handler exactly once.

    Mirrors: requestHeartbeatNow() / schedule() / setHeartbeatWakeHandler()
    in heartbeat-wake.ts
    """

    def __init__(self):
        self._handler: Callable[[str], Awaitable[str]] | None = None
        self._pending: PendingWake | None = None
        self._running = False
        self._scheduled = False
        self._timer: asyncio.TimerHandle | None = None
        self._generation = 0

    # ── Public API ──

    def set_handler(self, handler: Callable[[str], Awaitable[str]] | None) -> Callable[[], None]:
        """
        Register (or clear) the wake handler.

        Returns a disposer.  Stale disposers from previous registrations
        are no-ops, preventing a race where an old runner's cleanup clears
        a newer runner's handler.

        Mirrors: setHeartbeatWakeHandler() in heartbeat-wake.ts
        """
        self._generation += 1
        gen = self._generation
        self._handler = handler
        if handler and self._pending:
            self._schedule(DEFAULT_COALESCE_SECS)

        def dispose():
            if self._generation != gen:
                return                    # stale disposer
            self._generation += 1
            self._handler = None

        return dispose

    def request_now(self, reason: str = "requested", coalesce_secs: float = DEFAULT_COALESCE_SECS):
        """
        Request a heartbeat run.  Multiple calls within the coalesce
        window collapse into a single run.  The highest-priority reason
        wins.

        Mirrors: requestHeartbeatNow() in heartbeat-wake.ts
        """
        self._queue_reason(reason)
        self._schedule(coalesce_secs)

    @property
    def has_pending(self) -> bool:
        return self._pending is not None or self._timer is not None or self._scheduled

    # ── Internals ──

    def _queue_reason(self, reason: str):
        priority = REASON_PRIORITY.get(reason, REASON_PRIORITY["default"])
        candidate = PendingWake(reason, priority, time.time())
        if self._pending is None or candidate.priority >= self._pending.priority:
            self._pending = candidate

    def _schedule(self, delay: float, kind: str = "normal"):
        if self._timer is not None:
            # Keep retry cooldowns as hard minimums.
            if kind == "retry":
                return
            # If existing timer fires sooner, keep it.
            return
        delay = max(0.0, delay)
        loop = asyncio.get_running_loop()
        self._timer = loop.call_later(
            delay,
            lambda: asyncio.ensure_future(self._fire(delay, kind)),
        )

    async def _fire(self, delay: float, kind: str):
        self._timer = None
        self._scheduled = False

        handler = self._handler
        if handler is None:
            return
        if self._running:
            # Another heartbeat is in progress.  Don't run in parallel —
            # just mark that we need to re-run when it finishes.
            self._scheduled = True
            return

        reason = self._pending.reason if self._pending else "requested"
        self._pending = None
        self._running = True
        try:
            result = await handler(reason)
            # If the main queue was busy, retry soon.
            if result == "requests-in-flight":
                self._queue_reason("retry")
                self._schedule(DEFAULT_RETRY_SECS, kind="retry")
        except Exception:
            # Error already logged by the handler; schedule a retry.
            log.exception("heartbeat handler failed")
            self._queue_reason("retry")
            self._schedule(DEFAULT_RETRY_SECS, kind="retry")
        finally:
            self._running = False
            if self._pending or self._scheduled:
                self._schedule(0)

    def cancel(self):
        if self._timer:
            self._timer.cancel()
            self._timer = None


async def chapter_5():
    """Demonstrate wake coalescing and priority."""

    results: list[str] = []

    async def handler(reason: str) -> str:
        results.append(reason)
        print(f"    handler called with reason={reason!r}")
        await asyncio.sleep(0.1)  # simulate LLM call
        return "ok"

    coord = WakeCoordinator()
    coord.set_handler(handler)

    # Fire 3 requests within the coalesce window — only 1 run should happen,
    # and the highest-priority reason should win.
    print("  Sending 3 requests within 250ms (interval, retry, exec-event)...")
    coord.request_now("interval")
    coord.request_now("retry")
    coord.request_now("exec-event")  # highest priority
    await asyncio.sleep(0.5)

    print(f"  Handler was called {len(results)} time(s), reason={results}")
    assert len(results) == 1
    assert results[0] == "exec-event"

    coord.cancel()


# ╔════════════════════════════════════════════════════════════╗
# ║  CHAPTER 6 — Multi-Agent Scheduling                      ║
# ╚════════════════════════════════════════════════════════════╝
#
# OpenClaw supports multiple agents, each with its own heartbeat
# interval.  The scheduler maintains a Map<agentId, state> and
# sets ONE timer pointing to the next due agent.
#
# After every run it calls _schedule_next():
#   1. Find the earliest nextDueTs across all agents
#   2. Cancel the existing timer
#   3. Set a new call_later(delay) for that agent
#
# This is O(agents) per reschedule — fine for 1-10 agents.
#
# Mirrors: startHeartbeatRunner() / scheduleNext() in heartbeat-runner.ts

@dataclass
class AgentHeartbeatState:
    """Per-agent scheduling state."""
    agent_id: str
    interval_secs: float
    last_run_at: float | None = None
    next_due_at: float = 0.0


class HeartbeatScheduler:
    """
    Multi-agent scheduler that uses a single timer.

    Lifecycle:
        startup → compute next_due_at for each agent
                → schedule_next() sets ONE timer for the earliest
                → timer fires → coordinator.request_now("interval")
                → coordinator calls handler → handler iterates agents
                → runs any that are due → advances their next_due_at
                → schedule_next() again

    Between fires: zero threads, zero CPU, just a timer handle
    in the event loop's heap.
    """

    def __init__(self, coordinator: WakeCoordinator):
        self.agents: dict[str, AgentHeartbeatState] = {}
        self.coordinator = coordinator
        self._timer: asyncio.TimerHandle | None = None
        self._stopped = False
        self._run_callback: Callable[[str, str], Awaitable[None]] | None = None

    def configure(
        self,
        agents: list[AgentHeartbeatState],
        run_callback: Callable[[str, str], Awaitable[None]],
    ):
        """
        Set (or update) the agent list and the function to call per agent.

        run_callback(agent_id, reason) is called for each agent that is due.
        This mirrors updateConfig() in heartbeat-runner.ts.
        """
        now = time.time()
        self._run_callback = run_callback
        new_agents: dict[str, AgentHeartbeatState] = {}
        for agent in agents:
            prev = self.agents.get(agent.agent_id)
            if prev and prev.last_run_at is not None:
                agent.next_due_at = prev.last_run_at + agent.interval_secs
            elif agent.next_due_at <= now:
                agent.next_due_at = now + agent.interval_secs
            new_agents[agent.agent_id] = agent
        self.agents = new_agents
        self._schedule_next()

    async def _handle_wake(self, reason: str) -> str:
        """Called by the WakeCoordinator."""
        if self._stopped or not self._run_callback:
            return "disabled"

        now = time.time()
        is_interval = reason == "interval"
        ran = False

        for agent in self.agents.values():
            if is_interval and now < agent.next_due_at:
                continue
            try:
                await self._run_callback(agent.agent_id, reason)
            except Exception:
                log.exception("heartbeat run failed for %s", agent.agent_id)
            agent.last_run_at = now
            agent.next_due_at = now + agent.interval_secs
            ran = True

        self._schedule_next()
        return "ran" if ran else "not-due"

    def _schedule_next(self):
        """Set ONE timer for the earliest due agent."""
        if self._stopped:
            return
        if self._timer:
            self._timer.cancel()
            self._timer = None
        if not self.agents:
            return

        now = time.time()
        earliest = min(a.next_due_at for a in self.agents.values())
        delay = max(0.0, earliest - now)

        loop = asyncio.get_running_loop()
        self._timer = loop.call_later(
            delay,
            lambda: self.coordinator.request_now("interval", coalesce_secs=0),
        )

    def start(self):
        self.coordinator.set_handler(self._handle_wake)
        self._schedule_next()

    def stop(self):
        self._stopped = True
        if self._timer:
            self._timer.cancel()
        self.coordinator.cancel()


async def chapter_6():
    """Two agents at different intervals sharing one timer."""

    coord = WakeCoordinator()
    scheduler = HeartbeatScheduler(coord)

    t0 = time.time()
    runs: list[str] = []

    async def on_run(agent_id: str, reason: str):
        elapsed = time.time() - t0
        msg = f"  t={elapsed:.1f}s  {agent_id} fired (reason={reason})"
        print(msg)
        runs.append(agent_id)

    scheduler.configure([
        AgentHeartbeatState("fast-agent", interval_secs=1.0),
        AgentHeartbeatState("slow-agent", interval_secs=2.5),
    ], run_callback=on_run)
    scheduler.start()

    await asyncio.sleep(5.5)
    scheduler.stop()
    print(f"  fast-agent fired {runs.count('fast-agent')}x, "
          f"slow-agent fired {runs.count('slow-agent')}x")


# ╔════════════════════════════════════════════════════════════╗
# ║  CHAPTER 7 — Active Hours / Quiet Hours                  ║
# ╚════════════════════════════════════════════════════════════╝
#
# Nobody wants a heartbeat alert at 3 AM.  OpenClaw supports an
# activeHours window: heartbeats run only between start and end
# times, in the user's timezone.
#
# Handles wraparound (e.g. 22:00–06:00 = overnight window).
# If either time is invalid, the guard passes (fail-open).
#
# Mirrors: isWithinActiveHours() in heartbeat-active-hours.ts

@dataclass
class ActiveHoursConfig:
    """Active hours window in HH:MM format."""
    start: str = "08:00"     # inclusive
    end: str = "22:00"       # exclusive
    timezone: str = "UTC"


def _parse_hhmm(raw: str) -> int | None:
    """Parse "HH:MM" to minutes since midnight.  Returns None on bad input."""
    m = re.match(r"^([01]\d|2[0-3]|24):([0-5]\d)$", raw)
    if not m:
        return None
    h, mi = int(m.group(1)), int(m.group(2))
    if h == 24 and mi != 0:
        return None
    return h * 60 + mi


def is_within_active_hours(
    config: ActiveHoursConfig | None,
    now: datetime | None = None,
) -> bool:
    """
    Returns True if the current time falls within the active window.
    Returns True (fail-open) if config is None or times are invalid.

    Handles wraparound: start=22:00, end=06:00 means overnight.

    Mirrors: isWithinActiveHours() in heartbeat-active-hours.ts
    """
    if config is None:
        return True

    start_min = _parse_hhmm(config.start)
    end_min = _parse_hhmm(config.end)
    if start_min is None or end_min is None or start_min == end_min:
        return True  # fail-open on bad config

    if now is None:
        now = datetime.now(timezone.utc)

    # Get current time in the configured timezone.
    # Try zoneinfo first (requires tzdata on some platforms), fall back to UTC.
    local = now
    try:
        import zoneinfo
        tz = zoneinfo.ZoneInfo(config.timezone)
        local = now.astimezone(tz)
    except Exception:
        try:
            # Manual UTC-offset fallback for environments without tzdata.
            from datetime import timedelta
            _OFFSET_MAP = {
                "US/Eastern": -5, "US/Central": -6, "US/Mountain": -7,
                "US/Pacific": -8, "Europe/London": 0, "Europe/Berlin": 1,
                "Asia/Tokyo": 9, "UTC": 0,
            }
            offset_h = _OFFSET_MAP.get(config.timezone, 0)
            local = now.astimezone(timezone(timedelta(hours=offset_h)))
        except Exception:
            pass  # already set to UTC above

    current_min = local.hour * 60 + local.minute

    if end_min > start_min:
        # Normal range: e.g. 08:00–22:00
        return start_min <= current_min < end_min
    else:
        # Wraparound: e.g. 22:00–06:00
        return current_min >= start_min or current_min < end_min


async def chapter_7():
    """Active-hours gating."""

    from datetime import datetime, timedelta, timezone

    config = ActiveHoursConfig(start="09:00", end="17:00", timezone="US/Eastern")

    # US/Eastern is UTC-5 (ignoring DST for simplicity)
    et_tz = timezone(timedelta(hours=-5))

    test_times = [
        datetime(2025, 6, 15, 14, 30, tzinfo=timezone.utc),  # 09:30 ET → active
        datetime(2025, 6, 15, 1, 0, tzinfo=timezone.utc),    # 20:00 ET prev day → quiet
        datetime(2025, 6, 15, 14, 0, tzinfo=timezone.utc),   # 09:00 ET → active
        datetime(2025, 6, 15, 22, 0, tzinfo=timezone.utc),   # 17:00 ET → quiet
    ]

    for t in test_times:
        et = t.astimezone(et_tz)
        active = is_within_active_hours(config, t)
        status = "ACTIVE" if active else "QUIET"
        print(f"  {et.strftime('%H:%M ET')}  →  {status}")

    # Wraparound example: overnight window
    overnight = ActiveHoursConfig(start="22:00", end="06:00")
    print()
    for hour in [21, 23, 2, 6, 12]:
        t = datetime(2025, 6, 15, hour, 0, tzinfo=timezone.utc)
        active = is_within_active_hours(overnight, t)
        status = "ACTIVE" if active else "QUIET"
        print(f"  {hour:02d}:00 UTC (window 22:00–06:00)  →  {status}")


# ╔════════════════════════════════════════════════════════════╗
# ║  CHAPTER 8 — Delivery Targets & Visibility               ║
# ╚════════════════════════════════════════════════════════════╝
#
# A heartbeat result needs to go somewhere: WhatsApp, Discord,
# Telegram, Slack, etc.  OpenClaw resolves the target from the
# session's last-seen channel (target="last") or an explicit
# channel id (target="telegram").
#
# Visibility controls whether different result types are shown:
#   showOk     — deliver "HEARTBEAT_OK" messages (default: False)
#   showAlerts — deliver real content (default: True)
#   useIndicator — emit UI status indicators (default: True)
#
# These are layered: per-account > per-channel > channel-defaults > global.
#
# Mirrors: heartbeat-visibility.ts, outbound/targets.ts

class Channel(Enum):
    NONE = "none"
    WHATSAPP = "whatsapp"
    DISCORD = "discord"
    TELEGRAM = "telegram"
    SLACK = "slack"
    SIGNAL = "signal"
    WEBCHAT = "webchat"


@dataclass
class DeliveryTarget:
    """Where to send heartbeat results."""
    channel: Channel = Channel.NONE
    to: str | None = None
    account_id: str | None = None
    reason: str | None = None          # why this target was chosen


@dataclass
class HeartbeatVisibility:
    """Controls what gets delivered vs silenced."""
    show_ok: bool = False              # deliver "all clear" messages?
    show_alerts: bool = True           # deliver real content?
    use_indicator: bool = True         # emit UI status events?


def resolve_visibility(
    channel: Channel,
    *,
    channel_config: dict[str, Any] | None = None,
    account_config: dict[str, Any] | None = None,
    defaults: dict[str, Any] | None = None,
) -> HeartbeatVisibility:
    """
    Resolve visibility with 3-layer precedence:
    per-account > per-channel > defaults.

    Mirrors: resolveHeartbeatVisibility() in heartbeat-visibility.ts
    """
    def pick(key: str, default: bool) -> bool:
        if account_config and key in account_config:
            return account_config[key]
        if channel_config and key in channel_config:
            return channel_config[key]
        if defaults and key in defaults:
            return defaults[key]
        return default

    return HeartbeatVisibility(
        show_ok=pick("show_ok", False),
        show_alerts=pick("show_alerts", True),
        use_indicator=pick("use_indicator", True),
    )


async def chapter_8():
    """Delivery target resolution and visibility."""

    # Simulate: last message was on WhatsApp
    target = DeliveryTarget(Channel.WHATSAPP, to="+1234567890")

    # Default visibility: suppress OK, show alerts
    vis = resolve_visibility(Channel.WHATSAPP)
    print(f"  Default:      show_ok={vis.show_ok}, show_alerts={vis.show_alerts}")

    # Per-channel override: also show OK
    vis2 = resolve_visibility(
        Channel.WHATSAPP,
        channel_config={"show_ok": True},
    )
    print(f"  Channel cfg:  show_ok={vis2.show_ok}, show_alerts={vis2.show_alerts}")

    # Per-account override: suppress everything
    vis3 = resolve_visibility(
        Channel.WHATSAPP,
        channel_config={"show_ok": True},
        account_config={"show_alerts": False},
    )
    print(f"  Account cfg:  show_ok={vis3.show_ok}, show_alerts={vis3.show_alerts}")


# ╔════════════════════════════════════════════════════════════╗
# ║  CHAPTER 9 — Observable Events                           ║
# ╚════════════════════════════════════════════════════════════╝
#
# Every heartbeat emits a structured event for observability:
# status, duration, channel, preview text, indicator type.
#
# Consumers (UI, logging, monitoring) subscribe via callbacks.
#
# Mirrors: heartbeat-events.ts

@dataclass
class HeartbeatEvent:
    """Structured event emitted after every heartbeat cycle."""
    ts: float
    status: str                # "sent" | "ok-empty" | "ok-token" | "skipped" | "failed"
    duration_secs: float = 0.0
    reason: str | None = None
    channel: str | None = None
    to: str | None = None
    preview: str | None = None
    silent: bool = False
    indicator: str | None = None   # "ok" | "alert" | "error"


class HeartbeatEventBus:
    """Simple observable for heartbeat events."""

    def __init__(self):
        self._listeners: list[Callable[[HeartbeatEvent], None]] = []
        self._last: HeartbeatEvent | None = None

    def subscribe(self, fn: Callable[[HeartbeatEvent], None]) -> Callable[[], None]:
        self._listeners.append(fn)
        return lambda: self._listeners.remove(fn)

    def emit(self, **kwargs):
        evt = HeartbeatEvent(ts=time.time(), **kwargs)
        self._last = evt
        for fn in self._listeners:
            try:
                fn(evt)
            except Exception:
                pass  # don't let a bad listener break the heartbeat

    @property
    def last(self) -> HeartbeatEvent | None:
        return self._last


async def chapter_9():
    """Event bus for observability."""

    bus = HeartbeatEventBus()

    received: list[HeartbeatEvent] = []
    bus.subscribe(lambda evt: received.append(evt))

    bus.emit(status="ok-token", reason="interval", indicator="ok")
    bus.emit(status="sent", channel="whatsapp", preview="Deploy stuck...", indicator="alert")
    bus.emit(status="failed", reason="API timeout", indicator="error")

    for evt in received:
        print(f"  {evt.status:10s}  indicator={evt.indicator!s:6s}  "
              f"preview={evt.preview or '—'}")


# ╔════════════════════════════════════════════════════════════╗
# ║  CHAPTER 10 — System Event Queue (Exec/Cron Wake-Ups)    ║
# ╚════════════════════════════════════════════════════════════╝
#
# When an async exec completes or a cron reminder fires, the
# event is pushed onto a per-session in-memory queue.  The next
# heartbeat run peeks at the queue and uses a specialised prompt
# instead of the default "read HEARTBEAT.md" prompt.
#
# Events are ephemeral (not persisted) and session-scoped.
# Consecutive duplicates are suppressed.
#
# Mirrors: system-events.ts

EXEC_EVENT_PROMPT = (
    "An async command you ran earlier has completed. The result is "
    "shown in the system messages above. Please relay the command "
    "output to the user in a helpful way."
)


def build_cron_event_prompt(events: list[str]) -> str:
    body = "\n".join(events).strip()
    if not body:
        return "A scheduled cron event triggered with no content. Reply HEARTBEAT_OK."
    return (
        "A scheduled reminder has been triggered. The reminder content is:\n\n"
        + body
        + "\n\nPlease relay this reminder to the user in a helpful and friendly way."
    )


class SystemEventQueue:
    """
    Per-session in-memory queue for ephemeral system events.

    Events are drained (consumed) when the heartbeat reads them,
    so they're delivered exactly once.

    Mirrors: system-events.ts
    """

    def __init__(self, max_events: int = 20):
        self._queues: dict[str, list[str]] = {}
        self._last_text: dict[str, str] = {}
        self._max = max_events

    def enqueue(self, session_key: str, text: str):
        """Push an event.  Skips consecutive duplicates."""
        text = text.strip()
        if not text:
            return
        if self._last_text.get(session_key) == text:
            return  # consecutive duplicate
        self._last_text[session_key] = text
        queue = self._queues.setdefault(session_key, [])
        queue.append(text)
        if len(queue) > self._max:
            queue.pop(0)

    def peek(self, session_key: str) -> list[str]:
        """Read events without consuming them."""
        return list(self._queues.get(session_key, []))

    def drain(self, session_key: str) -> list[str]:
        """Read and consume all events for a session."""
        events = self._queues.pop(session_key, [])
        self._last_text.pop(session_key, None)
        return events

    def has_events(self, session_key: str) -> bool:
        return len(self._queues.get(session_key, [])) > 0


def resolve_heartbeat_prompt(
    reason: str,
    pending_events: list[str],
    default_prompt: str = DEFAULT_PROMPT,
) -> str:
    """
    Choose the right prompt based on the wake reason and pending events.

    Mirrors: the prompt-selection logic in runHeartbeatOnce() in heartbeat-runner.ts
    """
    has_exec = any("exec finished" in e.lower() for e in pending_events)
    is_cron = reason.startswith("cron:")
    cron_events = [
        e for e in pending_events
        if e.strip()
        and "heartbeat" not in e.lower()
        and "exec finished" not in e.lower()
    ]

    if has_exec:
        return EXEC_EVENT_PROMPT
    if is_cron and cron_events:
        return build_cron_event_prompt(cron_events)
    return default_prompt


async def chapter_10():
    """System event queue and prompt routing."""

    queue = SystemEventQueue()

    # Scenario 1: async exec completes
    queue.enqueue("session-1", "exec finished: npm run build (exit 0)")
    events = queue.peek("session-1")
    prompt = resolve_heartbeat_prompt("exec-event", events)
    print(f"  Exec event prompt:\n    {prompt[:80]}...")
    queue.drain("session-1")

    # Scenario 2: cron reminder
    queue.enqueue("session-1", "Reminder: standup in 10 minutes")
    events = queue.peek("session-1")
    prompt = resolve_heartbeat_prompt("cron:standup", events)
    print(f"\n  Cron event prompt:\n    {prompt[:80]}...")
    queue.drain("session-1")

    # Scenario 3: normal heartbeat (no events)
    prompt = resolve_heartbeat_prompt("interval", [])
    print(f"\n  Default prompt:\n    {prompt[:80]}...")

    # Consecutive duplicates suppressed
    queue.enqueue("session-2", "exec finished: test")
    queue.enqueue("session-2", "exec finished: test")  # suppressed
    queue.enqueue("session-2", "exec finished: lint")   # different → kept
    assert len(queue.peek("session-2")) == 2
    print("\n  Consecutive duplicate suppression: OK")


# ╔════════════════════════════════════════════════════════════╗
# ║  CHAPTER 11 — Full Implementation                        ║
# ╚════════════════════════════════════════════════════════════╝
#
# This brings everything together into a single HeartbeatSystem
# class that you could drop into any Python project.
#
# It composes all the pieces from chapters 1-10:
#   - WakeCoordinator for serialisation + coalescing
#   - HeartbeatScheduler for multi-agent timing
#   - Task file gating (skip when empty)
#   - LLM call + response post-processing
#   - Deduplication
#   - Active-hours gating
#   - Delivery target resolution + visibility
#   - Observable events
#   - System event queue for exec/cron wake-ups

@dataclass
class HeartbeatAgentConfig:
    """Full configuration for one agent's heartbeat."""
    agent_id: str
    interval_secs: float = 1800             # 30 minutes
    prompt: str | None = None               # custom prompt override
    ack_max_chars: int = DEFAULT_ACK_MAX_CHARS
    active_hours: ActiveHoursConfig | None = None
    delivery: DeliveryTarget = field(default_factory=DeliveryTarget)
    visibility: HeartbeatVisibility = field(default_factory=HeartbeatVisibility)


# The LLM callback type: takes a prompt string, returns the LLM's reply.
LLMCallback = Callable[[str, str], Awaitable[str]]

# The delivery callback type: sends text to a channel.
DeliverCallback = Callable[[DeliveryTarget, str], Awaitable[None]]


class HeartbeatSystem:
    """
    Complete heartbeat system ready for integration.

    Usage:
        system = HeartbeatSystem(
            llm_callback=my_llm_fn,
            deliver_callback=my_send_fn,
        )
        system.configure([
            HeartbeatAgentConfig(
                agent_id="my-agent",
                interval_secs=1800,
                delivery=DeliveryTarget(Channel.SLACK, to="#alerts"),
            ),
        ])
        system.start()
        ...
        system.stop()

    External wake-ups:
        system.request_wake("exec-event")
        system.request_wake("cron:daily-standup")

    Observability:
        system.events.subscribe(lambda evt: log.info(evt))
    """

    def __init__(
        self,
        llm_callback: LLMCallback,
        deliver_callback: DeliverCallback,
        read_task_file: Callable[[str], Awaitable[str | None]] | None = None,
    ):
        self._llm = llm_callback
        self._deliver = deliver_callback
        self._read_task_file = read_task_file

        self.events = HeartbeatEventBus()
        self.system_events = SystemEventQueue()

        self._coordinator = WakeCoordinator()
        self._scheduler = HeartbeatScheduler(self._coordinator)
        self._configs: dict[str, HeartbeatAgentConfig] = {}
        self._dedup: dict[str, DeduplicationState] = {}

    def configure(self, agents: list[HeartbeatAgentConfig]):
        """Set the list of agents and their heartbeat configs."""
        self._configs = {a.agent_id: a for a in agents}
        self._scheduler.configure(
            [AgentHeartbeatState(a.agent_id, a.interval_secs) for a in agents],
            run_callback=self._run_agent,
        )

    def start(self):
        """Start the scheduler.  Call this after configure()."""
        self._scheduler.start()

    def stop(self):
        """Stop all timers and clean up."""
        self._scheduler.stop()

    def request_wake(self, reason: str = "manual"):
        """Trigger an immediate heartbeat from outside the timer."""
        self._coordinator.request_now(reason, coalesce_secs=0.05)

    async def _run_agent(self, agent_id: str, reason: str):
        """
        Run a single heartbeat cycle for one agent.
        This is the core pipeline — see the lifecycle diagram in chapter 3
        of heartbeat_explainer.py.
        """
        config = self._configs.get(agent_id)
        if not config:
            return
        started = time.time()

        # ── Gate: active hours ──
        if not is_within_active_hours(config.active_hours):
            self.events.emit(status="skipped", reason="quiet-hours")
            return

        # ── Gate: task file empty ──
        is_exec = reason == "exec-event"
        is_cron = reason.startswith("cron:")
        task_content: str | None = None
        if self._read_task_file:
            task_content = await self._read_task_file(agent_id)

        if (
            task_content is not None
            and is_task_file_empty(task_content)
            and not is_exec
            and not is_cron
        ):
            self.events.emit(status="skipped", reason="empty-task-file")
            return

        # ── Resolve prompt ──
        pending = self.system_events.peek(agent_id)
        prompt = resolve_heartbeat_prompt(
            reason, pending, config.prompt or DEFAULT_PROMPT,
        )
        # Drain events now that we've used them
        if pending:
            self.system_events.drain(agent_id)

        # ── Call LLM ──
        try:
            reply = await self._llm(agent_id, prompt)
        except Exception as e:
            self.events.emit(
                status="failed",
                reason=str(e),
                duration_secs=time.time() - started,
                indicator="error",
            )
            return

        # ── Post-process ──
        stripped = strip_heartbeat_token(
            reply, mode="heartbeat", max_ack_chars=config.ack_max_chars,
        )

        if stripped.should_skip:
            indicator_status = "ok-token" if stripped.did_strip else "ok-empty"
            self.events.emit(
                status=indicator_status,
                reason=reason,
                duration_secs=time.time() - started,
                indicator="ok",
            )
            return

        final_text = stripped.text

        # ── Deduplicate ──
        dedup = self._dedup.setdefault(agent_id, DeduplicationState())
        if dedup.is_duplicate(final_text):
            self.events.emit(
                status="skipped",
                reason="duplicate",
                preview=final_text[:200],
                duration_secs=time.time() - started,
            )
            return

        # ── Visibility gate ──
        if not config.visibility.show_alerts:
            self.events.emit(
                status="skipped",
                reason="alerts-disabled",
                preview=final_text[:200],
                duration_secs=time.time() - started,
            )
            return

        # ── Deliver ──
        target = config.delivery
        if target.channel == Channel.NONE or not target.to:
            self.events.emit(
                status="skipped",
                reason="no-target",
                preview=final_text[:200],
                duration_secs=time.time() - started,
            )
            return

        try:
            await self._deliver(target, final_text)
            dedup.record(final_text)
            self.events.emit(
                status="sent",
                to=target.to,
                channel=target.channel.value,
                preview=final_text[:200],
                duration_secs=time.time() - started,
                indicator="alert",
            )
        except Exception as e:
            self.events.emit(
                status="failed",
                reason=str(e),
                duration_secs=time.time() - started,
                indicator="error",
            )


async def chapter_11():
    """Full system integration demo."""

    banner = "=" * 64

    # ── Mock LLM ──
    call_count = 0
    async def mock_llm(agent_id: str, prompt: str) -> str:
        nonlocal call_count
        call_count += 1
        if "async command" in prompt.lower():
            return "Your build succeeded. All 142 tests passed."
        if "reminder" in prompt.lower():
            return "Reminder: standup in 10 minutes."
        if call_count % 3 == 0:
            return "Disk usage at 94%. Consider cleaning /tmp."
        return HEARTBEAT_TOKEN

    # ── Mock delivery ──
    deliveries: list[tuple[str, str]] = []
    async def mock_deliver(target: DeliveryTarget, text: str):
        deliveries.append((target.to or "?", text))
        print(f"    DELIVERED to {target.channel.value}:{target.to}: {text!r}")

    # ── Mock task file ──
    async def mock_read_task_file(agent_id: str) -> str | None:
        return "# Tasks\n- Monitor disk usage\n- Check build status\n"

    # ── Build system ──
    system = HeartbeatSystem(
        llm_callback=mock_llm,
        deliver_callback=mock_deliver,
        read_task_file=mock_read_task_file,
    )

    # ── Subscribe to events ──
    events: list[HeartbeatEvent] = []
    system.events.subscribe(lambda e: events.append(e))

    # ── Configure agents ──
    system.configure([
        HeartbeatAgentConfig(
            agent_id="main",
            interval_secs=1.0,  # 1s for demo (normally 1800)
            delivery=DeliveryTarget(Channel.WHATSAPP, to="+1234567890"),
            visibility=HeartbeatVisibility(show_alerts=True),
        ),
    ])

    print(f"\n{banner}")
    print("  Chapter 11: Full System Demo (1s intervals)")
    print(banner)

    t0 = time.time()
    system.start()

    # Let 3 interval heartbeats fire
    await asyncio.sleep(3.5)

    # Inject an exec-event
    elapsed = time.time() - t0
    print(f"\n  t={elapsed:.1f}s: injecting exec-event...")
    system.system_events.enqueue("main", "exec finished: npm run build")
    system.request_wake("exec-event")
    await asyncio.sleep(0.5)

    # Inject a cron reminder
    elapsed = time.time() - t0
    print(f"\n  t={elapsed:.1f}s: injecting cron reminder...")
    system.system_events.enqueue("main", "Reminder: team standup in 10 minutes")
    system.request_wake("cron:standup")
    await asyncio.sleep(0.5)

    system.stop()

    # ── Summary ──
    print(f"\n{banner}")
    print("  Summary")
    print(banner)
    print(f"  Total events:     {len(events)}")
    print(f"  Deliveries:       {len(deliveries)}")
    for evt in events:
        indicator = evt.indicator or "—"
        print(f"    {evt.status:12s}  indicator={indicator:6s}  "
              f"reason={evt.reason or '—':15s}  "
              f"preview={evt.preview or '—'}")
    print(banner)


# ╔════════════════════════════════════════════════════════════╗
# ║  APPENDIX A — Use-Case Gallery                           ║
# ╚════════════════════════════════════════════════════════════╝
#
# Here are concrete examples of how you might use the heartbeat
# system in different Python projects.  These are not runnable
# (they reference fictional libraries) but show the integration
# patterns.
#
# ── Use Case 1: DevOps Monitoring Bot ──
#
#   Check deployment health every 15 minutes.  Alert on Slack
#   if any pods are crashlooping.
#
#     async def check_k8s(agent_id, prompt):
#         pods = await k8s_client.list_pods(namespace="prod")
#         crashlooping = [p for p in pods if p.restart_count > 5]
#         if crashlooping:
#             return f"ALERT: {len(crashlooping)} pods crashlooping: " + \
#                    ", ".join(p.name for p in crashlooping)
#         return "HEARTBEAT_OK"
#
#     system = HeartbeatSystem(llm_callback=check_k8s, deliver_callback=slack_send)
#     system.configure([HeartbeatAgentConfig(
#         agent_id="k8s-monitor",
#         interval_secs=900,
#         delivery=DeliveryTarget(Channel.SLACK, to="#ops-alerts"),
#     )])
#
# ── Use Case 2: Personal AI Assistant ──
#
#   An agent that reads your HEARTBEAT.md task list every 30 min,
#   uses an LLM to check whether each task needs attention, and
#   WhatsApps you only if something is actionable.
#
#     async def llm_check(agent_id, prompt):
#         return await openai.chat(model="gpt-4", messages=[
#             {"role": "system", "content": system_prompt_with_heartbeat_md},
#             {"role": "user",   "content": prompt},
#         ])
#
#     system.configure([HeartbeatAgentConfig(
#         agent_id="assistant",
#         interval_secs=1800,
#         active_hours=ActiveHoursConfig("08:00", "22:00", "US/Pacific"),
#         delivery=DeliveryTarget(Channel.WHATSAPP, to="+1555..."),
#     )])
#
# ── Use Case 3: Multi-Agent SaaS Platform ──
#
#   Each customer has their own agent with independent heartbeats.
#   Some run every 5 min, some every hour.  The scheduler handles
#   all of them with a single timer.
#
#     agents = []
#     for customer in customers:
#         agents.append(HeartbeatAgentConfig(
#             agent_id=f"agent-{customer.id}",
#             interval_secs=customer.heartbeat_interval,
#             delivery=DeliveryTarget(
#                 Channel.TELEGRAM,
#                 to=customer.telegram_chat_id,
#             ),
#         ))
#     system.configure(agents)
#
# ── Use Case 4: Background Job Completion Relay ──
#
#   A long-running job (ML training, data pipeline) finishes.
#   Instead of polling, the job pushes an event and requests
#   an immediate wake.  The heartbeat system relays the result.
#
#     # In your job completion handler:
#     system.system_events.enqueue("agent-1", "exec finished: train_model.py (loss=0.023)")
#     system.request_wake("exec-event")
#
# ── Use Case 5: Cron-Style Scheduled Reminders ──
#
#   A cron service fires periodic reminders.  The heartbeat system
#   picks them up and has the LLM format a friendly message.
#
#     # Your cron scheduler calls:
#     system.system_events.enqueue("agent-1", "Reminder: weekly report due Friday")
#     system.request_wake("cron:weekly-report")


# ╔════════════════════════════════════════════════════════════╗
# ║  MAIN — Run individual chapters or the full system        ║
# ╚════════════════════════════════════════════════════════════╝

async def run_all():
    banner = "=" * 64

    chapters = [
        ("Chapter 1: MVP Timer",                      chapter_1),
        ("Chapter 2: Task File Gating",               chapter_2),
        ("Chapter 3: LLM Response Post-Processing",   chapter_3),
        ("Chapter 4: Deduplication",                   chapter_4),
        ("Chapter 5: Wake Coordinator (Coalescing)",   chapter_5),
        ("Chapter 6: Multi-Agent Scheduling",          chapter_6),
        ("Chapter 7: Active Hours / Quiet Hours",      chapter_7),
        ("Chapter 8: Delivery Targets & Visibility",   chapter_8),
        ("Chapter 9: Observable Events",               chapter_9),
        ("Chapter 10: System Event Queue",             chapter_10),
        ("Chapter 11: Full Implementation",            chapter_11),
    ]

    print(banner)
    print("  Heartbeat Tutorial — Running All Chapters")
    print(banner)

    for title, fn in chapters:
        print(f"\n{'─' * 64}")
        print(f"  {title}")
        print(f"{'─' * 64}")
        await fn()

    print(f"\n{banner}")
    print("  All chapters complete.")
    print(f"  See Appendix A (in the source) for real-world use-case patterns.")
    print(banner)


def main():
    logging.basicConfig(level=logging.WARNING)
    asyncio.run(run_all())


if __name__ == "__main__":
    main()
