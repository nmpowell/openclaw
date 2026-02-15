"""
OpenClaw Heartbeat System — Simplified Python Mock-up
=====================================================

The heartbeat is a **periodic background poller** that wakes the AI agent on a
timer (default: every 30 minutes) and asks it to check a task file
(HEARTBEAT.md).  The agent either responds "HEARTBEAT_OK" (nothing to do) or
produces a real reply that gets delivered to the user over their messaging
channel (WhatsApp, Discord, etc.).

Key design goals:
  1. Let the agent notice things *without* the user sending a message first.
  2. Avoid wasting API calls when nothing is configured (empty HEARTBEAT.md).
  3. Suppress duplicate alerts — don't nag about the same thing twice.
  4. Support exec-event and cron-event wake-ups that override the prompt.
  5. Respect active-hours and quiet-hours so the user isn't pinged at 3 AM.

## Timer Architecture (the interesting part)
============================================

The TypeScript codebase uses **two cooperating layers**, neither of which
blocks a thread:

### Layer 1 — The Scheduler (`startHeartbeatRunner` in heartbeat-runner.ts)
  Maintains a Map of per-agent states (agentId → {intervalMs, nextDueMs, ...}).
  After every run (or config change) it calls `scheduleNext()`:

      scheduleNext():
          find the earliest nextDueMs across all agents
          clearTimeout(existing timer)
          setTimeout(callback, delay)   ← THE ONLY TIMER
          timer.unref()                 ← lets Node exit even if timer is pending

  When the timer fires, it calls `requestHeartbeatNow({ reason: "interval" })`.
  That's it — one setTimeout at a time, always pointing to the next due agent.
  This is O(agents) per reschedule, which is fine because you have ~1-10 agents.

### Layer 2 — The Wake Coordinator (`heartbeat-wake.ts`)
  A singleton "mailbox" that coalesces wake requests from multiple sources:

      requestHeartbeatNow(reason, coalesceMs=250)
        ↓
      queuePendingWakeReason(reason)   ← priority queue: manual > cron > interval > retry
        ↓
      schedule(coalesceMs)             ← short setTimeout for debounce
        ↓
      [timer fires]
        ↓
      if (running) → reschedule later  ← mutual exclusion, no lock needed
      else → handler({ reason })       ← calls back into the runner's run()

  This layer exists so that *any* part of the system (exec completion, cron
  service, manual /heartbeat command, the interval timer) can request a
  heartbeat via the same function, and:
    - Requests are debounced (default 250ms coalesce window)
    - Only one heartbeat runs at a time (no parallelism / no locks)
    - The highest-priority reason wins if multiple arrive in the window
    - Retry on transient failures with 1s backoff

### Why this works well in Node.js
  Node's event loop is single-threaded. setTimeout doesn't consume a thread —
  it registers a callback with libuv's timer heap. Between heartbeats (30
  minutes!) the process does other work (handling messages, running tools,
  etc.) with zero overhead from the heartbeat system.

  `timer.unref()` is the cherry on top: it tells Node "don't keep the process
  alive just for this timer". If everything else finishes, Node exits cleanly.

### Python equivalents
  Python has no built-in event loop in the same sense, but there are several
  good options (see Section 8 below for runnable examples):

  1. **asyncio** (recommended) — `asyncio.get_event_loop().call_later(delay, cb)`
     or `await asyncio.sleep(delay)` in a task. Closest to the Node model.
     Single-threaded, cooperative, zero extra threads.

  2. **threading.Timer** — `threading.Timer(delay, fn).start()` creates a
     real OS thread per timer. Fine for one heartbeat but wasteful at scale.
     Does NOT block the main thread.

  3. **sched.scheduler** — stdlib module, but you must call `scheduler.run()`
     which blocks. Only useful if you dedicate a thread to it.

  4. **APScheduler / Celery Beat** — production job schedulers with cron
     syntax, persistent stores, etc. Overkill for a single heartbeat loop
     but appropriate for a real deployment.

Run this file to see the full lifecycle printed step by step:

    python examples/heartbeat_explainer.py
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ────────────────────────────────────────────────────────────
# 1. CONSTANTS  (mirrors src/auto-reply/tokens.ts & heartbeat.ts)
# ────────────────────────────────────────────────────────────

HEARTBEAT_TOKEN = "HEARTBEAT_OK"
SILENT_REPLY_TOKEN = "NO_REPLY"

DEFAULT_HEARTBEAT_EVERY_SECS = 30 * 60          # 30 minutes
DEFAULT_ACK_MAX_CHARS = 300

# The default prompt injected as a user-message when the heartbeat fires.
# Kept intentionally tight to avoid the model hallucinating old tasks.
DEFAULT_HEARTBEAT_PROMPT = (
    "Read HEARTBEAT.md if it exists (workspace context). "
    "Follow it strictly. Do not infer or repeat old tasks from prior chats. "
    "If nothing needs attention, reply HEARTBEAT_OK."
)

# Specialised prompts for non-standard wake reasons.
EXEC_EVENT_PROMPT = (
    "An async command you ran earlier has completed. The result is shown in "
    "the system messages above. Please relay the command output to the user "
    "in a helpful way."
)


def build_cron_event_prompt(events: list[str]) -> str:
    body = "\n".join(events).strip()
    if not body:
        return "A scheduled cron event was triggered, but no content was found. Reply HEARTBEAT_OK."
    return (
        "A scheduled reminder has been triggered. The reminder content is:\n\n"
        + body
        + "\n\nPlease relay this reminder to the user in a helpful and friendly way."
    )


# ────────────────────────────────────────────────────────────
# 2. HEARTBEAT.md CONTENT CHECK
#    (mirrors isHeartbeatContentEffectivelyEmpty)
# ────────────────────────────────────────────────────────────

def is_heartbeat_content_empty(content: str | None) -> bool:
    """
    Returns True when the file exists but contains only comments,
    empty markdown headers, or blank lines — i.e. there are no
    actionable tasks.  Returning True lets the runner skip the
    (expensive) LLM API call entirely.

    A *missing* file returns False so the LLM can still decide.
    """
    if content is None:
        return False
    for line in content.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        # ATX heading lines (e.g. "# Tasks") are structural, not tasks
        if re.match(r"^#+(\s|$)", stripped):
            continue
        # Empty checkbox items like "- [ ]" are not tasks either
        if re.match(r"^[-*+]\s*(\[[\sXx]?\]\s*)?$", stripped):
            continue
        return False  # found real content
    return True


# ────────────────────────────────────────────────────────────
# 3. RESPONSE POST-PROCESSING
#    (mirrors stripHeartbeatToken / normalizeHeartbeatReply)
# ────────────────────────────────────────────────────────────

@dataclass
class StrippedReply:
    should_skip: bool
    text: str
    did_strip: bool


def strip_heartbeat_token(
    raw: str | None,
    *,
    mode: str = "message",
    max_ack_chars: int = DEFAULT_ACK_MAX_CHARS,
) -> StrippedReply:
    """
    After the LLM responds, this function decides what to do:

    - Pure "HEARTBEAT_OK"  → skip (nothing to deliver)
    - "HEARTBEAT_OK — also, your build failed"
        * In heartbeat mode: skip if the tail is ≤ ack_max_chars
        * In message mode: deliver the tail (strip the token)
    - No token at all → deliver the full text
    """
    if not raw or not raw.strip():
        return StrippedReply(should_skip=True, text="", did_strip=False)

    text = raw.strip()
    if HEARTBEAT_TOKEN not in text:
        return StrippedReply(should_skip=False, text=text, did_strip=False)

    # Strip token from edges (start/end) iteratively
    did_strip = False
    changed = True
    while changed:
        changed = False
        if text.startswith(HEARTBEAT_TOKEN):
            text = text[len(HEARTBEAT_TOKEN):].lstrip()
            did_strip = True
            changed = True
        if text.endswith(HEARTBEAT_TOKEN):
            text = text[: -len(HEARTBEAT_TOKEN)].rstrip()
            did_strip = True
            changed = True

    text = " ".join(text.split())  # collapse whitespace

    if not did_strip:
        return StrippedReply(should_skip=False, text=raw.strip(), did_strip=False)

    if not text:
        return StrippedReply(should_skip=True, text="", did_strip=True)

    # In heartbeat mode, short tails are swallowed (ack-only).
    if mode == "heartbeat" and len(text) <= max_ack_chars:
        return StrippedReply(should_skip=True, text="", did_strip=True)

    return StrippedReply(should_skip=False, text=text, did_strip=True)


# ────────────────────────────────────────────────────────────
# 4. DELIVERY TARGETS  (simplified)
# ────────────────────────────────────────────────────────────

class Channel(Enum):
    NONE = "none"
    WHATSAPP = "whatsapp"
    DISCORD = "discord"
    TELEGRAM = "telegram"


@dataclass
class DeliveryTarget:
    channel: Channel = Channel.NONE
    to: str | None = None
    account_id: str | None = None


# ────────────────────────────────────────────────────────────
# 5. HEARTBEAT RUNNER  (mirrors heartbeat-runner.ts)
# ────────────────────────────────────────────────────────────

@dataclass
class HeartbeatConfig:
    """Per-agent heartbeat configuration (from config.json)."""
    every_secs: int = DEFAULT_HEARTBEAT_EVERY_SECS
    prompt: str | None = None               # custom prompt override
    model: str | None = None                # model override for heartbeat runs
    ack_max_chars: int = DEFAULT_ACK_MAX_CHARS
    active_hours: str | None = None         # e.g. "08:00-22:00"
    session: str | None = None              # session key for heartbeat context
    target: str = "last"                    # delivery target strategy


@dataclass
class HeartbeatAgentState:
    agent_id: str
    config: HeartbeatConfig
    interval_secs: int
    last_run_ts: float | None = None
    next_due_ts: float = 0.0


@dataclass
class HeartbeatRunResult:
    status: str     # "ran" | "skipped" | "failed"
    reason: str | None = None
    duration_secs: float = 0.0


@dataclass
class AgentSession:
    """Minimal session state for the mock."""
    last_heartbeat_text: str = ""
    last_heartbeat_sent_at: float | None = None


def fake_llm_reply(prompt: str, heartbeat_md: str | None) -> str:
    """
    Stand-in for the real LLM call (getReplyFromConfig).
    In the real system this sends the heartbeat prompt as a user message
    into the agent's existing session, with HEARTBEAT.md already injected
    as workspace context in the system prompt.
    """
    if heartbeat_md and "check deployment" in heartbeat_md.lower():
        return "Your staging deployment has been stuck for 45 minutes. You may want to check the CI pipeline."
    if "async command" in prompt.lower():
        return "Your build finished successfully. All 142 tests passed."
    if "scheduled reminder" in prompt.lower():
        return "Reminder: you have a 1:1 with Sarah in 15 minutes."
    return HEARTBEAT_TOKEN


def run_heartbeat_once(
    agent_id: str,
    config: HeartbeatConfig,
    session: AgentSession,
    heartbeat_md_content: str | None,
    delivery: DeliveryTarget,
    reason: str = "interval",
    pending_events: list[str] | None = None,
) -> HeartbeatRunResult:
    """
    Execute a single heartbeat cycle for one agent.

    This is the core of the system. The lifecycle is:

    ┌─────────────────────────────────────────────────────────┐
    │  1. Gate checks (enabled? active hours? queue busy?)    │
    │  2. Skip if HEARTBEAT.md is empty (saves API cost)      │
    │  3. Pick the prompt (heartbeat / exec-event / cron)     │
    │  4. Call the LLM with that prompt                       │
    │  5. Post-process: strip HEARTBEAT_OK token              │
    │  6. Deduplicate (don't repeat the same alert < 24h)     │
    │  7. Deliver to channel (or swallow if "ok")             │
    └─────────────────────────────────────────────────────────┘
    """
    started = time.time()
    pending = pending_events or []

    # ── Step 1: gate checks ──
    if config.every_secs <= 0:
        return HeartbeatRunResult("skipped", "disabled")

    # (active_hours check would go here in the real code)

    # ── Step 2: skip if heartbeat file is effectively empty ──
    is_exec = reason == "exec-event"
    is_cron = reason.startswith("cron:")
    if (
        heartbeat_md_content is not None
        and is_heartbeat_content_empty(heartbeat_md_content)
        and not is_exec
        and not is_cron
    ):
        print(f"  [{agent_id}] HEARTBEAT.md is empty → skipping API call")
        return HeartbeatRunResult("skipped", "empty-heartbeat-file")

    # ── Step 3: choose the prompt ──
    has_exec_completion = any("exec finished" in e.lower() for e in pending)
    cron_events = [e for e in pending if e.strip() and "heartbeat" not in e.lower()]

    if has_exec_completion:
        prompt = EXEC_EVENT_PROMPT
        print(f"  [{agent_id}] Using EXEC_EVENT_PROMPT (async command finished)")
    elif is_cron and cron_events:
        prompt = build_cron_event_prompt(cron_events)
        print(f"  [{agent_id}] Using cron event prompt ({len(cron_events)} events)")
    else:
        prompt = config.prompt or DEFAULT_HEARTBEAT_PROMPT
        print(f"  [{agent_id}] Using default heartbeat prompt")

    # ── Step 4: call the LLM ──
    print(f"  [{agent_id}] Calling LLM...")
    reply_text = fake_llm_reply(prompt, heartbeat_md_content)
    print(f"  [{agent_id}] LLM replied: {reply_text!r}")

    # ── Step 5: post-process — strip HEARTBEAT_OK ──
    stripped = strip_heartbeat_token(
        reply_text, mode="heartbeat", max_ack_chars=config.ack_max_chars,
    )

    if stripped.should_skip:
        print(f"  [{agent_id}] Response is HEARTBEAT_OK → nothing to deliver")
        return HeartbeatRunResult("ran", duration_secs=time.time() - started)

    final_text = stripped.text

    # ── Step 6: deduplicate ──
    if (
        session.last_heartbeat_text.strip() == final_text.strip()
        and session.last_heartbeat_sent_at is not None
        and (started - session.last_heartbeat_sent_at) < 86400
    ):
        print(f"  [{agent_id}] Duplicate alert within 24h → suppressed")
        return HeartbeatRunResult("ran", duration_secs=time.time() - started)

    # ── Step 7: deliver ──
    if delivery.channel == Channel.NONE or not delivery.to:
        print(f"  [{agent_id}] No delivery target → alert logged but not sent")
        return HeartbeatRunResult("ran", duration_secs=time.time() - started)

    print(f"  [{agent_id}] DELIVERING to {delivery.channel.value}:{delivery.to}")
    print(f"  [{agent_id}]   → \"{final_text}\"")

    # Record for dedup
    session.last_heartbeat_text = final_text
    session.last_heartbeat_sent_at = started

    return HeartbeatRunResult("ran", duration_secs=time.time() - started)


# ────────────────────────────────────────────────────────────
# 6. WAKE COORDINATOR
#    (mirrors heartbeat-wake.ts — the "mailbox" singleton)
#
#    This is the key piece that makes the timer non-blocking.
#    Multiple sources can call request_heartbeat_now() at any
#    time; the coordinator debounces + serialises actual runs.
# ────────────────────────────────────────────────────────────

# Priority: higher number wins when multiple wake reasons
# arrive within the coalesce window.
REASON_PRIORITY = {
    "retry": 0,
    "interval": 1,
    "default": 2,      # fallback
    "manual": 3,
    "exec-event": 3,
    "cron": 3,
}


@dataclass
class PendingWake:
    reason: str
    priority: int
    requested_at: float


class WakeCoordinator:
    """
    Singleton mailbox that coalesces heartbeat wake requests.

    In the TypeScript code this is module-level state in heartbeat-wake.ts.
    The pattern is:

        requestHeartbeatNow()
          → queue the reason (highest priority wins)
          → schedule a short debounce timer (250ms default)
          → when timer fires:
              if already running → mark "scheduled" and re-queue
              else → call handler(reason), then re-schedule if pending

    This gives you:
      - At most ONE heartbeat running at a time (no locks, no mutex —
        just a boolean flag, safe because Node is single-threaded and
        Python asyncio is single-threaded within a loop).
      - Coalescing: if 3 wake requests arrive in 250ms, only one run happens.
      - Priority: "manual" beats "interval" beats "retry".
    """

    def __init__(self):
        self.handler: HeartbeatWakeHandler | None = None
        self.pending: PendingWake | None = None
        self.running = False
        self.scheduled = False
        self._timer: asyncio.TimerHandle | None = None

    def set_handler(self, handler: HeartbeatWakeHandler | None):
        self.handler = handler

    def request_now(self, reason: str = "requested", coalesce_secs: float = 0.25):
        """
        Any part of the system calls this to request a heartbeat.
        Mirrors requestHeartbeatNow() in heartbeat-wake.ts.
        """
        self._queue_reason(reason)
        self._schedule(coalesce_secs)

    def _queue_reason(self, reason: str):
        priority = REASON_PRIORITY.get(reason, REASON_PRIORITY["default"])
        candidate = PendingWake(reason, priority, time.time())
        if self.pending is None or candidate.priority >= self.pending.priority:
            self.pending = candidate

    def _schedule(self, delay: float):
        if self._timer is not None:
            return  # already scheduled, keep the earlier one
        loop = asyncio.get_event_loop()
        self._timer = loop.call_later(delay, lambda: asyncio.ensure_future(self._fire()))

    async def _fire(self):
        """
        Timer callback.  Runs the handler if not already running,
        otherwise marks "scheduled" so we re-run after current finishes.
        """
        self._timer = None
        self.scheduled = False

        if self.handler is None:
            return
        if self.running:
            # Another heartbeat is in progress — don't run in parallel.
            # Just mark that we need to re-run when it finishes.
            self.scheduled = True
            return

        reason = self.pending.reason if self.pending else None
        self.pending = None
        self.running = True
        try:
            await self.handler(reason or "requested")
        finally:
            self.running = False
            # If new requests arrived while we were running, fire again.
            if self.pending or self.scheduled:
                self._schedule(0)


# Type alias for the handler function the coordinator calls.
HeartbeatWakeHandler = object  # actually Callable[[str], Awaitable[...]]


# ────────────────────────────────────────────────────────────
# 7. SCHEDULER  (mirrors startHeartbeatRunner)
#
#    Layer 1: maintains per-agent due times, sets a single
#    setTimeout pointing to the next due agent.
# ────────────────────────────────────────────────────────────

class HeartbeatScheduler:
    """
    Mirrors startHeartbeatRunner() from heartbeat-runner.ts.

    Lifecycle:
      1. On startup (or config change), compute nextDueMs for each agent.
      2. Find the earliest due time across all agents.
      3. Set ONE timer (setTimeout / call_later) for that delay.
      4. When timer fires → requestHeartbeatNow("interval")
         → the wake coordinator calls back into run()
         → run() iterates agents, runs any that are due
         → advances their nextDueMs
         → calls scheduleNext() to set the next timer.

    This means:
      - Between heartbeats (e.g. 30 minutes) there is NO thread, NO polling
        loop, NO sleep — just a single registered timer callback sitting in
        the event loop's timer heap.
      - In Node: setTimeout + timer.unref() (lets process exit)
      - In Python asyncio: loop.call_later() (same idea)
    """

    def __init__(self, agents: list[HeartbeatAgentState], coordinator: WakeCoordinator):
        self.agents = {a.agent_id: a for a in agents}
        self.coordinator = coordinator
        self.stopped = False
        self._timer: asyncio.TimerHandle | None = None

        # Register ourselves as the wake handler.
        coordinator.set_handler(self._on_wake)

    async def _on_wake(self, reason: str):
        """Called by the WakeCoordinator when it's time to run."""
        if self.stopped:
            return
        now = time.time()
        is_interval = reason == "interval"

        for agent in self.agents.values():
            if is_interval and now < agent.next_due_ts:
                continue
            # In the real code this calls runHeartbeatOnce() and awaits the LLM.
            print(f"    [scheduler] running heartbeat for {agent.agent_id} (reason={reason})")
            agent.last_run_ts = now
            agent.next_due_ts = now + agent.interval_secs

        self._schedule_next()

    def _schedule_next(self):
        """
        Find the earliest due agent and set a single timer.
        This is the entire "cost" of the heartbeat system between runs.
        """
        if self.stopped:
            return
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        if not self.agents:
            return

        now = time.time()
        earliest = min(a.next_due_ts for a in self.agents.values())
        delay = max(0.0, earliest - now)

        loop = asyncio.get_event_loop()
        self._timer = loop.call_later(
            delay,
            lambda: self.coordinator.request_now("interval", coalesce_secs=0),
        )
        return delay  # for demo printing

    def start(self):
        """Kick off the first timer."""
        now = time.time()
        for agent in self.agents.values():
            if agent.next_due_ts <= now:
                agent.next_due_ts = now + agent.interval_secs
        delay = self._schedule_next()
        return delay

    def stop(self):
        self.stopped = True
        if self._timer:
            self._timer.cancel()


# ────────────────────────────────────────────────────────────
# 8. DEMO — walk through scenarios + show the async timer
# ────────────────────────────────────────────────────────────

def run_sync_scenarios():
    """Synchronous scenarios (same as before)."""
    banner = "=" * 64
    print(banner)
    print("  OpenClaw Heartbeat System — Walkthrough")
    print(banner)

    session = AgentSession()
    delivery = DeliveryTarget(Channel.WHATSAPP, to="+1234567890")

    print("\n--- Scenario A: HEARTBEAT.md is empty (only headers) ---")
    run_heartbeat_once(
        "agent-1", HeartbeatConfig(), session,
        heartbeat_md_content="# Tasks\n\n- [ ]\n",
        delivery=delivery,
    )

    print("\n--- Scenario B: HEARTBEAT.md has tasks, but nothing needs attention ---")
    run_heartbeat_once(
        "agent-1", HeartbeatConfig(), session,
        heartbeat_md_content="# Tasks\n- Water the plants every Monday\n",
        delivery=delivery,
    )

    print("\n--- Scenario C: LLM detects an issue and sends an alert ---")
    run_heartbeat_once(
        "agent-1", HeartbeatConfig(), session,
        heartbeat_md_content="# Tasks\n- Check deployment status on staging\n",
        delivery=delivery,
    )

    print("\n--- Scenario D: same alert fires again → deduplicated ---")
    run_heartbeat_once(
        "agent-1", HeartbeatConfig(), session,
        heartbeat_md_content="# Tasks\n- Check deployment status on staging\n",
        delivery=delivery,
    )

    print("\n--- Scenario E: exec-event wake (async command finished) ---")
    run_heartbeat_once(
        "agent-1", HeartbeatConfig(), session,
        heartbeat_md_content=None,
        delivery=delivery,
        reason="exec-event",
        pending_events=["exec finished: npm run build"],
    )

    print("\n--- Scenario F: cron-event wake (scheduled reminder) ---")
    run_heartbeat_once(
        "agent-1", HeartbeatConfig(), session,
        heartbeat_md_content=None,
        delivery=delivery,
        reason="cron:daily-standup",
        pending_events=["Reminder: prepare daily standup notes"],
    )

    print("\n--- Scenario G: no delivery target configured ---")
    run_heartbeat_once(
        "agent-1", HeartbeatConfig(), session,
        heartbeat_md_content="# Tasks\n- Check deployment status on staging\n",
        delivery=DeliveryTarget(),
    )


async def run_async_timer_demo():
    """
    Demonstrate the real timer architecture using asyncio.

    This shows that NO threads are consumed and NO busy-loops run.
    The event loop sleeps between heartbeats, exactly like Node.js.
    We use a 1-second interval here instead of 30 minutes for demo speed.
    """
    banner = "=" * 64
    print(f"\n{banner}")
    print("  Async Timer Demo (asyncio — mirrors the real Node.js approach)")
    print(banner)
    print()
    print("  The key insight: between heartbeats, NO thread is blocked.")
    print("  asyncio.call_later() registers a callback in the event loop's")
    print("  timer heap — identical to Node's setTimeout + timer.unref().")
    print("  The process is free to do other work (or idle at zero CPU).")
    print()

    DEMO_INTERVAL = 1.0  # 1 second (instead of 30 minutes)

    coordinator = WakeCoordinator()
    agent_state = HeartbeatAgentState(
        agent_id="demo-agent",
        config=HeartbeatConfig(every_secs=int(DEMO_INTERVAL)),
        interval_secs=int(DEMO_INTERVAL),
        next_due_ts=0,
    )
    scheduler = HeartbeatScheduler([agent_state], coordinator)

    t0 = time.time()
    scheduler.start()
    print(f"  t=0.0s: scheduler started (interval={DEMO_INTERVAL}s)")

    # Also demonstrate an out-of-band wake at t≈1.5s
    async def inject_manual_wake():
        await asyncio.sleep(1.5)
        elapsed = time.time() - t0
        print(f"  t={elapsed:.1f}s: >>> external event: requestHeartbeatNow('exec-event')")
        coordinator.request_now("exec-event", coalesce_secs=0.05)

    asyncio.ensure_future(inject_manual_wake())

    # Let it run for ~3.5 seconds to see a few heartbeats
    await asyncio.sleep(3.5)
    scheduler.stop()
    elapsed = time.time() - t0
    print(f"  t={elapsed:.1f}s: scheduler stopped")

    print()
    print("  What happened above:")
    print("    - t≈1.0s: first interval timer fired (scheduled at start)")
    print("    - t≈1.5s: external exec-event arrived → immediate wake")
    print("    - t≈2.0s: next interval timer fired")
    print("    - t≈3.0s: next interval timer fired")
    print("    - Between all of these, the event loop was idle (no threads, no CPU).")
    print(f"\n{banner}")
    print("  Done. The timer architecture is the same in Node.js and Python asyncio:")
    print("  a single callback in the event loop's timer heap, re-set after each run.")
    print(banner)


def main():
    run_sync_scenarios()
    asyncio.run(run_async_timer_demo())


if __name__ == "__main__":
    main()
