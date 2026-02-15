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

Run this file to see the full lifecycle printed step by step:

    python examples/heartbeat_explainer.py
"""

from __future__ import annotations

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
# 6. SCHEDULER  (mirrors startHeartbeatRunner)
# ────────────────────────────────────────────────────────────

class HeartbeatScheduler:
    """
    The runner maintains a per-agent timer.  Every `interval` seconds it
    fires `requestHeartbeatNow()` which coalesces into `run()`.

    In the real code this is a setTimeout loop.  Here we just simulate
    discrete ticks so you can see the flow.
    """

    def __init__(self, agents: list[HeartbeatAgentState]):
        self.agents = {a.agent_id: a for a in agents}
        self.stopped = False

    def tick(self, now: float, run_fn):
        """Advance one scheduler tick: run any agents that are due."""
        if self.stopped:
            return
        for agent in self.agents.values():
            if now >= agent.next_due_ts:
                run_fn(agent.agent_id)
                agent.last_run_ts = now
                agent.next_due_ts = now + agent.interval_secs

    def stop(self):
        self.stopped = True


# ────────────────────────────────────────────────────────────
# 7. DEMO — walk through every important scenario
# ────────────────────────────────────────────────────────────

def main():
    banner = "=" * 64
    print(banner)
    print("  OpenClaw Heartbeat System — Walkthrough")
    print(banner)

    session = AgentSession()
    delivery = DeliveryTarget(Channel.WHATSAPP, to="+1234567890")

    # ── Scenario A: empty HEARTBEAT.md → skip entirely ──
    print("\n--- Scenario A: HEARTBEAT.md is empty (only headers) ---")
    run_heartbeat_once(
        "agent-1", HeartbeatConfig(), session,
        heartbeat_md_content="# Tasks\n\n- [ ]\n",
        delivery=delivery,
    )

    # ── Scenario B: HEARTBEAT.md has real tasks → LLM says OK ──
    print("\n--- Scenario B: HEARTBEAT.md has tasks, but nothing needs attention ---")
    run_heartbeat_once(
        "agent-1", HeartbeatConfig(), session,
        heartbeat_md_content="# Tasks\n- Water the plants every Monday\n",
        delivery=delivery,
    )

    # ── Scenario C: LLM finds something to report ──
    print("\n--- Scenario C: LLM detects an issue and sends an alert ---")
    run_heartbeat_once(
        "agent-1", HeartbeatConfig(), session,
        heartbeat_md_content="# Tasks\n- Check deployment status on staging\n",
        delivery=delivery,
    )

    # ── Scenario D: same alert again → deduplicated ──
    print("\n--- Scenario D: same alert fires again → deduplicated ---")
    run_heartbeat_once(
        "agent-1", HeartbeatConfig(), session,
        heartbeat_md_content="# Tasks\n- Check deployment status on staging\n",
        delivery=delivery,
    )

    # ── Scenario E: exec-event wake-up ──
    print("\n--- Scenario E: exec-event wake (async command finished) ---")
    run_heartbeat_once(
        "agent-1", HeartbeatConfig(), session,
        heartbeat_md_content=None,
        delivery=delivery,
        reason="exec-event",
        pending_events=["exec finished: npm run build"],
    )

    # ── Scenario F: cron-event wake-up ──
    print("\n--- Scenario F: cron-event wake (scheduled reminder) ---")
    run_heartbeat_once(
        "agent-1", HeartbeatConfig(), session,
        heartbeat_md_content=None,
        delivery=delivery,
        reason="cron:daily-standup",
        pending_events=["Reminder: prepare daily standup notes"],
    )

    # ── Scenario G: no delivery target → logged only ──
    print("\n--- Scenario G: no delivery target configured ---")
    run_heartbeat_once(
        "agent-1", HeartbeatConfig(), session,
        heartbeat_md_content="# Tasks\n- Check deployment status on staging\n",
        delivery=DeliveryTarget(),  # channel=NONE
    )

    # ── Scenario H: show the scheduler ──
    print("\n--- Scenario H: scheduler tick simulation ---")
    agent_state = HeartbeatAgentState(
        agent_id="agent-1",
        config=HeartbeatConfig(every_secs=1800),
        interval_secs=1800,
        next_due_ts=0,
    )
    scheduler = HeartbeatScheduler([agent_state])

    def mock_run(agent_id):
        print(f"  scheduler → heartbeat fired for {agent_id}")

    now = 0.0
    for minute in [0, 15, 30, 45, 60]:
        now = minute * 60.0
        print(f"  t={minute}m: ", end="")
        scheduler.tick(now, mock_run)
        if minute != 0 and now < agent_state.next_due_ts:
            print(f"  scheduler → not due yet (next at t={agent_state.next_due_ts/60:.0f}m)")

    scheduler.stop()
    print(f"\n{banner}")
    print("  Done. See docstrings and comments for the full mental model.")
    print(banner)


if __name__ == "__main__":
    main()
