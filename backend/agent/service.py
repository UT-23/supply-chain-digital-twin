"""
The agent service: perceive -> reason -> act, on a cycle.

TRIGGERING -- event-driven, with a heartbeat floor and a cooldown ceiling.

The loop wakes every AGENT_POLL_SECONDS (cheap, DB-only) and checks for unhandled
events. It calls the model when either:

  * there is at least one unhandled event (trigger="events"), or
  * AGENT_HEARTBEAT_SECONDS have passed with no call (trigger="heartbeat").

and never more often than AGENT_MIN_INTERVAL_SECONDS apart.

Why not a plain timer: an LLM call costs money and takes seconds. The simulation
is quiet most of the time, so a fixed 5s timer would spend most of its budget
concluding that nothing is wrong. Why not purely event-driven: then the agent
would never notice a slow drift that no discrete event fired for, and would go
permanently silent if the simulation stopped emitting. The heartbeat covers that.
The cooldown stops a burst of simultaneous disruptions from firing overlapping
calls.

RUNNER -- a LangGraph state graph:

    START -> perceive -> reason -> (tool calls?) -> act -> reason -> ... -> END

`perceive` reads the database and retrieves comparable past cases, `reason` is
the model call with tools bound, `act` executes them, and the conditional edge
loops back only while the model keeps asking for tools.

CONCURRENCY -- the same as the Phase 3 worker: an asyncio task whose database
work runs in asyncio.to_thread against the shared pooled engine, while the model
call happens on the event loop. Exactly one cycle is ever in flight.

PROVIDER -- Groq via langchain-groq; Gemini via langchain-openai against its
OpenAI-compatible endpoint. Groq's free tier is why: this loop runs
continuously, and a paid per-token API would bill for every cycle.

The LangGraph migration changed HOW the loop runs, not WHAT it records. The tool
schemas, tool dispatch, cites_memory heuristic, agent_decisions row shape and
rate-limit budget are all unchanged.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Annotated, Any, Callable, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.graph.message import add_messages

from db.models import AgentDecision
from db.session import SessionLocal

from . import memory, outcomes, perception, schemas
from .prompts import SYSTEM_PROMPT, build_user_message
from .tools import ToolError

log = logging.getLogger("agent.service")


class _GraphState(TypedDict, total=False):
    """
    State carried between graph nodes.

    `messages` uses LangGraph's add_messages reducer so each node returns only
    what it appends. `record` is the agent_decisions row under construction and
    is mutated in place -- it is the same dict shape the hand-written loop
    produced, so persistence and the WebSocket payload are unchanged.
    """
    messages: Annotated[list, add_messages]
    record: dict
    model: Any
    trigger: str
    iterations: int
    perceived: dict | None
    cases: list | None


def _f(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _i(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


ENABLED = os.getenv("AGENT_ENABLED", "1").lower() not in ("0", "false", "no")
POLL_SECONDS = _f("AGENT_POLL_SECONDS", 5.0)
MIN_INTERVAL_SECONDS = _f("AGENT_MIN_INTERVAL_SECONDS", 20.0)
HEARTBEAT_SECONDS = _f("AGENT_HEARTBEAT_SECONDS", 300.0)

# Verified by querying /models with the project's own key on 2026-09-14, not
# from documentation: Groq's docs list llama-3.3-70b-versatile, but this account
# gets 404 model_not_found for it. The account's tool-capable models are
# openai/gpt-oss-120b, openai/gpt-oss-20b, qwen/qwen3.8-27b, qwen/qwen3.6-27b.
#
# openai/gpt-oss-120b is the default, chosen on measured free-tier limits:
#
#   gpt-oss-120b   ~200k tokens/day, generous output-per-minute. At this prompt
#                  size that is ~45 calls/day. Serialises tool calls (one per
#                  turn) but the MAX_ITERATIONS loop handles that. Strongest
#                  reasoner available on this account.
#   qwen3.8-27b    the only model here that emits PARALLEL tool calls, but its
#                  output-tokens-per-minute ceiling is 1000 -- a single request
#                  with max_tokens=4096 is rejected outright ("Request too large
#                  ... OTPM Limit 1000"), and even at max_tokens=800 it allows
#                  roughly one call per minute. Unusable for this agent's turn
#                  sizes. Set AGENT_MODEL + AGENT_MAX_TOKENS<=800 to try it.
#
# Free-tier quotas are per-day and shared across probes, so exhausting one model
# does not free the other. Re-check with client.models.list(); the documented
# llama-3.3-70b-versatile 404s on this account.
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")

# Providers, both reached through the OpenAI SDK. Gemini publishes an
# OpenAI-compatible endpoint, so the same client code and the same tool
# definitions serve both -- see schemas.tool_schemas(dialect=...) for the one
# place they differ. The default is and stays Groq; Gemini exists so a single
# experiment can be run when Groq's daily quota is spent.
PROVIDERS = {
    "groq": {
        "base_url": GROQ_BASE_URL,
        "key_env": "GROQ_API_KEY",
        "default_model": "openai/gpt-oss-120b",
        "dialect": "openai",
        "key_hint": "https://console.groq.com/keys",
        # Groq DOES return x-ratelimit-* headers; these were read from a live
        # response on 2026-09-14. Note the headers expose requests/day and
        # tokens/MINUTE but not the 200k tokens/day cap, which is the one that
        # actually bit -- that only surfaces in the 429 body.
        "free_tier": {"rpd": 1000, "tpm": 8000, "tpd": 200000,
                      "checked": "2026-09-14"},
    },
    "gemini": {
        "base_url": os.getenv(
            "GEMINI_BASE_URL",
            "https://generativelanguage.googleapis.com/v1beta/openai/",
        ),
        "key_env": "GEMINI_API_KEY",
        "default_model": "gemini-3.5-flash",
        "dialect": "gemini",
        "key_hint": "https://aistudio.google.com/apikey",
        # Read off the AI Studio dashboard on 2026-09-14 for gemini-3.5-flash.
        # Google publishes no per-model free-tier numbers in its docs and its
        # OpenAI-compatible endpoint returns no x-ratelimit-* headers, so the
        # dashboard is the only source. Re-check if runs start failing.
        # 20 RPD makes this provider unusable for the memory A/B, which needs
        # ~18 requests -- past the 80% (=16) stop ceiling.
        "free_tier": {"rpm": 5, "rpd": 20, "checked": "2026-09-14"},
    },
}

PROVIDER = os.getenv("AGENT_PROVIDER", "groq").strip().lower()
if PROVIDER not in PROVIDERS:
    raise RuntimeError(f"AGENT_PROVIDER must be one of {sorted(PROVIDERS)}, got {PROVIDER!r}")
_P = PROVIDERS[PROVIDER]
DIALECT = _P["dialect"]

MODEL = os.getenv("AGENT_MODEL", _P["default_model"])
TEMPERATURE = _f("AGENT_TEMPERATURE", 0.3)
# Groq counts prompt + max_tokens against the 8,000 tokens-per-minute ceiling,
# so an oversized max_tokens silently consumes half the budget and produces
# HTTP 413 "Request too large" on a perfectly reasonable prompt. Observed
# completions for this agent are 150-500 tokens; 1024 is generous.
MAX_TOKENS = _i("AGENT_MAX_TOKENS", 1024)
# Hard ceiling on tool round-trips in one cycle, so a confused model cannot loop.
MAX_ITERATIONS = _i("AGENT_MAX_ITERATIONS", 6)
READ_ONLY = os.getenv("AGENT_READ_ONLY", "0").lower() in ("1", "true", "yes")

# -- memory --
# Master switch. AGENT_MEMORY=0 is the control condition for an A/B: the agent
# reasons on current evidence only, exactly as it did before this phase.
MEMORY_ENABLED = os.getenv("AGENT_MEMORY", "1").lower() not in ("0", "false", "no")
MEMORY_CASES = _i("AGENT_MEMORY_CASES", 4)
# How long to wait before judging what came of a decision. Must be long enough
# for a simulated condition to plausibly resolve: sensor spikes last 3-8
# simulation cycles, so at the 5s default this is comfortably past that.
OUTCOME_WINDOW_SECONDS = _f("AGENT_OUTCOME_WINDOW_SECONDS", 120.0)
API_TIMEOUT = _f("AGENT_API_TIMEOUT", 120.0)

# Free-tier rate limits are hit routinely, so retry the per-minute ones.
RETRY_ATTEMPTS = _i("AGENT_RETRY_ATTEMPTS", 5)
RETRY_BASE_SECONDS = _f("AGENT_RETRY_BASE_SECONDS", 20.0)
RETRY_MAX_SECONDS = _f("AGENT_RETRY_MAX_SECONDS", 90.0)

# Cycle-level backoff. The per-call retries above handle a single rate-limited
# invocation; they do nothing about a cycle that fails outright, which the loop
# used to re-attempt on the very next poll. On 2026-09-20 that produced 872
# attempts in 7.7 hours for ONE successful decision, because every failure was
# retried 5s later against a quota that was already gone. After a failed cycle
# we now skip 1 poll, then 2, then 4 ... doubling until a cycle succeeds.
FAILURE_BACKOFF_MAX_CYCLES = _i("AGENT_FAILURE_BACKOFF_MAX_CYCLES", 32)

# A tokens-per-day rejection is categorically different from a per-minute one:
# no amount of waiting inside the day helps, and each further attempt eats into
# whatever few tokens do free up. On TPD we stop attempting altogether.
TPD_COOLDOWN_HOURS = _f("AGENT_TPD_COOLDOWN_HOURS", 4.0)


def _is_tpd_rejection(text: str | None) -> bool:
    """
    True if `text` is a tokens-per-day rejection rather than a per-minute one.

    Groq states the window in the 429 body ("tokens per day (TPD)"); there is no
    header that carries it, so the body is the only signal available.
    """
    if not text:
        return False
    low = text.lower()
    return "(tpd)" in low or "tokens per day" in low or "per day" in low


class BudgetExhausted(RuntimeError):
    """Raised when the next call would cross the configured free-tier ceiling."""


class RateBudget:
    """
    Hard spend guard for free-tier providers.

    Free tiers are metered on several axes at once (requests/minute,
    requests/day, tokens/minute, tokens/day) and a provider will happily let a
    long run walk straight into the ceiling. This tracks what a run has actually
    consumed and refuses the call that would cross `stop_fraction` of a limit,
    rather than discovering the ceiling by hitting it.

    Limits are supplied by the caller and are never guessed: Google does not
    publish per-model free-tier numbers in its documentation and its OpenAI
    compatibility endpoint returns no x-ratelimit-* headers, so the only
    authoritative source is the provider's own dashboard.
    """

    def __init__(
        self,
        max_requests_per_day: int | None = None,
        max_tokens_per_day: int | None = None,
        min_seconds_between_calls: float = 0.0,
        stop_fraction: float = 0.8,
        requests_already_used: int = 0,
        tokens_already_used: int = 0,
    ) -> None:
        self.max_requests_per_day = max_requests_per_day
        self.max_tokens_per_day = max_tokens_per_day
        self.min_seconds_between_calls = min_seconds_between_calls
        self.stop_fraction = stop_fraction
        # Usage the day already had before this run started. Without it the guard
        # only sees its own spend and will happily start a run into a nearly-full
        # daily bucket -- which is exactly how a 9/9 rate-limited run happened.
        self.requests = int(requests_already_used)
        self.tokens = int(tokens_already_used)
        self._last_call: float | None = None

    def _ceiling(self, limit: int | None) -> float | None:
        return None if limit is None else limit * self.stop_fraction

    def check(self, estimated_tokens: int = 0) -> None:
        """Raise BudgetExhausted if the next call would cross a ceiling."""
        rc = self._ceiling(self.max_requests_per_day)
        if rc is not None and self.requests + 1 > rc:
            raise BudgetExhausted(
                f"next request would be #{self.requests + 1}, past "
                f"{self.stop_fraction:.0%} of the {self.max_requests_per_day} "
                f"requests/day free-tier ceiling"
            )
        tc = self._ceiling(self.max_tokens_per_day)
        if tc is not None and self.tokens + estimated_tokens > tc:
            raise BudgetExhausted(
                f"next request (~{estimated_tokens} tokens) would take the run to "
                f"{self.tokens + estimated_tokens}, past {self.stop_fraction:.0%} "
                f"of the {self.max_tokens_per_day} tokens/day free-tier ceiling"
            )

    async def pace(self) -> None:
        """Sleep so consecutive calls respect the configured minimum spacing."""
        if self.min_seconds_between_calls <= 0 or self._last_call is None:
            return
        elapsed = time.monotonic() - self._last_call
        wait = self.min_seconds_between_calls - elapsed
        if wait > 0:
            log.info("pacing: sleeping %.1fs to stay under the per-minute limit", wait)
            await asyncio.sleep(wait)

    def record(self, tokens: int = 0) -> None:
        self.requests += 1
        self.tokens += tokens
        self._last_call = time.monotonic()

    def summary(self) -> dict:
        return {
            "requests_used": self.requests,
            "tokens_used": self.tokens,
            "requests_per_day_limit": self.max_requests_per_day,
            "tokens_per_day_limit": self.max_tokens_per_day,
            "stop_fraction": self.stop_fraction,
            "requests_ceiling": self._ceiling(self.max_requests_per_day),
            "tokens_ceiling": self._ceiling(self.max_tokens_per_day),
        }


# Installed by callers that need a spend guard (the probes). None = unmetered.
BUDGET: RateBudget | None = None


def set_budget(budget: RateBudget | None) -> None:
    global BUDGET
    BUDGET = budget


class AgentUnavailable(RuntimeError):
    """Raised when the model client cannot be constructed (missing key/SDK)."""


def _build_client():
    """
    Construct the async OpenAI-SDK client for the selected provider.

    Both Groq and Gemini are reached through the OpenAI SDK -- Gemini publishes
    an OpenAI-compatible endpoint -- so only the base URL, key and model differ.

    Deferred to call time rather than import time so the rest of the backend runs
    (and /health reports honestly) when the SDK or credentials are absent.
    """
    try:
        from openai import AsyncOpenAI
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise AgentUnavailable(
            "the `openai` package is not installed -- pip install -r backend/requirements.txt"
        ) from exc

    key_env = _P["key_env"]
    api_key = os.getenv(key_env)
    if not api_key:
        raise AgentUnavailable(
            f"{key_env} is not set in backend/.env -- the agent cannot reason "
            f"without it. Get a free key at {_P['key_hint']}"
        )

    return AsyncOpenAI(api_key=api_key, base_url=_P["base_url"], timeout=API_TIMEOUT)


# Phrases a model uses when it is drawing on the supplied case history. This is
# a TEXT HEURISTIC, not ground truth: it can miss a paraphrase and can fire on a
# coincidental phrase. It exists so the UI can distinguish "memory was supplied"
# (retrieved_case_count) from "memory was apparently used" (cites_memory), which
# are different claims. Treat it as indicative.
_MEMORY_PHRASES = re.compile(
    r"(self[- ]resolv|resolved (?:itself|themselves)|past (?:similar|case|spike|incident)"
    r"|previous case|prior case|recent, similar|in the last day|left unattended"
    r"|historical|history shows|earlier (?:case|spike|incident))",
    re.IGNORECASE,
)


def _cites_memory(reasoning: str | None, cases: list[dict[str, Any]] | None) -> bool:
    """True when the reasoning appears to draw on the retrieved cases."""
    if not reasoning or not cases:
        return False
    # An explicit "#41" reference is unambiguous; phrase match is the fallback.
    for case in cases:
        if f"#{case.get('case_id')}" in reasoning:
            return True
    return bool(_MEMORY_PHRASES.search(reasoning))


def _now() -> datetime:
    return datetime.now(timezone.utc)


class AgentService:
    """Owns the decision loop and the state /health reports."""

    def __init__(self, broadcast: Callable[[dict], Any] | None = None) -> None:
        self._task: asyncio.Task | None = None
        self._stopping = asyncio.Event()
        self._client = None
        self._graph = None   # compiled LangGraph, built lazily
        # Called with each persisted decision so it can be pushed to the UI.
        self._broadcast = broadcast

        self.started_at: datetime | None = None
        self.last_poll_at: datetime | None = None
        self.last_decision_at: datetime | None = None
        self.decisions = 0
        self.actions_taken = 0
        self.no_action_decisions = 0
        self.failures = 0
        self.cases_retrieved_total = 0
        self.decisions_scored = 0
        self.last_error: str | None = None
        self.unavailable_reason: str | None = None

        # Cycle-level backoff state (see FAILURE_BACKOFF_MAX_CYCLES).
        self._consecutive_failures = 0
        self._skip_polls_remaining = 0
        self._tpd_blocked_until: float | None = None        # time.monotonic()
        self._tpd_blocked_until_wall: datetime | None = None  # for /health
        self.cycles_skipped = 0

    # -- lifecycle ---------------------------------------------------------

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    def start(self) -> None:
        if self.running:
            return
        self._stopping.clear()
        self.started_at = _now()
        self._task = asyncio.create_task(self._run(), name="agent-service")
        log.info(
            "agent service started (model=%s, temp=%.2f, poll=%.0fs, min_interval=%.0fs, "
            "heartbeat=%.0fs, read_only=%s)",
            MODEL, TEMPERATURE, POLL_SECONDS, MIN_INTERVAL_SECONDS, HEARTBEAT_SECONDS, READ_ONLY,
        )

    async def stop(self) -> None:
        if self._task is None:
            return
        self._stopping.set()
        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._task
        self._task = None
        if self._client is not None:
            with contextlib.suppress(Exception):
                await self._client.close()
            self._client = None
        log.info("agent service stopped after %d decisions", self.decisions)

    async def _run(self) -> None:
        last_call = 0.0
        while not self._stopping.is_set():
            self.last_poll_at = _now()
            try:
                # Judge what came of earlier decisions whose window has closed.
                scored = await asyncio.to_thread(self._score_due)
                self.decisions_scored += scored

                pending = await asyncio.to_thread(self._count_pending)
                elapsed = time.monotonic() - last_call

                trigger = None
                if pending and elapsed >= MIN_INTERVAL_SECONDS:
                    trigger = "events"
                elif elapsed >= HEARTBEAT_SECONDS:
                    trigger = "heartbeat"

                if trigger:
                    skip = self._skip_reason()
                    if skip is not None:
                        self.cycles_skipped += 1
                        log.debug("skipping %s cycle: %s", trigger, skip)
                    else:
                        last_call = time.monotonic()
                        stored = await self.run_cycle(trigger=trigger)
                        self._note_cycle_result(stored)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - the loop must survive anything
                self.failures += 1
                self.last_error = f"{type(exc).__name__}: {exc}"
                log.exception("agent loop error")

            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self._stopping.wait(), timeout=POLL_SECONDS)

    def _skip_reason(self) -> str | None:
        """
        Reason to skip this poll's cycle, or None to proceed.

        Consumes one unit of the backoff when it defers, so the counter only
        drains on polls where a cycle would actually have run.
        """
        if self._tpd_blocked_until is not None:
            if time.monotonic() < self._tpd_blocked_until:
                until = self._tpd_blocked_until_wall
                return ("daily token quota (TPD) exhausted; holding until "
                        f"{until.isoformat() if until else 'restart'}")
            log.info("TPD cooldown elapsed -- resuming normal operation")
            self._tpd_blocked_until = None
            self._tpd_blocked_until_wall = None
            self._consecutive_failures = 0
            self._skip_polls_remaining = 0

        if self._skip_polls_remaining > 0:
            self._skip_polls_remaining -= 1
            return (f"backing off after {self._consecutive_failures} consecutive "
                    f"failure(s); {self._skip_polls_remaining} poll(s) still to skip")
        return None

    def _note_cycle_result(self, stored: dict[str, Any] | None) -> None:
        """Update the cycle-level backoff from the decision just persisted."""
        error = (stored or {}).get("error")

        if not error:
            if self._consecutive_failures:
                log.info("agent recovered after %d consecutive failure(s)",
                         self._consecutive_failures)
            self._consecutive_failures = 0
            self._skip_polls_remaining = 0
            return

        self._consecutive_failures += 1

        if _is_tpd_rejection(error):
            self._tpd_blocked_until = time.monotonic() + TPD_COOLDOWN_HOURS * 3600.0
            self._tpd_blocked_until_wall = _now() + timedelta(hours=TPD_COOLDOWN_HOURS)
            self._skip_polls_remaining = 0
            log.error(
                "daily token quota (TPD) exhausted -- suspending agent for %.1fh "
                "(until %s). Restart the backend to override.",
                TPD_COOLDOWN_HOURS, self._tpd_blocked_until_wall.isoformat())
            return

        self._skip_polls_remaining = min(
            2 ** (self._consecutive_failures - 1), FAILURE_BACKOFF_MAX_CYCLES)
        log.warning("cycle failed (%d consecutive) -- skipping next %d poll(s): %s",
                    self._consecutive_failures, self._skip_polls_remaining, error[:120])

    @staticmethod
    def _count_pending() -> int:
        with SessionLocal() as session:
            return perception.has_new_work(session)

    @staticmethod
    def _score_due() -> int:
        with SessionLocal() as session:
            return outcomes.score_due_decisions(session, int(OUTCOME_WINDOW_SECONDS))

    # -- one decision cycle ------------------------------------------------

    async def run_cycle(self, trigger: str = "manual") -> dict[str, Any]:
        """
        Perceive, reason, act, persist. Returns the stored decision as a dict.

        Never raises for an API failure: the failure is recorded as a decision row
        with `error` set, so a broken agent is visible rather than silent.
        """
        started = time.perf_counter()

        # Perception and memory retrieval now happen inside the graph's
        # `perceive` node. The record is pre-seeded with empty values so a
        # failure before/at perception still persists an honest row rather than
        # a KeyError.
        record: dict[str, Any] = {
            "trigger": trigger,
            "input_state": None,
            "reasoning": None,
            "tool_calls": [],
            "tool_results": [],
            "model": MODEL,
            "error": None,
            "stop_reason": None,
            "input_tokens": 0,
            "output_tokens": 0,
            "signature": None,
            "retrieved_case_count": 0,
            "retrieved_case_ids": [],
        }

        try:
            # Credential/SDK check only -- the graph builds its own LangChain
            # client. Keeping this here preserves the AgentUnavailable message
            # the probes and /health rely on.
            if self._client is None:
                self._client = _build_client()
                self.unavailable_reason = None
            await self._reason_and_act(None, trigger, record, None)
        except AgentUnavailable as exc:
            self.unavailable_reason = str(exc)
            record["error"] = str(exc)
            log.error("agent unavailable: %s", exc)
        except Exception as exc:  # noqa: BLE001
            record["error"] = f"{type(exc).__name__}: {exc}"
            log.exception("agent cycle failed")

        record["latency_ms"] = int((time.perf_counter() - started) * 1000)
        record["action_count"] = len(record["tool_calls"])
        # A cycle that reasoned but called nothing is a real "no action" decision;
        # one that errored is not.
        record["no_action"] = record["error"] is None and record["action_count"] == 0

        stored = await asyncio.to_thread(self._persist, record)

        self.decisions += 1
        self.last_decision_at = _now()
        if record["error"]:
            self.failures += 1
            self.last_error = record["error"]
        elif record["no_action"]:
            self.no_action_decisions += 1
        else:
            self.actions_taken += record["action_count"]

        if self._broadcast is not None:
            with contextlib.suppress(Exception):
                await self._broadcast({"type": "agent_decision", "decision": stored})

        return stored

    @staticmethod
    def _retrieve(sig: dict[str, Any]) -> list[dict[str, Any]]:
        with SessionLocal() as session:
            return memory.retrieve_similar(session, sig, limit=MEMORY_CASES)

    @staticmethod
    def _perceive() -> dict[str, Any]:
        with SessionLocal() as session:
            return perception.build_state(session)

    # ------------------------------------------------------------------
    # LangGraph implementation
    # ------------------------------------------------------------------
    #
    #     START -> perceive -> reason -> (tool calls?) -> act -> reason -> ...
    #                                         |  no
    #                                         v
    #                                        END
    #
    # `perceive` reads the database and retrieves comparable past cases.
    # `reason` is the model call with the tools bound.
    # `act` executes whatever the model asked for.
    # The conditional edge loops back to `reason` only while the model keeps
    # asking for tools, and stops as soon as it answers in plain text.
    #
    # What deliberately did NOT change in this migration: the tool schemas
    # (schemas.tool_schemas, generated from the Phase 4 docstrings), the tool
    # dispatch (_execute_tool), the memory-citation heuristic (_cites_memory),
    # the agent_decisions record shape, and the rate-limit budget. Those are the
    # parts verified over several sessions; this changes only how the loop runs.

    def _build_model(self, tool_defs: list[dict[str, Any]]):
        """
        A LangChain chat model for the selected provider, with tools bound.

        bind_tools receives the SAME OpenAI-format dicts the hand-written loop
        sent, so the model sees byte-identical tool names, descriptions and
        parameter schemas -- which is what keeps already-measured results
        comparable with anything run after this migration.
        """
        key = os.environ[_P["key_env"]]
        if PROVIDER == "groq":
            from langchain_groq import ChatGroq

            model = ChatGroq(
                model=MODEL, api_key=key, temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS, timeout=API_TIMEOUT,
                max_retries=0,   # retries handled by _invoke_with_backoff
            )
        else:
            # Gemini is reached through its OpenAI-compatible endpoint, so the
            # generic OpenAI chat model serves it with the same bound schemas.
            from langchain_openai import ChatOpenAI

            model = ChatOpenAI(
                model=MODEL, api_key=key, base_url=_P["base_url"],
                temperature=TEMPERATURE, max_tokens=MAX_TOKENS,
                timeout=API_TIMEOUT, max_retries=0,
            )
        return model.bind_tools(tool_defs)

    async def _invoke_with_backoff(self, model, messages):
        """
        One model invocation, with the same free-tier handling as before.

        Per-minute limits are waited out; a tokens-per-day rejection is re-raised
        immediately because waiting cannot help. The RateBudget guard is checked
        before spending, exactly as in the pre-migration loop.
        """
        estimate = sum(len(str(getattr(m, "content", m))) for m in messages) // 4

        delay = RETRY_BASE_SECONDS
        for attempt in range(RETRY_ATTEMPTS):
            if BUDGET is not None:
                BUDGET.check(estimate)
                await BUDGET.pace()
            try:
                reply = await model.ainvoke(messages)
                if BUDGET is not None:
                    usage = getattr(reply, "usage_metadata", None) or {}
                    BUDGET.record(usage.get("total_tokens") or estimate)
                return reply
            except Exception as exc:  # noqa: BLE001 - provider SDKs raise varied types
                body = str(exc)
                if "429" not in body and "rate limit" not in body.lower():
                    raise
                if BUDGET is not None:
                    BUDGET.record(0)
                if "per day" in body or "(TPD)" in body:
                    raise
                if attempt == RETRY_ATTEMPTS - 1:
                    raise

                wait = delay
                hint = re.search(r"try again in ([0-9.]+)s", body)
                if hint:
                    wait = min(float(hint.group(1)) + 1.0, RETRY_MAX_SECONDS)
                else:
                    hint_m = re.search(r"try again in (\d+)m([0-9.]+)s", body)
                    if hint_m:
                        wait = min(
                            int(hint_m.group(1)) * 60 + float(hint_m.group(2)) + 1.0,
                            RETRY_MAX_SECONDS,
                        )
                log.warning("rate limited (attempt %d/%d), sleeping %.1fs",
                            attempt + 1, RETRY_ATTEMPTS, wait)
                await asyncio.sleep(wait)
                delay = min(delay * 2, RETRY_MAX_SECONDS)

    def _build_graph(self):
        """Compile the perceive -> reason -> act state graph (once, then cached)."""
        from langgraph.graph import END, StateGraph

        async def perceive(gs: dict) -> dict:
            """
            Read current DB state and retrieve comparable past cases.

            Passes through untouched when a caller already supplied the state --
            that is how the probes pin one situation byte-identically across
            arms. Perception must not overwrite a pinned fixture.
            """
            rec = gs["record"]
            cases = gs.get("cases")
            perceived = gs.get("perceived")

            if perceived is None:
                perceived = await asyncio.to_thread(self._perceive)
                sig = memory.signature(perceived)
                cases = (
                    await asyncio.to_thread(self._retrieve, sig)
                    if MEMORY_ENABLED else []
                )
                rec["signature"] = sig
                rec["retrieved_case_count"] = len(cases)
                rec["retrieved_case_ids"] = [c["case_id"] for c in cases]
                self.cases_retrieved_total += len(cases)

            rec["input_state"] = perceived
            return {
                "perceived": perceived,
                "cases": cases or [],
                "messages": [
                    SystemMessage(content=SYSTEM_PROMPT),
                    # Compact JSON, not indent=2: pretty-printing the snapshot
                    # inflated it by ~48% (3,858 vs 2,609 tokens) in pure
                    # whitespace, against an 8,000 tokens-per-minute ceiling.
                    HumanMessage(content=build_user_message(
                        json.dumps(perceived, default=str),
                        gs["trigger"],
                        memory.summarize_for_prompt(cases) if cases else None,
                    )),
                ],
            }

        async def reason(gs: dict) -> dict:
            """Model call with the tools bound."""
            reply = await self._invoke_with_backoff(gs["model"], gs["messages"])
            rec = gs["record"]
            usage = getattr(reply, "usage_metadata", None) or {}
            rec["input_tokens"] += usage.get("input_tokens", 0) or 0
            rec["output_tokens"] += usage.get("output_tokens", 0) or 0
            meta = getattr(reply, "response_metadata", None) or {}
            rec["stop_reason"] = meta.get("finish_reason") or rec.get("stop_reason")

            text = reply.content if isinstance(reply.content, str) else ""
            if text.strip():
                rec["_reasoning_parts"].append(text.strip())
            return {"messages": [reply], "iterations": gs.get("iterations", 0) + 1}

        async def act(gs: dict) -> dict:
            """Execute every tool the model asked for, in order."""
            rec = gs["record"]
            last = gs["messages"][-1]
            out: list[Any] = []

            for call in (getattr(last, "tool_calls", None) or []):
                name = call.get("name")
                payload = call.get("args") or {}
                outcome = await asyncio.to_thread(self._execute_tool, name, payload)

                rec["tool_calls"].append(
                    {"tool": name, "input": payload, "ok": outcome["ok"]}
                )
                rec["tool_results"].append({
                    "tool": name, "ok": outcome["ok"],
                    "result": outcome.get("result"), "error": outcome.get("error"),
                })
                out.append(ToolMessage(
                    tool_call_id=call.get("id"),
                    name=name,
                    content=json.dumps(
                        outcome.get("result") if outcome["ok"]
                        else {"error": outcome.get("error")},
                        default=str,
                    ),
                ))
            return {"messages": out}

        def should_continue(gs: dict) -> str:
            """Loop only while the model is still asking for tools."""
            last = gs["messages"][-1]
            if not getattr(last, "tool_calls", None):
                return END
            if gs.get("iterations", 0) >= MAX_ITERATIONS:
                log.warning("agent hit MAX_ITERATIONS=%d", MAX_ITERATIONS)
                return END
            return "act"

        graph = StateGraph(_GraphState)
        graph.add_node("perceive", perceive)
        graph.add_node("reason", reason)
        graph.add_node("act", act)
        graph.set_entry_point("perceive")
        graph.add_edge("perceive", "reason")
        graph.add_conditional_edges("reason", should_continue, {"act": "act", END: END})
        graph.add_edge("act", "reason")
        return graph.compile()

    async def _reason_and_act(
        self,
        state: dict[str, Any] | None,
        trigger: str,
        record: dict[str, Any],
        cases: list[dict[str, Any]] | None = None,
    ) -> None:
        """
        Run one decision through the LangGraph state graph.

        The signature is unchanged so the probes keep working untouched. When a
        probe supplies `state` and `cases`, the perceive node passes them through
        rather than re-reading the database -- that is what keeps a controlled
        A/B byte-identical across arms.
        """
        tool_defs = schemas.tool_schemas(read_only=READ_ONLY, dialect=DIALECT)

        if self._graph is None:
            self._graph = self._build_graph()

        record["_reasoning_parts"] = []
        final = await self._graph.ainvoke(
            {
                "trigger": trigger,
                "record": record,
                "model": self._build_model(tool_defs),
                "iterations": 0,
                "messages": [],
                "perceived": state,
                "cases": cases,
            },
            config={"recursion_limit": MAX_ITERATIONS * 4},
        )

        # Read the cases back off the FINAL graph state, not the argument. On the
        # production path they are retrieved inside the perceive node, so the
        # incoming `cases` is None -- using it would silently pin cites_memory to
        # False for every live decision while still looking correct in the probes.
        effective_cases = (final or {}).get("cases") or cases

        record["reasoning"] = "\n\n".join(record.pop("_reasoning_parts", [])) or None
        record["cites_memory"] = _cites_memory(record["reasoning"], effective_cases)

    @staticmethod
    def _execute_tool(name: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Dispatch one tool call. Tool failures are results, not exceptions."""
        fn = schemas.TOOL_REGISTRY.get(name)
        if fn is None:
            return {"ok": False, "error": f"Unknown tool {name!r}"}
        if READ_ONLY and name not in schemas.READ_ONLY_NAMES:
            return {"ok": False, "error": f"{name} is disabled: agent is in read-only mode"}

        try:
            return {"ok": True, "result": fn(**payload)}
        except ToolError as exc:
            return {"ok": False, "error": str(exc)}
        except TypeError as exc:
            return {"ok": False, "error": f"Invalid arguments for {name}: {exc}"}
        except Exception as exc:  # noqa: BLE001
            log.exception("tool %s raised", name)
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    @staticmethod
    def _persist(record: dict[str, Any]) -> dict[str, Any]:
        with SessionLocal() as session:
            row = AgentDecision(
                created_at=_now(),
                trigger=record["trigger"],
                input_state=record["input_state"],
                reasoning=record["reasoning"],
                tool_calls=record["tool_calls"],
                tool_results=record["tool_results"],
                action_count=record["action_count"],
                no_action=record["no_action"],
                model=record["model"],
                input_tokens=record["input_tokens"],
                output_tokens=record["output_tokens"],
                latency_ms=record["latency_ms"],
                stop_reason=record["stop_reason"],
                error=record["error"],
                signature=record.get("signature"),
                retrieved_case_count=record.get("retrieved_case_count", 0),
                retrieved_case_ids=record.get("retrieved_case_ids"),
                cites_memory=bool(record.get("cites_memory")),
            )
            session.add(row)
            session.commit()
            return serialize_decision(row)

    # -- health ------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        return {
            "enabled": ENABLED,
            "running": self.running,
            "model": MODEL,
            "provider": PROVIDER,
            "base_url": GROQ_BASE_URL,
            "temperature": TEMPERATURE,
            "read_only": READ_ONLY,
            "trigger_policy": {
                "poll_seconds": POLL_SECONDS,
                "min_interval_seconds": MIN_INTERVAL_SECONDS,
                "heartbeat_seconds": HEARTBEAT_SECONDS,
            },
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "last_poll_at": self.last_poll_at.isoformat() if self.last_poll_at else None,
            "last_decision_at": (
                self.last_decision_at.isoformat() if self.last_decision_at else None
            ),
            "decisions": self.decisions,
            "no_action_decisions": self.no_action_decisions,
            "actions_taken": self.actions_taken,
            "failures": self.failures,
            "memory": {
                "enabled": MEMORY_ENABLED,
                "cases_per_decision": MEMORY_CASES,
                "outcome_window_seconds": OUTCOME_WINDOW_SECONDS,
                "cases_retrieved_total": self.cases_retrieved_total,
                "decisions_scored": self.decisions_scored,
            },
            "backoff": {
                "consecutive_failures": self._consecutive_failures,
                "skip_polls_remaining": self._skip_polls_remaining,
                "cycles_skipped": self.cycles_skipped,
                "max_backoff_cycles": FAILURE_BACKOFF_MAX_CYCLES,
                "tpd_cooldown_hours": TPD_COOLDOWN_HOURS,
                "tpd_blocked": self._tpd_blocked_until is not None,
                "tpd_blocked_until": (
                    self._tpd_blocked_until_wall.isoformat()
                    if self._tpd_blocked_until_wall else None
                ),
            },
            "last_error": self.last_error,
            "unavailable_reason": self.unavailable_reason,
        }


def serialize_decision(row: AgentDecision) -> dict[str, Any]:
    """Wire format for the WebSocket and for scripts. Omits the bulky input_state."""
    return {
        "id": row.id,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "trigger": row.trigger,
        "reasoning": row.reasoning,
        "tool_calls": row.tool_calls or [],
        "tool_results": row.tool_results or [],
        "action_count": row.action_count,
        "no_action": row.no_action,
        "model": row.model,
        "latency_ms": row.latency_ms,
        "stop_reason": row.stop_reason,
        "retrieved_case_count": row.retrieved_case_count,
        "retrieved_case_ids": row.retrieved_case_ids or [],
        "cites_memory": row.cites_memory,
        "input_tokens": row.input_tokens,
        "output_tokens": row.output_tokens,
        "error": row.error,
    }


agent_service = AgentService()
