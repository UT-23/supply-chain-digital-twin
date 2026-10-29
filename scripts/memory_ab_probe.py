"""
Does memory change what the agent does? A controlled A/B.

Runs on one provider per invocation:

    python scripts/memory_ab_probe.py --provider groq  --rpd 1000
    python scripts/memory_ab_probe.py --provider gemini --rpd 20 --rpm 5

Needs that provider's key (GROQ_API_KEY / GEMINI_API_KEY). Touches no live data:
the state is synthetic, the cases are synthetic, and tool execution is stubbed.

All three arms of a run use one provider and one model. Mixing providers within
a run would confound the thing being measured with the model answering.

## Spend guard

Daily free-tier limits must be passed in (--rpd / --tpd), read from the
provider's own dashboard. They are never guessed: neither Groq nor Google
publishes reliable per-model free-tier numbers that can be relied on in code,
and Gemini's OpenAI-compatible endpoint returns no x-ratelimit-* headers. The
run aborts before crossing --stop-fraction (default 0.8) of any daily limit and
prints exactly what it consumed.

## Why this rather than "early vs late in a run"

Comparing an early decision to a later one confounds everything that drifted in
between -- different readings, different shipments, a different time of day in
the simulation. This holds the situation BYTE-IDENTICAL and varies exactly one
thing: whether past cases are in the prompt, and what those cases say.

Three arms per scenario:

  A  no memory           -- the control; how it decides with no history
  B  history says benign -- 3 prior cases, same situation, no action taken,
                            condition cleared on its own every time
  C  history says real   -- 3 prior cases, same situation, ignored, and the
                            condition persisted or worsened every time

If memory does nothing, all three arms decide the same way. If memory works, B
should make action less likely than A, and C should make it more likely.

## What a null result means

If B == C, say so. It would mean the retrieved cases are not influencing the
model -- the memory is recording and retrieving correctly (that is separately
tested) but not changing behaviour, which is exactly the thing a "self-evolving"
claim would need. A null result here is a real finding, not a failed script.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

# Windows consoles default to cp1252, which cannot encode much of what a model
# writes (dashes, arrows, degree signs). Force UTF-8 so printing results never
# crashes the run after the API calls have already been spent.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

BAND = {"low": 40.0, "high": 64.0}


def ambiguous_state() -> dict:
    """
    A situation deliberately placed NEAR the agent's decision boundary.

    This matters more than it sounds. The first version of this fixture read
    69.4% humidity, carried severity="critical", and sat at a warehouse holding
    3,400 units. Every arm then escalated in 100% of runs -- the model anchored
    on the word "critical" and said so in its reasoning. With every arm pinned at
    100% the test had ZERO DYNAMIC RANGE: memory could not have shifted the
    outcome even if it worked perfectly, so the null result meant nothing.

    Three changes pull it back toward the boundary:
      * severity "critical" -> "warning". A pre-applied label does the model's
        judging for it. "warning" is what the simulation actually attaches to
        organically-arising conditions, so this stays representative.
      * humidity 69.4% -> 66.1%, i.e. ~2 points over a 64% ceiling rather than
        ~5. Still outside the band, but arguably within drift.
      * on-site inventory 3,400 -> 180 units, so "spoilage risk to valuable
        stock" is no longer an automatic argument for escalating.

    If the no-memory arm comes back at 0% or 100% again, this needs further
    tuning BEFORE spending a day's quota on the full three-arm run --
    `--arms A` exists for exactly that pilot check.
    """
    return {
        "observed_at": "2026-09-13T16:00:00+00:00",
        "sites": [{
            "warehouse_id": 1, "code": "WH1", "name": "Warehouse 1", "kind": "warehouse",
            "inventory_units_on_site": 180,
            "readings": [{
                "metric": "humidity", "current": 66.1, "unit": "%",
                "recent_values_oldest_first": [65.4, 65.9, 65.6, 66.3, 65.8, 66.1],
                "normal_band": BAND,
                "last_recorded_at": "2026-09-13T16:00:00+00:00",
            }, {
                "metric": "temperature", "current": 22.4, "unit": "C",
                "recent_values_oldest_first": [22.1, 22.3, 22.4, 22.2, 22.5, 22.4],
                "normal_band": {"low": 16.5, "high": 27.5},
                "last_recorded_at": "2026-09-13T16:00:00+00:00",
            }],
        }],
        "shipments": {"counts_by_status": {"IN_TRANSIT": 32}, "notable_active": []},
        "inventory": {"locations_out_of_stock": 0, "lowest_stock": []},
        "unhandled_events": [{
            "event_id": 7001, "occurred_at": "2026-09-13T15:59:30+00:00", "age_seconds": 30.0,
            "type": "SENSOR_SPIKE", "severity": "warning",
            "entity_type": "warehouse", "entity_id": 1,
            "message": "humidity anomaly at WH1 (~66.1)",
            "details": {"warehouse_code": "WH1", "reading_type": "humidity",
                        "approx_value": 66.1, "normal_range": [40.0, 64.0],
                        "duration_cycles": 4},
        }],
        "open_alerts": [], "recent_agent_actions": [],
    }


# Fingerprint of the retuned situation. max_severity 1 = "warning" (was 2 =
# "critical"); the excursion bucket is still "above" since 66.1 sits ~9% past a
# 24-point band, inside the same bucket the old 69.4 fell in.
SIG = {
    "event_types": ["SENSOR_SPIKE"], "max_severity": 1,
    "sensors": [{"metric": "humidity", "excursion": "above", "trend": "flat",
                 "site_kind": "warehouse"}],
    "shipment": None, "stock_out_count": "none",
}


def _case(cid: int, took_action: bool, status: str, label: str, influenceable: bool):
    return {
        "case_id": cid, "similarity": 0.97, "age_hours": round(cid * 0.7, 1),
        "situation": SIG,
        "action_taken": None if not took_action else [
            {"tool": "escalate_alert", "input": {"severity": "critical"}}],
        "took_no_action": not took_action,
        "outcomes": [{
            "condition": "SENSOR_SPIKE", "what_happened": status,
            "verdict": label, "agent_could_influence": influenceable,
        }],
    }


# Arm B: ignored three times, cleared by itself three times.
BENIGN_CASES = [_case(i, False, "cleared", "self_resolved", True) for i in (41, 42, 43)]
# Arm C: ignored three times, persisted/worsened every time.
SERIOUS_CASES = [
    _case(51, False, "persisted", "missed", True),
    _case(52, False, "worsened", "missed", True),
    _case(53, False, "persisted", "missed", True),
]

ARMS = {
    "A_no_memory": None,
    "B_history_benign": BENIGN_CASES,
    "C_history_serious": SERIOUS_CASES,
}


async def run_arm(cases, repeats: int) -> list[dict]:
    from agent import service as svc

    out = []
    for _ in range(repeats):
        agent = svc.AgentService()
        record = {
            "trigger": "probe", "input_state": None, "reasoning": None,
            "tool_calls": [], "tool_results": [], "model": svc.MODEL,
            "provider": svc.PROVIDER,
            "error": None, "stop_reason": None, "input_tokens": 0, "output_tokens": 0,
        }
        try:
            agent._client = svc._build_client()
            await agent._reason_and_act(ambiguous_state(), "events", record, cases)
        except svc.BudgetExhausted:
            raise                      # stops the whole probe, never swallowed
        except Exception as exc:       # noqa: BLE001
            record["error"] = f"{type(exc).__name__}: {exc}"
        out.append(record)
    return out


def acted(record: dict) -> bool:
    """Did it take a real action? Marking an event handled is bookkeeping, not action."""
    return any(c["tool"] != "mark_event_handled" for c in record["tool_calls"])


async def main_async(args) -> int:
    repeats = args.repeats

    # NOTE: do NOT set AGENT_READ_ONLY here -- it hides every write tool from the
    # model, so the action rate this probe measures would be pinned at zero by
    # construction. Safety comes from the stubbed _execute_tool below.
    from agent import service as svc

    try:
        svc._build_client()
    except svc.AgentUnavailable as exc:
        print(f"Cannot run the A/B: {exc}\n")
        print("This probe is the evidence for whether memory changes behaviour.")
        print("It cannot be simulated -- it needs real model calls.")
        print("Set the key in backend/.env and re-run, or pass --provider to")
        print("use a provider whose quota is not exhausted.")
        return 2

    svc.AgentService._execute_tool = staticmethod(
        lambda name, payload: {"ok": True, "result": {"stubbed": True, "tool": name}}
    )

    # ---- spend guard ----
    budget = None
    if not args.no_budget:
        if args.rpd is None and args.tpd is None:
            print("Refusing to run without a spend guard.\n")
            print("Pass the provider's free-tier DAILY limits, read from its own")
            print("dashboard -- these are never guessed. For example:")
            print("    --rpd 20            (requests/day)")
            print("    --rpd 20 --tpd 1000000")
            print("Google AI Studio limits: https://aistudio.google.com/rate-limit")
            print("Or pass --no-budget to run unguarded (not advised on a free tier).")
            return 4

        budget = svc.RateBudget(
            max_requests_per_day=args.rpd,
            max_tokens_per_day=args.tpd,
            min_seconds_between_calls=(60.0 / args.rpm) if args.rpm else 0.0,
            stop_fraction=args.stop_fraction,
            tokens_already_used=args.tokens_used_today,
        )
        svc.set_budget(budget)

        # Refuse to start a run that provably cannot finish inside the ceiling,
        # rather than burning most of a daily quota to find out.
        n_arms = len({a.strip().upper() for a in args.arms.split(',')}) if args.arms else len(ARMS)
        needed = repeats * n_arms * 3            # measured ~2.8 calls/run
        if args.tpd is not None:
            est_tokens = needed * 4200            # measured ~4.1k per call
            room = args.tpd * args.stop_fraction - args.tokens_used_today
            if est_tokens > room:
                print("Refusing to start: not enough daily token headroom.")
                print(f"  needs about   : {est_tokens:,} tokens ({needed} calls x ~4,200)")
                print(f"  already used  : {args.tokens_used_today:,}")
                print(f"  headroom to {args.stop_fraction:.0%}: {room:,.0f}")
                print("  Wait for the daily window to roll, or use a provider with headroom.")
                return 4
        if args.rpd is not None and needed > args.rpd * args.stop_fraction:
            print("Refusing to start: this run cannot finish inside the ceiling.")
            print(f"  needs about  : {needed} requests "
                  f"({repeats} repeats x {n_arms} arm(s) x ~3 calls)")
            print(f"  ceiling      : {args.rpd * args.stop_fraction:.0f} "
                  f"({args.stop_fraction:.0%} of {args.rpd} requests/day)")
            print("  Lower --repeats, raise the limit, or use a provider with headroom.")
            return 4

    print(f"provider : {svc.PROVIDER}")
    print(f"model    : {svc.MODEL}")
    print(f"repeats  : {repeats} per arm")
    if budget is not None:
        print(f"budget   : stop at {args.stop_fraction:.0%} of "
              f"{args.rpd if args.rpd is not None else 'n/a'} req/day / "
              f"{args.tpd if args.tpd is not None else 'n/a'} tok/day"
              + (f", calls spaced {60.0 / args.rpm:.1f}s apart" if args.rpm else ""))
    print()

    selected = ARMS
    if args.arms:
        wanted = {a.strip().upper() for a in args.arms.split(",") if a.strip()}
        selected = {k: v for k, v in ARMS.items() if k[0].upper() in wanted}
        if not selected:
            print(f"No arms matched {args.arms!r}. Valid: A, B, C")
            return 4
        print(f"arms     : {', '.join(selected)}  (PILOT -- not a full experiment)")

    results: dict[str, list[dict]] = {}
    stopped_early = None
    for arm, cases in selected.items():
        print(f"running {arm} x{repeats} ...")
        try:
            results[arm] = await run_arm(cases, repeats)
        except svc.BudgetExhausted as exc:
            stopped_early = str(exc)
            results.setdefault(arm, [])
            break

    out_path = ROOT / "memory_ab_results.json"

    if stopped_early:
        s = budget.summary() if budget else {}
        print("\n" + "=" * 72)
        print("STOPPED: FREE-TIER BUDGET GUARD TRIPPED")
        print("=" * 72)
        print(f"  reason                 : {stopped_early}")
        print(f"  requests used this run : {s.get('requests_used')}")
        print(f"  tokens used this run   : {s.get('tokens_used', 0):,}")
        print(f"  request stop-ceiling   : {s.get('requests_ceiling')}")
        print(f"  token stop-ceiling     : {s.get('tokens_ceiling')}")
        print("\n  Aborted before crossing into paid usage.")
        print("  No verdict is rendered from a partial run.")
        out_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
        print(f"\n  partial transcripts: {out_path}")
        return 5

    if budget is not None:
        s = budget.summary()
        print(f"\nbudget used: {s['requests_used']} requests, {s['tokens_used']:,} tokens")

    print("\n" + "=" * 72)
    print("IDENTICAL SITUATION, DIFFERENT HISTORY")
    # Derived from the fixture so it can never drift out of sync with it again.
    _st = ambiguous_state()
    _site = _st["sites"][0]
    _hum = next(r for r in _site["readings"] if r["metric"] == "humidity")
    _ev = _st["unhandled_events"][0]
    print(f"humidity {_hum['current']}% (band {_hum['normal_band']['low']:.0f}-"
          f"{_hum['normal_band']['high']:.0f}), severity={_ev['severity']}, "
          f"{_site['inventory_units_on_site']} units on site")
    print("=" * 72)

    # Rates are computed over VALID runs only. A failed API call is not a
    # decision to take no action, and counting it as one once turned a single
    # 429 into a confident "memory works BACKWARDS" verdict.
    summary = {}
    valid_counts = {}
    for arm, runs in results.items():
        ok = [r for r in runs if not r["error"]]
        rates = [acted(r) for r in ok]
        summary[arm] = sum(rates)
        valid_counts[arm] = len(ok)
        failed_here = len(runs) - len(ok)
        print(f"\n{arm}")
        print(f"  acted in {sum(rates)}/{len(ok)} VALID runs"
              + (f"  ({failed_here} errored, excluded)" if failed_here else ""))
        for i, r in enumerate(runs):
            if r["error"]:
                print(f"  run {i}: ERROR {r['error'][:180]}")
                continue
            tools = [c["tool"] for c in r["tool_calls"]] or ["(no action)"]
            print(f"  run {i}: {tools}")
            print(f"          {(r['reasoning'] or '')[:300]}")

    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)
    # A pilot (--arms A) is not a three-arm experiment; report the rate and stop
    # rather than printing "no comparison is possible" as if something failed.
    if len(results) < len(ARMS):
        print("PILOT ONLY -- no A/B comparison attempted.")
        for arm, runs in results.items():
            ok = [r for r in runs if not r["error"]]
            k = sum(1 for r in ok if acted(r))
            pct = f"{k / len(ok):.0%}" if ok else "n/a"
            print(f"  {arm}: acted in {k}/{len(ok)} valid runs ({pct})")
            if ok and 0 < k < len(ok):
                print("  -> Decisions DIVERGED on identical input: the scenario sits")
                print("     near the decision boundary, so a memory effect could show.")
            elif ok:
                print(f"  -> Pinned at {pct}: zero dynamic range. Retune the scenario")
                print("     before spending quota on the full three-arm run.")
        out_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
        print(f"\n  transcripts: {out_path}")
        return 0

    a = summary.get("A_no_memory", 0)
    b = summary.get("B_history_benign", 0)
    c = summary.get("C_history_serious", 0)
    na = valid_counts.get("A_no_memory", 0)
    nb = valid_counts.get("B_history_benign", 0)
    nc = valid_counts.get("C_history_serious", 0)
    print(f"  action rate  A(no memory)={a}/{na}  "
          f"B(benign history)={b}/{nb}  C(serious history)={c}/{nc}   (valid runs only)")

    # Compare proportions, not raw counts -- arms can have different denominators
    # once errored runs are excluded.
    def _rate(k, n):
        return (k / n) if n else None

    ra, rb, rc = _rate(a, na), _rate(b, nb), _rate(c, nc)
    if None in (ra, rb, rc):
        print("\n  An arm has no valid runs at all; no comparison is possible.")
        out_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
        return 3

    if min(na, nb, nc) < 2:
        print(f"\n  WARNING: an arm has only {min(na, nb, nc)} valid run(s). Any")
        print("  difference below is indistinguishable from model variance.")

    if ra == rb == rc and ra in (0.0, 1.0):
        print(f"\n  NULL RESULT, and the test had ZERO DYNAMIC RANGE: every arm")
        print(f"  acted in {ra:.0%} of valid runs, so this scenario cannot")
        print("  discriminate between them even in principle. The retrieved cases")
        print("  made no difference -- but neither would anything else, because the")
        print("  scenario does not sit near the agent's decision boundary.")
        print("\n  Check the reasoning text: if it cites the past cases and then")
        print("  overrides them, memory is reaching the model but not moving the")
        print("  decision. That is a real and reportable finding.")
        out_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
        print(f"\n  full transcripts: {out_path}")
        return 0

    errored = sum(1 for runs in results.values() for r in runs if r["error"])
    total_runs = sum(len(runs) for runs in results.values())
    if errored:
        print(f"\n  {errored}/{total_runs} run(s) FAILED (see errors above).")
        # A failed call is not a decision not to act. Counting it as one would
        # manufacture a null result out of an infrastructure problem.
        if errored * 2 >= total_runs:
            print("  REFUSING TO RENDER A VERDICT: too many runs failed for the")
            print("  numbers above to mean anything. Fix the failures and re-run.")
            out_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
            print(f"\n  full transcripts: {out_path}")
            return 3
        print("  Treat the rates below as provisional.")

    # B vs C is the hypothesis: benign history should suppress action relative to
    # a history of the condition persisting. EQUAL rates are a NULL result, not a
    # pass -- an earlier version of this check used `c >= b` and called ties wins.
    if rb == rc:
        print("\n  NULL RESULT. Benign history and serious history produced the")
        print("  same action rate, so the content of the retrieved cases made no")
        print("  measurable difference to what the agent did.")
        if ra == rb == rc:
            print("  Memory presence made no difference either.")
        print("  Memory is recorded and retrieved correctly (test_agent_memory.py")
        print("  covers that), but it is not changing behaviour. On this evidence")
        print("  an 'adaptive' or 'self-evolving' claim is NOT supported.")
        if max(ra, rb, rc) == 0:
            print("\n  NOTE: the agent acted in NO arm. With zero dynamic range this")
            print("  test cannot detect an effect even if one exists -- fix the")
            print("  action rate before drawing any conclusion from it.")
    elif rc > rb:
        print("\n  Memory shifted behaviour in the expected direction: a history of")
        print("  the condition resolving itself made action less likely, and a")
        print("  history of it persisting made action more likely.")
        print("  Check the reasoning text above actually cites the past cases.")
    else:
        print("\n  Memory changed behaviour, but BACKWARDS: a benign history made")
        print("  action MORE likely than a serious one. That is worse than a null")
        print("  result -- it suggests the cases are being misread.")

    if repeats < 3:
        print("\n  Run with --repeats 3 or more: with fewer runs per arm you cannot")
        print("  tell a real effect from ordinary model variation.")

    out_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\n  full transcripts: {out_path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--provider", default=None, choices=("groq", "gemini"),
                    help="which LLM provider to run the WHOLE experiment on")
    ap.add_argument("--model", default=None, help="override the provider's default model")
    ap.add_argument("--rpd", type=int, default=None,
                    help="provider's free-tier REQUESTS-PER-DAY limit, from your "
                         "provider dashboard (never guessed)")
    ap.add_argument("--tpd", type=int, default=None,
                    help="provider's free-tier TOKENS-PER-DAY limit, if it has one")
    ap.add_argument("--rpm", type=float, default=None,
                    help="provider's free-tier REQUESTS-PER-MINUTE limit; calls are "
                         "spaced out to stay under it")
    ap.add_argument("--stop-fraction", type=float, default=0.8,
                    help="abort before crossing this fraction of a daily limit")
    ap.add_argument("--arms", default=None,
                    help="comma-separated arm letters to run (e.g. 'A' for a "
                         "cheap pilot of the no-memory control). Default: all.")
    ap.add_argument("--tokens-used-today", type=int, default=0,
                    help="tokens the provider says are already spent today; read "
                         "from a 429 body (\"Used N\") so the guard starts from the "
                         "real position rather than zero")
    ap.add_argument("--no-budget", action="store_true",
                    help="run without a spend guard (not advised on a free tier)")
    args = ap.parse_args()

    # Must be set before agent.service is imported: it reads provider config at
    # import time. Every arm of one run therefore uses one provider and one
    # model -- mixing them within a run would make the comparison meaningless.
    if args.provider:
        os.environ["AGENT_PROVIDER"] = args.provider
        os.environ.pop("AGENT_MODEL", None)
    if args.model:
        os.environ["AGENT_MODEL"] = args.model

    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
