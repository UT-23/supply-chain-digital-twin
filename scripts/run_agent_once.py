"""
Run exactly one agent decision cycle against the dev database and print the transcript.

    python scripts/run_agent_once.py
    python scripts/run_agent_once.py --read-only   # agent may observe, not mutate
    python scripts/run_agent_once.py --show-state  # also dump what it was shown

Requires GROQ_API_KEY in backend/.env. Without it the cycle still runs and
still records a decision row, but with `error` set and no reasoning -- which is
the honest outcome, not a simulated one.
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
# crashes the run after the API calls have already been paid for.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--read-only", action="store_true")
    ap.add_argument("--show-state", action="store_true")
    ap.add_argument("--trigger", default="manual")
    args = ap.parse_args()

    if args.read_only:
        os.environ["AGENT_READ_ONLY"] = "1"

    from agent.service import AgentService  # imported after env is set
    from db.session import engine

    service = AgentService()

    async def go():
        return await service.run_cycle(trigger=args.trigger)

    decision = asyncio.run(go())

    if args.show_state:
        # serialize_decision omits the bulky snapshot; read it back from the row.
        from sqlalchemy import select
        from db.models import AgentDecision
        from db.session import SessionLocal
        with SessionLocal() as s:
            with_state = s.get(AgentDecision, decision["id"]).input_state
        print("=" * 70)
        print("STATE SHOWN TO THE MODEL")
        print("=" * 70)
        print(json.dumps(with_state, indent=2, default=str)[:6000])

    print("=" * 70)
    print(f"DECISION #{decision['id']}   trigger={decision['trigger']}   "
          f"model={decision['model']}")
    print("=" * 70)

    if decision.get("error"):
        print(f"\nERROR: {decision['error']}")
        print("\nNo reasoning was produced. This row is recorded so a broken agent")
        print("is visible in the log rather than silently absent.")
        engine.dispose()
        return 1

    print("\n--- REASONING (the model's own words) ---")
    print(decision.get("reasoning") or "(none)")

    calls = decision.get("tool_calls") or []
    results = decision.get("tool_results") or []
    print(f"\n--- ACTIONS ({len(calls)}) ---")
    if not calls:
        print("none -- the agent reviewed conditions and chose not to act")
    for call, result in zip(calls, results):
        status = "ok" if call["ok"] else "FAILED"
        print(f"\n  {call['tool']}  [{status}]")
        print(f"    input:  {json.dumps(call['input'], default=str)[:400]}")
        payload = result.get("result") if call["ok"] else result.get("error")
        print(f"    result: {json.dumps(payload, default=str)[:400]}")

    print(f"\n--- COST ---")
    print(f"  latency: {decision['latency_ms']} ms")
    print(f"  tokens:  {decision['input_tokens']} in / {decision['output_tokens']} out")
    print(f"  stop_reason: {decision['stop_reason']}")

    engine.dispose()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
