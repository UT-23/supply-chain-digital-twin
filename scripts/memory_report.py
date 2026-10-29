"""
Early-vs-late analysis over a real run: did behaviour change as memory accumulated?

    python scripts/memory_report.py
    python scripts/memory_report.py --min-similarity 0.8

Reads the agent_decisions / decision_outcomes tables produced by an actual server
run. Needs no API key -- it analyses decisions already recorded.

It does three things:

 1. Summarises outcomes by condition type, separating conditions the agent could
    influence from ones that resolve regardless.
 2. Finds the most similar EARLY/LATE pairs of decisions -- same kind of
    situation, one before memory had anything to offer, one after -- and prints
    both so you can read whether the reasoning changed.
 3. Reports whether action rate moved as retrieved-case count grew.

Caveat worth keeping in mind while reading the output: early-vs-late over a live
run is observational. Conditions drift, so a difference may be caused by the
situation rather than by memory. For a controlled comparison use
scripts/memory_ab_probe.py, which holds the situation fixed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.orm import selectinload

# Windows consoles default to cp1252, which cannot encode much of what a model
# writes (dashes, arrows, degree signs). Force UTF-8 so printing results never
# crashes the run after the API calls have already been paid for.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from agent import memory  # noqa: E402
from db.models import AgentDecision, DecisionOutcome  # noqa: E402
from db.session import SessionLocal, engine  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-similarity", type=float, default=0.7)
    ap.add_argument("--pairs", type=int, default=3)
    args = ap.parse_args()

    with SessionLocal() as s:
        total = s.scalar(select(func.count()).select_from(AgentDecision)) or 0
        reasoned = s.scalar(
            select(func.count()).select_from(AgentDecision)
            .where(AgentDecision.error.is_(None))) or 0
        scored = s.scalar(
            select(func.count()).select_from(AgentDecision)
            .where(AgentDecision.outcome_scored_at.is_not(None))) or 0

        print("=" * 72)
        print("AGENT RUN SUMMARY")
        print("=" * 72)
        print(f"  decisions recorded : {total}")
        print(f"  reasoned (no error): {reasoned}")
        print(f"  outcome-scored     : {scored}")

        if reasoned == 0:
            print("\nNo decisions with actual reasoning were recorded.")
            print("Every cycle errored -- most likely GROQ_API_KEY is unset.")
            print("There is nothing to analyse until the agent has really run.")
            engine.dispose()
            return 1

        # ---- outcomes by condition type
        print("\n" + "=" * 72)
        print("OUTCOMES BY CONDITION TYPE")
        print("=" * 72)
        rows = s.execute(
            select(
                DecisionOutcome.subject_type,
                DecisionOutcome.attribution,
                DecisionOutcome.label,
                func.count(),
            ).group_by(
                DecisionOutcome.subject_type,
                DecisionOutcome.attribution,
                DecisionOutcome.label,
            ).order_by(DecisionOutcome.subject_type)
        ).all()
        if not rows:
            print("  none scored yet -- let the run continue past the outcome window")
        for subject, attribution, label, n in rows:
            flag = "" if attribution == "agent_influenceable" else "   <- agent cannot influence"
            print(f"  {subject:<18} {label:<24} n={n}{flag}")

        # ---- action rate vs memory depth
        print("\n" + "=" * 72)
        print("ACTION RATE vs RETRIEVED CASES")
        print("=" * 72)
        buckets = s.execute(
            select(
                AgentDecision.retrieved_case_count,
                func.count(),
                func.sum(func.cast(AgentDecision.no_action, __import__("sqlalchemy").Integer)),
            )
            .where(AgentDecision.error.is_(None))
            .group_by(AgentDecision.retrieved_case_count)
            .order_by(AgentDecision.retrieved_case_count)
        ).all()
        for cases, n, no_action in buckets:
            no_action = no_action or 0
            rate = (n - no_action) / n if n else 0
            print(f"  {cases} case(s) retrieved: {n:>3} decisions, acted in "
                  f"{n - no_action:>3} ({rate:.0%})")
        if len(buckets) < 2:
            print("  Not enough variation in retrieved-case count to compare.")

        # ---- early vs late matched pairs
        print("\n" + "=" * 72)
        print("MATCHED EARLY / LATE PAIRS")
        print("=" * 72)
        decisions = s.scalars(
            select(AgentDecision)
            .options(selectinload(AgentDecision.outcomes))
            .where(AgentDecision.error.is_(None), AgentDecision.signature.is_not(None))
            .order_by(AgentDecision.id)
        ).all()

        if len(decisions) < 2:
            print("  need at least 2 reasoned decisions")
            engine.dispose()
            return 0

        pairs = []
        for i, early in enumerate(decisions):
            for late in decisions[i + 1:]:
                # Only interesting when memory actually had something to offer later.
                if late.retrieved_case_count <= early.retrieved_case_count:
                    continue
                score = memory.similarity(early.signature or {}, late.signature or {})
                if score >= args.min_similarity:
                    pairs.append((score, early, late))

        pairs.sort(key=lambda p: (-p[0], -(p[2].id - p[1].id)))
        if not pairs:
            print(f"  No pairs above similarity {args.min_similarity} where the later")
            print("  decision had more retrieved cases. Either the run is too short,")
            print("  or situations did not repeat. Try --min-similarity 0.5.")

        for score, early, late in pairs[:args.pairs]:
            print(f"\n{'-' * 72}")
            print(f"similarity {score:.2f}   situation: "
                  f"{(early.signature or {}).get('event_types')}")
            for tag, d in (("EARLY", early), ("LATE ", late)):
                tools = [c.get("tool") for c in (d.tool_calls or [])] or ["(no action)"]
                print(f"\n  {tag} decision #{d.id}  "
                      f"cases_retrieved={d.retrieved_case_count} "
                      f"ids={d.retrieved_case_ids or []}")
                print(f"        actions: {tools}")
                print(f"        reasoning: {(d.reasoning or '')[:420]}")
            changed = (
                [c.get("tool") for c in (early.tool_calls or [])]
                != [c.get("tool") for c in (late.tool_calls or [])]
            )
            print(f"\n  => action changed: {'YES' if changed else 'no'}")
            cites = any(
                f"#{cid}" in (late.reasoning or "")
                for cid in (late.retrieved_case_ids or [])
            )
            print(f"  => later reasoning cites a retrieved case: {'YES' if cites else 'no'}")

        print("\n" + "=" * 72)
        print("Read the reasoning yourself. 'action changed' alone is weak evidence --")
        print("the model varies between identical calls. What would be convincing is")
        print("the later reasoning explicitly citing a past case and saying why it")
        print("changed the call. For a controlled test, use memory_ab_probe.py.")

    engine.dispose()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
