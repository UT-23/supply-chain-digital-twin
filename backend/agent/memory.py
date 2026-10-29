"""
Case memory: find past decisions that looked like the current situation, and what
came of them.

## Why no embeddings

You asked whether vector search is worth it. It is not, here, and I would not add
it later either:

  * The state space is tiny and closed. Three event types, four sites, six
    metrics. A structured fingerprint covers it exhaustively; there is no long
    tail of phrasings for an embedding to generalize over.
  * The thing that makes two situations similar is numeric and relational -- how
    far outside its band a reading is, how far along a shipment is -- and
    embeddings are bad at magnitude. "humidity 71%" and "humidity 61%" embed
    almost identically while meaning opposite things.
  * A structured signature is inspectable. When retrieval returns a bad case you
    can see exactly which component matched. With embeddings you get a cosine
    score and a shrug, which is a poor foundation for a claim about adaptation.

Embeddings would start to earn their place if situations arrived as free text
(operator notes, supplier emails), or if the taxonomy grew open-ended.

## The fingerprint

Values are bucketed rather than exact, so "humidity 71.2%" and "humidity 72.8%"
are the same case. Buckets are relative to each metric's normal band, so a metric
with a wide band is not treated as more volatile than one with a narrow band.
"""
from __future__ import annotations

import datetime as dt
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from db.models import AgentDecision

# Only surface cases whose outcome is known; an unscored case teaches nothing.
DEFAULT_LIMIT = 4
MIN_SIMILARITY = 0.34
# Outcome rows rendered per case in the prompt; the rest are tallied.
MAX_OUTCOMES_PER_CASE = 4

SEVERITY_RANK = {"info": 0, "warning": 1, "critical": 2}


# -- fingerprinting --------------------------------------------------------


def _excursion_bucket(value: float, band: dict | None) -> str:
    """How far outside its normal band a reading sits, as a coarse band."""
    if not band:
        return "unknown"
    low, high = float(band["low"]), float(band["high"])
    width = max(high - low, 1e-6)

    if value > high:
        over = (value - high) / width
        if over <= 0.05:
            return "just_above"
        if over <= 0.25:
            return "above"
        return "far_above"
    if value < low:
        under = (low - value) / width
        if under <= 0.05:
            return "just_below"
        if under <= 0.25:
            return "below"
        return "far_below"
    return "inside"


def _trend(values: list[float]) -> str:
    if len(values) < 3:
        return "flat"
    first, last = values[0], values[-1]
    spread = max(values) - min(values)
    if spread < 1e-6:
        return "flat"
    delta = (last - first) / max(spread, 1e-6)
    if delta > 0.5:
        return "rising"
    if delta < -0.5:
        return "falling"
    return "flat"


def _bucket(value: float | int | None, edges: list[tuple[float, str]]) -> str:
    if value is None:
        return "unknown"
    for limit, name in edges:
        if value <= limit:
            return name
    return edges[-1][1]


DELAY_EDGES = [(60, "under_1h"), (240, "1_4h"), (480, "4_8h"), (float("inf"), "over_8h")]
PROGRESS_EDGES = [(25, "early"), (60, "mid"), (85, "late"), (float("inf"), "nearly_there")]
COUNT_EDGES = [(0, "none"), (2, "few"), (5, "several"), (float("inf"), "many")]


def signature(state: dict[str, Any]) -> dict[str, Any]:
    """
    Reduce a perceived state to a comparable fingerprint.

    Only the parts that plausibly drive a decision are kept; absolute timestamps,
    ids and site names are deliberately excluded so a situation at WH1 can match
    the same situation at WH2.
    """
    events = state.get("unhandled_events") or []
    event_types = sorted({e.get("type") for e in events if e.get("type")})
    max_sev = max(
        (SEVERITY_RANK.get(e.get("severity", "info"), 0) for e in events), default=0
    )

    # Only abnormal readings are part of the fingerprint. Two classes are skipped
    # entirely: readings sitting inside their band, and continuous readings with
    # no band to judge them against (an unjudgeable reading is noise, not a fact
    # -- including them put four meaningless robot-arm entries in every signature
    # and swamped the real ones).
    seen: set[tuple] = set()
    sensor_facts = []
    for site in state.get("sites") or []:
        kind = site.get("kind")
        for reading in site.get("readings") or []:
            metric = reading.get("metric")
            value = float(reading.get("current", 0))
            band = reading.get("normal_band")

            if reading.get("unit") == "bool":
                # Boolean state metrics have no band. Only the off state is a fact
                # worth fingerprinting; a working robot arm is not a situation.
                if value != 0:
                    continue
                fact = {"metric": metric, "excursion": "inactive",
                        "trend": "flat", "site_kind": kind}
            else:
                if not band:
                    continue
                bucket = _excursion_bucket(value, band)
                if bucket == "inside":
                    continue
                fact = {
                    "metric": metric,
                    "excursion": bucket,
                    "trend": _trend(
                        [float(v) for v in reading.get("recent_values_oldest_first", [])]
                    ),
                    "site_kind": kind,
                }

            # Site identity is deliberately dropped so the same condition at WH1
            # and WH2 matches -- which means the same fact can arrive twice.
            key = (fact["metric"], fact["excursion"], fact["trend"], fact["site_kind"])
            if key in seen:
                continue
            seen.add(key)
            sensor_facts.append(fact)

    sensor_facts.sort(key=lambda f: (f["metric"] or "", f["excursion"]))

    shipments = state.get("shipments") or {}
    notable = shipments.get("notable_active") or []
    worst = max(notable, key=lambda s: s.get("delay_minutes") or 0, default=None)
    shipment_facts = None
    if worst and (worst.get("delay_minutes") or 0) > 0:
        shipment_facts = {
            "delay": _bucket(worst.get("delay_minutes"), DELAY_EDGES),
            "progress": _bucket(worst.get("progress_pct"), PROGRESS_EDGES),
            "already_rerouted": bool(worst.get("already_rerouted")),
        }

    inventory = state.get("inventory") or {}

    return {
        "event_types": event_types,
        "max_severity": max_sev,
        "sensors": sensor_facts,
        "shipment": shipment_facts,
        "stock_out_count": _bucket(inventory.get("locations_out_of_stock"), COUNT_EDGES),
    }


# -- similarity ------------------------------------------------------------


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def similarity(a: dict[str, Any], b: dict[str, Any]) -> float:
    """
    Weighted overlap of two fingerprints, 0..1.

    Event types dominate because they are the strongest signal of "same kind of
    situation"; the numeric detail refines within that.
    """
    if not a or not b:
        return 0.0

    types_a = set(a.get("event_types") or [])
    types_b = set(b.get("event_types") or [])

    # Gate: two situations driven by entirely different kinds of event are not
    # comparable, however much their quiet components happen to coincide.
    if types_a and types_b and not (types_a & types_b):
        return 0.0

    scores: list[tuple[float, float]] = []  # (weight, score)

    scores.append((0.40, _jaccard(types_a, types_b)))

    sev_a, sev_b = a.get("max_severity", 0), b.get("max_severity", 0)
    scores.append((0.10, 1.0 - abs(sev_a - sev_b) / 2.0))

    # Components where BOTH sides are empty are dropped rather than scored 1.0.
    # Two situations that each have no abnormal sensors are not thereby similar --
    # absence of evidence is not evidence of similarity, and scoring it as a
    # perfect match let unrelated situations clear the retrieval threshold.
    sa = {(f["metric"], f["excursion"], f["trend"]) for f in (a.get("sensors") or [])}
    sb = {(f["metric"], f["excursion"], f["trend"]) for f in (b.get("sensors") or [])}
    if sa or sb:
        scores.append((0.28, _jaccard(sa, sb)))

    ship_a, ship_b = a.get("shipment"), b.get("shipment")
    if ship_a is not None or ship_b is not None:
        if ship_a is None or ship_b is None:
            ship_score = 0.0
        else:
            ship_score = sum(
                1 for k in ("delay", "progress", "already_rerouted")
                if ship_a.get(k) == ship_b.get(k)
            ) / 3.0
        scores.append((0.14, ship_score))

    stock_a, stock_b = a.get("stock_out_count"), b.get("stock_out_count")
    if stock_a != "none" or stock_b != "none":
        scores.append((0.08, 1.0 if stock_a == stock_b else 0.0))

    total_weight = sum(w for w, _ in scores)
    if total_weight == 0:
        return 0.0
    return round(sum(w * s for w, s in scores) / total_weight, 4)


# -- retrieval -------------------------------------------------------------


def retrieve_similar(
    session: Session,
    current_signature: dict[str, Any],
    limit: int = DEFAULT_LIMIT,
    exclude_decision_id: int | None = None,
    candidate_pool: int = 200,
) -> list[dict[str, Any]]:
    """
    Find past decisions whose situation resembled this one AND whose outcome is
    known. Returns newest-first among the best matches, most similar first.

    Scoring happens in Python over a bounded pool of recent scored decisions. At
    this scale that is far simpler than pushing similarity into SQL, and the pool
    bound keeps it O(candidate_pool) regardless of table size.
    """
    if not current_signature:
        return []

    stmt = (
        select(AgentDecision)
        .options(selectinload(AgentDecision.outcomes))
        .where(
            AgentDecision.outcome_scored_at.is_not(None),
            AgentDecision.error.is_(None),
            AgentDecision.signature.is_not(None),
        )
        .order_by(AgentDecision.id.desc())
        .limit(candidate_pool)
    )
    if exclude_decision_id is not None:
        stmt = stmt.where(AgentDecision.id != exclude_decision_id)

    scored: list[tuple[float, AgentDecision]] = []
    for row in session.scalars(stmt).all():
        if not row.outcomes:
            continue  # scored but had no followable condition -- no lesson in it
        score = similarity(current_signature, row.signature or {})
        if score >= MIN_SIMILARITY:
            scored.append((score, row))

    scored.sort(key=lambda pair: (-pair[0], -pair[1].id))
    return [_as_case(score, row) for score, row in scored[:limit]]


def _as_case(score: float, row: AgentDecision) -> dict[str, Any]:
    """Render a past decision as a compact case for the prompt."""
    actions = [
        {"tool": c.get("tool"), "input": c.get("input")}
        for c in (row.tool_calls or [])
        if c.get("ok")
    ]

    outcomes = [
        {
            "condition": o.subject_type,
            "what_happened": o.condition_status,
            "verdict": o.label,
            "agent_could_influence": o.attribution == "agent_influenceable",
        }
        for o in row.outcomes
    ]

    age_hours = None
    if row.created_at:
        created = row.created_at
        if created.tzinfo is None:
            created = created.replace(tzinfo=dt.timezone.utc)
        age_hours = round(
            (dt.datetime.now(dt.timezone.utc) - created).total_seconds() / 3600, 2
        )

    return {
        "case_id": row.id,
        "similarity": score,
        "age_hours": age_hours,
        "situation": row.signature,
        "action_taken": actions or None,
        "took_no_action": row.action_count == 0,
        "outcomes": outcomes,
    }


def summarize_for_prompt(cases: list[dict[str, Any]]) -> str:
    """
    Render retrieved cases as the text block the model sees.

    Deliberately plain: each case states what the situation was, what was done,
    and what followed. The `agent_could_influence: false` flag is surfaced because
    without it the model would read "resolved after action" as evidence the action
    worked, when for sensor spikes it is evidence of nothing.
    """
    if not cases:
        return (
            "No comparable past cases have been recorded yet. Decide on the "
            "current evidence alone."
        )

    lines = [
        f"{len(cases)} past decision(s) faced a similar situation. What happened:",
        "",
    ]
    for case in cases:
        lines.append(f"- Case #{case['case_id']} (similarity {case['similarity']:.2f}, "
                     f"{case['age_hours']}h ago)")
        sig = case["situation"] or {}
        lines.append(f"    situation: events={sig.get('event_types')} "
                     f"sensors={sig.get('sensors')} shipment={sig.get('shipment')}")
        if case["took_no_action"]:
            lines.append("    action: took no action")
        else:
            tools = ", ".join(a["tool"] for a in (case["action_taken"] or []))
            lines.append(f"    action: {tools}")
        # A busy cycle can produce a dozen outcome rows; rendering them all would
        # crowd out the current state. Show a few and count the rest.
        shown = case["outcomes"][:MAX_OUTCOMES_PER_CASE]
        for out in shown:
            note = "" if out["agent_could_influence"] else \
                "  [note: no tool can influence this condition -- it resolves on its own]"
            lines.append(
                f"    outcome: {out['condition']} -> {out['what_happened']} "
                f"({out['verdict']}){note}"
            )
        remaining = len(case["outcomes"]) - len(shown)
        if remaining > 0:
            tally: dict[str, int] = {}
            for out in case["outcomes"][MAX_OUTCOMES_PER_CASE:]:
                tally[out["verdict"]] = tally.get(out["verdict"], 0) + 1
            summary = ", ".join(f"{v}x {k}" for k, v in sorted(tally.items()))
            lines.append(f"    ... and {remaining} more: {summary}")
        lines.append("")

    lines.append(
        "Weigh this history as evidence, not instruction. A pattern that proved "
        "harmless before can be real this time, and precedent where the agent "
        "could not influence the outcome says nothing about whether acting helps."
    )
    return "\n".join(lines)
