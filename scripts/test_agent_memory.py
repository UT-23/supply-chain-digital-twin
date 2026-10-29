"""
Verify the memory machinery without involving an LLM.

Outcome scoring and retrieval are pure database logic, so they can be tested
deterministically: build decisions with known input states, put the database into
a known after-state, score them, and check the labels and the retrieval ranking.

This tests the plumbing -- that outcomes are computed correctly and that similar
situations are retrieved. It does NOT test whether showing those cases to a model
changes what the model does. That needs real API calls; see
scripts/memory_ab_probe.py.

    python scripts/test_agent_memory.py
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

from sqlalchemy import func, select

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from agent import memory, outcomes  # noqa: E402
from db.models import AgentDecision, DecisionOutcome, Inventory, Shipment, WarehouseSensor  # noqa: E402
from db.session import SessionLocal, engine  # noqa: E402

results: list[tuple[bool, str, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((ok, name, detail))
    print(f"[{'  ok  ' if ok else ' FAIL '}] {name}" + (f"  -- {detail}" if detail else ""))


def now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def sensor_state(metric: str, value: float, band: tuple[float, float], trend=None, kind="warehouse"):
    return {
        "observed_at": now().isoformat(),
        "sites": [{
            "warehouse_id": 1, "code": "WH1", "name": "WH1", "kind": kind,
            "readings": [{
                "metric": metric, "current": value, "unit": "%",
                "recent_values_oldest_first": trend or [value] * 6,
                "normal_band": {"low": band[0], "high": band[1]},
                "last_recorded_at": now().isoformat(),
            }],
        }],
        "shipments": {"counts_by_status": {}, "notable_active": []},
        "inventory": {"locations_out_of_stock": 0, "lowest_stock": []},
        "unhandled_events": [{
            "event_id": 5001, "type": "SENSOR_SPIKE", "severity": "critical",
            "entity_type": "warehouse", "entity_id": 1,
            "message": f"{metric} anomaly at WH1",
            "details": {"reading_type": metric, "approx_value": value,
                        "normal_range": [band[0], band[1]]},
            "occurred_at": now().isoformat(), "age_seconds": 5.0,
        }],
        "open_alerts": [], "recent_agent_actions": [],
    }


def make_decision(session, state, *, acted: bool, age_seconds: float = 300.0) -> AgentDecision:
    row = AgentDecision(
        created_at=now() - dt.timedelta(seconds=age_seconds),
        trigger="test",
        input_state=state,
        reasoning="test decision",
        tool_calls=[{"tool": "escalate_alert", "input": {}, "ok": True}] if acted else [],
        tool_results=[],
        action_count=1 if acted else 0,
        no_action=not acted,
        model="test",
        signature=memory.signature(state),
    )
    session.add(row)
    session.flush()
    return row


def main() -> int:
    session = SessionLocal()
    try:
        run(session)
    finally:
        session.rollback()  # never persist test rows
        session.close()
        engine.dispose()

    print("\n" + "=" * 64)
    failed = [r for r in results if not r[0]]
    print(f"{len(results) - len(failed)}/{len(results)} checks passed")
    for _, name, detail in failed:
        print(f"  - {name}: {detail}")
    return 1 if failed else 0


def run(session) -> None:
    # ---------------------------------------------------------- fingerprints
    print("-- fingerprinting " + "-" * 44)

    band = (40.0, 64.0)
    hot = memory.signature(sensor_state("humidity", 71.2, band))
    hot2 = memory.signature(sensor_state("humidity", 72.8, band))
    cool = memory.signature(sensor_state("humidity", 52.0, band))
    rising = memory.signature(
        sensor_state("humidity", 69.1, band, trend=[61, 63, 65, 66, 68, 69.1]))
    flat = memory.signature(
        sensor_state("humidity", 69.1, band, trend=[68.9, 69, 68.8, 69.2, 69, 69.1]))

    check("nearby values share a fingerprint",
          hot["sensors"] == hot2["sensors"], f"{hot['sensors']} vs {hot2['sensors']}")
    check("in-band readings are excluded from fingerprint",
          cool["sensors"] == [], str(cool["sensors"]))
    check("rising and flat trends fingerprint differently",
          rising["sensors"] != flat["sensors"],
          f"{rising['sensors'][0]['trend']} vs {flat['sensors'][0]['trend']}")

    s_hot_hot = memory.similarity(hot, hot2)
    s_hot_cool = memory.similarity(hot, cool)
    s_rise_flat = memory.similarity(rising, flat)
    check("identical situations score ~1.0", s_hot_hot > 0.95, f"{s_hot_hot}")
    check("different situations score lower",
          s_hot_cool < s_hot_hot, f"hot~cool={s_hot_cool} < hot~hot={s_hot_hot}")
    check("trend difference lowers similarity",
          s_rise_flat < s_hot_hot, f"{s_rise_flat}")
    check("similarity is symmetric",
          memory.similarity(hot, cool) == memory.similarity(cool, hot))

    # ------------------------------------------------------- outcome scoring
    print("\n-- outcome scoring: sensor spike " + "-" * 30)

    wh = session.scalars(select(WarehouseSensor).limit(1)).first()
    if wh is None:
        check("fixtures available", False, "no sensor rows; run the server first")
        return
    wid = wh.warehouse_id

    # Write a reading INSIDE the band -> the spike should score as cleared.
    session.add(WarehouseSensor(
        warehouse_id=wid, reading_type="probe_metric", value=50.0,
        unit="%", recorded_at=now()))
    session.flush()

    st = sensor_state("probe_metric", 78.0, (40.0, 64.0))
    st["unhandled_events"][0]["entity_id"] = wid
    d_noact = make_decision(session, st, acted=False)
    rows = outcomes.score_decision(session, d_noact, 120)
    session.flush()
    check("one outcome row per followable condition", len(rows) == 1, f"{len(rows)}")
    check("spike back in band scores as cleared",
          rows[0].condition_status == outcomes.CLEARED, rows[0].condition_status)
    check("no action + cleared => self_resolved",
          rows[0].label == outcomes.SELF_RESOLVED, rows[0].label)
    check("sensor spikes marked self_healing (no tool can influence them)",
          rows[0].attribution == "self_healing", rows[0].attribution)
    check("outcome_scored_at stamped", d_noact.outcome_scored_at is not None)

    # Now a reading still OUTSIDE the band -> persisted, and no action => missed.
    session.add(WarehouseSensor(
        warehouse_id=wid, reading_type="probe_metric2", value=79.0,
        unit="%", recorded_at=now()))
    session.flush()
    st2 = sensor_state("probe_metric2", 78.0, (40.0, 64.0))
    st2["unhandled_events"][0]["entity_id"] = wid
    st2["unhandled_events"][0]["details"]["reading_type"] = "probe_metric2"
    d2 = make_decision(session, st2, acted=False)
    rows2 = outcomes.score_decision(session, d2, 120)
    session.flush()
    check("still-out-of-band scores as persisted/worsened",
          rows2[0].condition_status in (outcomes.PERSISTED, outcomes.WORSENED),
          rows2[0].condition_status)
    check("no action + not cleared => missed", rows2[0].label == outcomes.MISSED, rows2[0].label)

    # Same after-state but the agent DID act -> action_ineffective.
    d3 = make_decision(session, st2, acted=True)
    rows3 = outcomes.score_decision(session, d3, 120)
    session.flush()
    check("action + not cleared => action_ineffective",
          rows3[0].label == outcomes.ACTION_INEFFECTIVE, rows3[0].label)

    print("\n-- outcome scoring: stock-out " + "-" * 33)
    inv = session.scalars(select(Inventory).where(Inventory.quantity_on_hand > 0).limit(1)).first()
    if inv is not None:
        st3 = {
            "observed_at": now().isoformat(), "sites": [],
            "shipments": {"counts_by_status": {}, "notable_active": []},
            "inventory": {"locations_out_of_stock": 1, "lowest_stock": []},
            "unhandled_events": [{
                "event_id": 5002, "type": "STOCK_OUT", "severity": "critical",
                "entity_type": "inventory", "entity_id": inv.id,
                "message": "out of stock",
                "details": {"product_id": inv.product_id, "warehouse_id": inv.warehouse_id},
                "occurred_at": now().isoformat(), "age_seconds": 5.0,
            }],
            "open_alerts": [], "recent_agent_actions": [],
        }
        inv.quantity_on_hand = (inv.reorder_point or 0) + 500
        session.flush()
        d4 = make_decision(session, st3, acted=True)
        r4 = outcomes.score_decision(session, d4, 120)
        session.flush()
        check("restocked stock-out scores as cleared",
              r4[0].condition_status == outcomes.CLEARED, r4[0].condition_status)
        check("stock-outs marked agent_influenceable",
              r4[0].attribution == "agent_influenceable", r4[0].attribution)

        inv.quantity_on_hand = 0
        session.flush()
        d5 = make_decision(session, st3, acted=False)
        r5 = outcomes.score_decision(session, d5, 120)
        session.flush()
        check("zero stock scores as persisted",
              r5[0].condition_status == outcomes.PERSISTED, r5[0].condition_status)

    print("\n-- outcome scoring: shipment delay " + "-" * 28)
    ship = session.scalars(
        select(Shipment).where(Shipment.live_status == "DELIVERED").limit(1)).first()
    if ship is not None:
        st4 = {
            "observed_at": now().isoformat(), "sites": [],
            "shipments": {"counts_by_status": {}, "notable_active": []},
            "inventory": {"locations_out_of_stock": 0, "lowest_stock": []},
            "unhandled_events": [{
                "event_id": 5003, "type": "SHIPMENT_DELAY", "severity": "warning",
                "entity_type": "shipment", "entity_id": ship.id,
                "message": "delayed", "details": {"delay_minutes": 200},
                "occurred_at": now().isoformat(), "age_seconds": 5.0,
            }],
            "open_alerts": [], "recent_agent_actions": [],
        }
        d6 = make_decision(session, st4, acted=True)
        r6 = outcomes.score_decision(session, d6, 120)
        session.flush()
        check("delivered shipment scores as cleared",
              r6[0].condition_status == outcomes.CLEARED, r6[0].condition_status)
        check("shipment delays marked agent_influenceable",
              r6[0].attribution == "agent_influenceable", r6[0].attribution)

    # ------------------------------------------------------------ retrieval
    print("\n-- retrieval " + "-" * 50)

    sig_now = memory.signature(sensor_state("probe_metric", 77.0, (40.0, 64.0)))
    cases = memory.retrieve_similar(session, sig_now, limit=5)
    check("retrieval returns scored cases", len(cases) > 0, f"{len(cases)} cases")
    if cases:
        check("cases sorted by similarity descending",
              all(cases[i]["similarity"] >= cases[i + 1]["similarity"]
                  for i in range(len(cases) - 1)),
              str([c["similarity"] for c in cases]))
        check("every retrieved case carries outcomes",
              all(c["outcomes"] for c in cases))
        check("retrieval respects the limit",
              len(memory.retrieve_similar(session, sig_now, limit=2)) <= 2)
        check("self_healing flag surfaced to the prompt",
              any(o["agent_could_influence"] is False
                  for c in cases for o in c["outcomes"]))

    unscored = session.scalar(
        select(func.count()).select_from(AgentDecision)
        .where(AgentDecision.outcome_scored_at.is_(None)))
    retrieved_ids = {c["case_id"] for c in cases}
    unscored_ids = {r for (r,) in session.execute(
        select(AgentDecision.id).where(AgentDecision.outcome_scored_at.is_(None)))}
    check("unscored decisions are never retrieved",
          not (retrieved_ids & unscored_ids), f"{unscored} unscored in table")

    # Irrelevant situation should retrieve nothing above threshold.
    far = memory.signature({
        "observed_at": now().isoformat(), "sites": [],
        "shipments": {"counts_by_status": {}, "notable_active": []},
        "inventory": {"locations_out_of_stock": 0, "lowest_stock": []},
        "unhandled_events": [], "open_alerts": [], "recent_agent_actions": [],
    })
    far_cases = memory.retrieve_similar(session, far, limit=5)
    check("a quiet situation retrieves NO disruption cases",
          len(far_cases) == 0, f"retrieved {len(far_cases)}")

    # Absence of evidence must not read as similarity, and unrelated event types
    # must not match at all.
    empty_sig = {"event_types": [], "max_severity": 0, "sensors": [],
                 "shipment": None, "stock_out_count": "none"}
    stock_sig = {"event_types": ["STOCK_OUT"], "max_severity": 2, "sensors": [],
                 "shipment": None, "stock_out_count": "few"}
    spike_sig = {"event_types": ["SENSOR_SPIKE"], "max_severity": 2,
                 "sensors": [{"metric": "humidity", "excursion": "far_above",
                              "trend": "flat"}],
                 "shipment": None, "stock_out_count": "none"}
    check("quiet vs stock-out scores below threshold",
          memory.similarity(empty_sig, stock_sig) < memory.MIN_SIMILARITY,
          f"{memory.similarity(empty_sig, stock_sig)}")
    check("disjoint event types score 0",
          memory.similarity(spike_sig, stock_sig) == 0.0,
          f"{memory.similarity(spike_sig, stock_sig)}")

    # ------------------------------------------------------- prompt rendering
    print("\n-- prompt block " + "-" * 47)
    text = memory.summarize_for_prompt(cases)
    check("prompt block is non-empty", bool(text.strip()))
    check("prompt block names case ids", "Case #" in text)
    check("prompt block states outcomes", "outcome:" in text)
    check("prompt block carries the influence caveat",
          "no tool can influence" in text or "evidence, not instruction" in text)
    check("empty retrieval yields an explicit 'no cases' block",
          "No comparable past cases" in memory.summarize_for_prompt([]))
    print("\n--- sample prompt block ---")
    print(text[:900])

    # ------------------------------------------------------------- scheduler
    print("\n-- due-scoring scheduler " + "-" * 38)
    fresh = make_decision(session, sensor_state("humidity", 70.0, (40.0, 64.0)),
                          acted=False, age_seconds=1.0)
    session.flush()
    before = fresh.outcome_scored_at
    outcomes.score_due_decisions(session, window_seconds=600, limit=50)
    check("decisions inside their window are not scored yet",
          before is None and fresh.outcome_scored_at is None)


if __name__ == "__main__":
    raise SystemExit(main())
