"""
Scores what actually happened after each agent decision, from database state.

A decision is scored once its observation window has elapsed. For every condition
the decision was looking at, the scorer re-checks the database and records whether
that condition cleared, persisted, or got worse -- then combines that with whether
the agent acted, to produce a label the memory layer can learn from.

## The causality caveat -- read this before trusting any label

These labels are CORRELATIONAL. The database can tell you a condition cleared. It
cannot tell you the condition cleared *because* the agent acted. Two things make
that gap wide in the current simulation:

  * SENSOR_SPIKE conditions expire on their own after SIM_SPIKE_CYCLES_MIN..MAX
    cycles, and no tool in the action layer can affect a sensor reading at all.
    Every sensor spike therefore clears regardless of what the agent does. A
    "resolved_after_action" label on a sensor spike means nothing causally.
  * STOCK_OUT conditions partially self-heal: the simulation restocks a product
    once it is drawn again below its reorder point.

`ATTRIBUTION` below records, per condition type, whether the agent had any tool
capable of influencing it. Read every label through that field.
"""
from __future__ import annotations

import datetime as dt
import logging
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from db.models import AgentDecision, DecisionOutcome, Event, Inventory, Shipment, WarehouseSensor
from simulation import sensors

log = logging.getLogger("agent.outcomes")

# Whether the action layer has any tool that can influence this condition type.
# This describes the CURRENT system, not the world: if a dehumidifier-control tool
# were added, SENSOR_SPIKE would become agent_influenceable.
ATTRIBUTION: dict[str, str] = {
    # No tool changes a sensor reading, and simulated spikes expire on a timer.
    "SENSOR_SPIKE": "self_healing",
    # restock_inventory genuinely raises quantity_on_hand.
    "STOCK_OUT": "agent_influenceable",
    # reroute_shipment genuinely reduces delay_minutes and moves the ETA.
    "SHIPMENT_DELAY": "agent_influenceable",
    "SHIPMENT_DELIVERED": "self_healing",
}

CLEARED, PERSISTED, WORSENED, UNKNOWN = "cleared", "persisted", "worsened", "unknown"

SELF_RESOLVED = "self_resolved"
MISSED = "missed"
RESOLVED_AFTER_ACTION = "resolved_after_action"
ACTION_INEFFECTIVE = "action_ineffective"


def _aware(value: dt.datetime | None) -> dt.datetime | None:
    if value is None:
        return None
    return value.replace(tzinfo=dt.timezone.utc) if value.tzinfo is None else value


def _label(status: str, acted: bool) -> str:
    if status == UNKNOWN:
        return "unknown"
    if acted:
        return RESOLVED_AFTER_ACTION if status == CLEARED else ACTION_INEFFECTIVE
    return SELF_RESOLVED if status == CLEARED else MISSED


# -- per-condition re-checks -----------------------------------------------


def _check_sensor_spike(session: Session, event: dict[str, Any]) -> dict[str, Any]:
    """Is the metric that spiked back inside its normal band?"""
    details = event.get("details") or {}
    warehouse_id = event.get("entity_id")
    metric = details.get("reading_type")
    if warehouse_id is None or not metric:
        return {"status": UNKNOWN, "detail": {"why": "event lacks warehouse_id/reading_type"}}

    row = session.execute(
        select(WarehouseSensor.value, WarehouseSensor.recorded_at)
        .where(
            WarehouseSensor.warehouse_id == warehouse_id,
            WarehouseSensor.reading_type == metric,
        )
        .order_by(WarehouseSensor.id.desc())
        .limit(1)
    ).first()
    if row is None:
        return {"status": UNKNOWN, "detail": {"why": "no readings for metric"}}

    current = float(row[0])
    band = details.get("normal_range")
    if not band:
        return {"status": UNKNOWN, "detail": {"why": "no normal_range on event"}}

    low, high = float(band[0]), float(band[1])
    peak = details.get("approx_value")
    inside = low <= current <= high

    status = CLEARED if inside else PERSISTED
    if not inside and peak is not None and abs(current - (low + high) / 2) > abs(
        float(peak) - (low + high) / 2
    ):
        status = WORSENED

    return {
        "status": status,
        "detail": {
            "metric": metric,
            "value_at_event": peak,
            "value_now": current,
            "normal_band": [low, high],
        },
    }


def _check_stock_out(session: Session, event: dict[str, Any]) -> dict[str, Any]:
    """Has stock been replenished at the affected location?"""
    details = event.get("details") or {}
    product_id = details.get("product_id")
    warehouse_id = details.get("warehouse_id")

    row = None
    if product_id is not None and warehouse_id is not None:
        row = session.scalars(
            select(Inventory).where(
                Inventory.product_id == product_id,
                Inventory.warehouse_id == warehouse_id,
            )
        ).first()
    elif event.get("entity_id") is not None:
        row = session.get(Inventory, event["entity_id"])

    if row is None:
        return {"status": UNKNOWN, "detail": {"why": "inventory row not found"}}

    qty = row.quantity_on_hand
    reorder = row.reorder_point or 0
    if qty > reorder and qty > 0:
        status = CLEARED
    elif qty == 0:
        status = PERSISTED
    else:
        status = PERSISTED  # above zero but still at/below reorder point

    return {
        "status": status,
        "detail": {
            "product_id": row.product_id,
            "warehouse_id": row.warehouse_id,
            "quantity_now": qty,
            "reorder_point": reorder,
        },
    }


def _check_shipment_delay(session: Session, event: dict[str, Any]) -> dict[str, Any]:
    """Delivered, recovered, still late, or later than before?"""
    shipment_id = event.get("entity_id")
    if shipment_id is None:
        return {"status": UNKNOWN, "detail": {"why": "event lacks shipment id"}}

    ship = session.get(Shipment, shipment_id)
    if ship is None:
        return {"status": UNKNOWN, "detail": {"why": "shipment not found"}}

    details = event.get("details") or {}
    delay_at_event = details.get("cumulative_delay_minutes")
    if delay_at_event is None:
        delay_at_event = details.get("delay_minutes")
    delay_now = ship.delay_minutes or 0

    if ship.live_status == "DELIVERED":
        status = CLEARED
    elif delay_at_event is not None and delay_now > float(delay_at_event):
        status = WORSENED
    elif delay_at_event is not None and delay_now < float(delay_at_event):
        # Recovered time but not yet arrived.
        status = CLEARED if delay_now == 0 else PERSISTED
    else:
        status = PERSISTED

    return {
        "status": status,
        "detail": {
            "shipment_id": ship.id,
            "live_status": ship.live_status,
            "progress_pct": ship.progress_pct,
            "delay_at_event": delay_at_event,
            "delay_now": delay_now,
            "was_rerouted": ship.rerouted_at is not None,
        },
    }


CHECKS = {
    "SENSOR_SPIKE": _check_sensor_spike,
    "STOCK_OUT": _check_stock_out,
    "SHIPMENT_DELAY": _check_shipment_delay,
}


# -- scoring ---------------------------------------------------------------


def score_decision(
    session: Session, decision: AgentDecision, window_seconds: int
) -> list[DecisionOutcome]:
    """
    Evaluate one decision. Writes DecisionOutcome rows and stamps
    outcome_scored_at. Returns the rows created (possibly empty).
    """
    now = dt.datetime.now(dt.timezone.utc)
    state = decision.input_state or {}
    subjects = [e for e in (state.get("unhandled_events") or []) if e.get("type") in CHECKS]

    acted = (decision.action_count or 0) > 0
    created = _aware(decision.created_at) or now
    rows: list[DecisionOutcome] = []

    for event in subjects:
        etype = event["type"]
        try:
            result = CHECKS[etype](session, event)
        except Exception as exc:  # noqa: BLE001 - a bad event must not stall scoring
            log.exception("outcome check failed for %s", etype)
            result = {"status": UNKNOWN, "detail": {"error": str(exc)}}

        status = result["status"]

        # Time to clear is only meaningful when the condition did clear, and is an
        # upper bound: we observe at the window edge, not the moment it cleared.
        time_to_clear = None
        if status == CLEARED:
            time_to_clear = int((now - created).total_seconds())

        row = DecisionOutcome(
            decision_id=decision.id,
            scored_at=now,
            window_seconds=window_seconds,
            subject_type=etype,
            subject_event_id=event.get("event_id"),
            subject_entity_type=event.get("entity_type"),
            subject_entity_id=event.get("entity_id"),
            condition_status=status,
            agent_acted=acted,
            label=_label(status, acted),
            attribution=ATTRIBUTION.get(etype, "unknown"),
            time_to_clear_seconds=time_to_clear,
            detail=result.get("detail"),
        )
        session.add(row)
        rows.append(row)

    decision.outcome_scored_at = now
    return rows


def score_due_decisions(session: Session, window_seconds: int, limit: int = 20) -> int:
    """
    Score every decision whose observation window has closed.

    Returns the number of decisions scored (not outcome rows written).
    """
    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=window_seconds)

    due = session.scalars(
        select(AgentDecision)
        .where(
            AgentDecision.outcome_scored_at.is_(None),
            AgentDecision.created_at <= cutoff,
            AgentDecision.error.is_(None),  # failed cycles have nothing to score
        )
        .order_by(AgentDecision.id)
        .limit(limit)
    ).all()

    for decision in due:
        score_decision(session, decision, window_seconds)

    if due:
        session.commit()
        log.info("scored %d decision(s)", len(due))
    return len(due)
