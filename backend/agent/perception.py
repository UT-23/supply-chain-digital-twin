"""
Builds the state snapshot the agent reasons over.

This module decides what the model gets to see, which bounds what it can possibly
reason about. Two deliberate choices:

  * Sensor readings come with their normal operating band and a short trend, not
    a pre-computed "is_breach" flag. Handing the model a boolean would mean the
    threshold decision had already been made in Python and the model was only
    writing prose about it.
  * Numbers are given raw. No bucketing into "high"/"critical", for the same reason.
"""
from __future__ import annotations

import datetime as dt
import json
import os
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from db.models import (
    Alert,
    Event,
    Inventory,
    Product,
    Shipment,
    Warehouse,
    WarehouseSensor,
)
from simulation import sensors

# How many recent samples of each metric to include, newest last.
TREND_SAMPLES = 6
MAX_SHIPMENTS = 12
MAX_INVENTORY_ROWS = 12
MAX_EVENTS = 15

# Hard ceiling on the serialized snapshot, in approximate tokens (chars/4).
#
# This exists because an unhandled-event backlog is unbounded: if the agent is
# down or failing, events pile up, and a snapshot carrying 15 of them with full
# `details` JSON reached ~6,500 tokens -- which, on top of the system prompt and
# tool schemas, produced a 10,808-token request against Groq's 8,000 TPM ceiling
# and HTTP 413 on every cycle. Trimming here is far better than discovering the
# limit at the provider.
MAX_STATE_TOKENS = int(os.getenv("AGENT_MAX_STATE_TOKENS", "1800"))

# Keys worth keeping on an event when the snapshot has to be trimmed. The rest
# of `details` is diagnostic detail the model does not need to decide.
ESSENTIAL_DETAIL_KEYS = (
    "reading_type", "approx_value", "normal_range", "warehouse_code",
    "product_id", "warehouse_id", "delay_minutes", "cumulative_delay_minutes",
    "order_id", "quantity_before", "reorder_point", "duration_cycles",
)


def _approx_tokens(payload: Any) -> int:
    """
    Approximate tokens, measured the SAME way the prompt serializes the state.

    This previously measured compact JSON while the prompt sent indent=2, so the
    budget under-counted by ~35% and let oversized snapshots through.
    """
    return len(json.dumps(payload, default=str)) // 4


def _shrink_to_budget(state: dict[str, Any]) -> dict[str, Any]:
    """
    Trim the snapshot until it fits MAX_STATE_TOKENS, most-expendable first.

    Order matters: detail noise goes before whole events, and events go before
    anything that changes what the agent can reason about structurally. The
    newest events are kept -- they are the ones the agent was woken for.
    """
    if _approx_tokens(state) <= MAX_STATE_TOKENS:
        return state

    # 1. strip non-essential keys from event details
    for ev in state.get("unhandled_events") or []:
        details = ev.get("details") or {}
        ev["details"] = {k: v for k, v in details.items() if k in ESSENTIAL_DETAIL_KEYS}
    if _approx_tokens(state) <= MAX_STATE_TOKENS:
        return state

    # 2. drop the oldest events, keeping at least a couple
    events = state.get("unhandled_events") or []
    while len(events) > 2 and _approx_tokens(state) > MAX_STATE_TOKENS:
        events.pop()            # list is newest-first
    state["truncated"] = {
        "reason": "snapshot exceeded MAX_STATE_TOKENS",
        "events_shown": len(events),
    }
    if _approx_tokens(state) <= MAX_STATE_TOKENS:
        return state

    # 3. last resort: shorten the per-metric trends
    for site in state.get("sites") or []:
        for reading in site.get("readings") or []:
            reading["recent_values_oldest_first"] = reading.get(
                "recent_values_oldest_first", [])[-3:]
    return state


def _aware(value: dt.datetime | None) -> dt.datetime | None:
    if value is None:
        return None
    return value.replace(tzinfo=dt.timezone.utc) if value.tzinfo is None else value


def _iso(value: dt.datetime | None) -> str | None:
    v = _aware(value)
    return v.isoformat() if v else None


def build_state(session: Session) -> dict[str, Any]:
    """Assemble the full perceived state. Safe to call on an empty database."""
    now = dt.datetime.now(dt.timezone.utc)

    state = {
        "observed_at": now.isoformat(),
        "sites": _sites(session),
        "shipments": _shipments(session, now),
        "inventory": _inventory(session),
        "unhandled_events": _events(session, now),
        "open_alerts": _open_alerts(session),
        "recent_agent_actions": _recent_actions(session, now),
    }
    return _shrink_to_budget(state)


def _sites(session: Session) -> list[dict[str, Any]]:
    """Per-site sensor readings, each with its normal band and recent trend."""
    out = []
    for wh in session.scalars(select(Warehouse).order_by(Warehouse.code)).all():
        types = session.scalars(
            select(WarehouseSensor.reading_type)
            .where(WarehouseSensor.warehouse_id == wh.id)
            .distinct()
        ).all()

        readings = []
        for reading_type in sorted(types):
            rows = session.execute(
                select(WarehouseSensor.value, WarehouseSensor.unit, WarehouseSensor.recorded_at)
                .where(
                    WarehouseSensor.warehouse_id == wh.id,
                    WarehouseSensor.reading_type == reading_type,
                )
                .order_by(WarehouseSensor.id.desc())
                .limit(TREND_SAMPLES)
            ).all()
            if not rows:
                continue

            values = [float(r[0]) for r in rows][::-1]  # oldest -> newest
            band = sensors.normal_range(reading_type, wh.kind or "warehouse")

            readings.append({
                "metric": reading_type,
                "current": values[-1],
                "unit": rows[0][1],
                "recent_values_oldest_first": values,
                "normal_band": (
                    None if band is None else {"low": band[0], "high": band[1]}
                ),
                "last_recorded_at": _iso(rows[0][2]),
            })

        out.append({
            "warehouse_id": wh.id,
            "code": wh.code,
            "name": wh.name,
            "kind": wh.kind,
            "readings": readings,
        })
    return out


def _shipments(session: Session, now: dt.datetime) -> dict[str, Any]:
    active_states = ("PENDING", "IN_TRANSIT", "DELAYED")

    counts = dict(
        session.execute(
            select(Shipment.live_status, func.count())
            .where(Shipment.live_status.is_not(None))
            .group_by(Shipment.live_status)
        ).all()
    )

    # Show the ones most likely to need a decision: delayed first, then least progressed.
    rows = session.scalars(
        select(Shipment)
        .where(Shipment.live_status.in_(active_states))
        .order_by(
            Shipment.delay_minutes.desc().nullslast(),
            Shipment.progress_pct.asc().nullsfirst(),
        )
        .limit(MAX_SHIPMENTS)
    ).all()

    return {
        "counts_by_status": counts,
        "notable_active": [
            {
                "shipment_id": s.id,
                "order_id": s.order_id,
                "status": s.live_status,
                "progress_pct": s.progress_pct,
                "delay_minutes": s.delay_minutes or 0,
                "eta": _iso(s.eta),
                "hours_until_eta": (
                    None if s.eta is None
                    else round((_aware(s.eta) - now).total_seconds() / 3600, 1)
                ),
                "shipping_mode": s.shipping_mode,
                "current_route": s.current_route,
                "already_rerouted": s.rerouted_at is not None,
            }
            for s in rows
        ],
    }


def _inventory(session: Session) -> dict[str, Any]:
    """The stock positions worth looking at: lowest relative to reorder point."""
    rows = session.execute(
        select(Inventory, Product, Warehouse)
        .join(Product, Product.id == Inventory.product_id)
        .join(Warehouse, Warehouse.id == Inventory.warehouse_id)
        .order_by(Inventory.quantity_on_hand.asc())
        .limit(MAX_INVENTORY_ROWS)
    ).all()

    out_of_stock = session.scalar(
        select(func.count()).select_from(Inventory).where(Inventory.quantity_on_hand == 0)
    ) or 0

    return {
        "locations_out_of_stock": out_of_stock,
        "lowest_stock": [
            {
                "product_id": inv.product_id,
                "product_name": prod.name,
                "warehouse_id": inv.warehouse_id,
                "warehouse_code": wh.code,
                "quantity_on_hand": inv.quantity_on_hand,
                "reorder_point": inv.reorder_point,
                "already_flagged": prod.low_stock_flagged,
            }
            for inv, prod, wh in rows
        ],
    }


def _events(session: Session, now: dt.datetime) -> list[dict[str, Any]]:
    rows = session.scalars(
        select(Event)
        .where(Event.handled_at.is_(None), Event.actor != "agent")
        .order_by(Event.occurred_at.desc(), Event.id.desc())
        .limit(MAX_EVENTS)
    ).all()

    return [
        {
            "event_id": e.id,
            "occurred_at": _iso(e.occurred_at),
            "age_seconds": round((now - (_aware(e.occurred_at) or now)).total_seconds(), 1),
            "type": e.event_type,
            "severity": e.severity,
            "entity_type": e.entity_type,
            "entity_id": e.entity_id,
            "message": e.message,
            "details": e.details,
        }
        for e in rows
    ]


def _open_alerts(session: Session) -> list[dict[str, Any]]:
    rows = session.scalars(
        select(Alert)
        .where(Alert.status != "resolved")
        .order_by(Alert.created_at.desc())
        .limit(10)
    ).all()
    return [
        {
            "alert_id": a.id,
            "severity": a.severity,
            "reason": a.reason,
            "status": a.status,
            "created_at": _iso(a.created_at),
            "entity_type": a.entity_type,
            "entity_id": a.entity_id,
        }
        for a in rows
    ]


def _recent_actions(session: Session, now: dt.datetime) -> list[dict[str, Any]]:
    """
    What the agent itself did recently.

    Included so it does not repeat an action it already took -- without this the
    agent re-reroutes the same shipment every cycle.
    """
    rows = session.scalars(
        select(Event)
        .where(Event.actor == "agent")
        .order_by(Event.occurred_at.desc())
        .limit(10)
    ).all()
    return [
        {
            "type": e.event_type,
            "entity_type": e.entity_type,
            "entity_id": e.entity_id,
            "message": e.message,
            "age_seconds": round((now - (_aware(e.occurred_at) or now)).total_seconds(), 1),
        }
        for e in rows
    ]


def has_new_work(session: Session) -> int:
    """Count of unhandled non-agent events. Drives the event-triggered cycle."""
    return session.scalar(
        select(func.count())
        .select_from(Event)
        .where(Event.handled_at.is_(None), Event.actor != "agent")
    ) or 0
