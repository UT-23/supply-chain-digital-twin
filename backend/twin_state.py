"""
Builds the live twin snapshot the 3D frontend renders.

Two kinds of thing are in here, deliberately kept apart:

  * MEASUREMENTS -- the latest reading per metric per site, shipment counts,
    inventory totals. Facts, no judgement.
  * THE AGENT'S VERDICT -- `agent_status` per site, derived from what the agent
    actually did (open alerts it raised, actions it took). This exists so the
    3D scene can colour a site by the agent's conclusion instead of running a
    second, parallel threshold check of its own.

`agent_status` is deliberately "unassessed" when the agent has not looked at a
site recently. That is honest: the agent reasons on a cadence (every
AGENT_MIN_INTERVAL_SECONDS at most, event-triggered), so for much of the time
there simply is no current verdict, and inventing one from thresholds would put
the scene back to judging for itself.
"""
from __future__ import annotations

import datetime as dt
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from db.models import Alert, Event, Inventory, Shipment, Warehouse, WarehouseSensor


def _latest_readings(session: Session, warehouse_id: int) -> dict[str, float]:
    """Latest value per metric for one site."""
    newest = (
        select(
            WarehouseSensor.reading_type,
            func.max(WarehouseSensor.id).label("max_id"),
        )
        .where(WarehouseSensor.warehouse_id == warehouse_id)
        .group_by(WarehouseSensor.reading_type)
        .subquery()
    )
    rows = session.execute(
        select(WarehouseSensor.reading_type, WarehouseSensor.value)
        .join(newest, WarehouseSensor.id == newest.c.max_id)
    ).all()
    return {rt: float(v) for rt, v in rows}


def build_snapshot(session: Session) -> dict[str, Any]:
    sites: dict[str, Any] = {}
    for wh in session.scalars(select(Warehouse).order_by(Warehouse.code)).all():
        readings = _latest_readings(session, wh.id)
        stock = session.scalar(
            select(func.coalesce(func.sum(Inventory.quantity_on_hand), 0))
            .where(Inventory.warehouse_id == wh.id)
        ) or 0
        sites[wh.code] = {
            "warehouse_id": wh.id,
            "name": wh.name,
            "kind": wh.kind,
            "readings": readings,
            "inventory": int(stock),
        }

    counts = dict(
        session.execute(
            select(Shipment.live_status, func.count())
            .where(Shipment.live_status.is_not(None))
            .group_by(Shipment.live_status)
        ).all()
    )

    open_alerts = session.scalar(
        select(func.count()).select_from(Alert).where(Alert.status != "resolved")
    ) or 0

    _apply_agent_status(session, sites)

    return {
        "observed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "sites": sites,
        "shipments": {
            "in_transit": counts.get("IN_TRANSIT", 0),
            "delayed": counts.get("DELAYED", 0),
            "delivered": counts.get("DELIVERED", 0),
        },
        "open_alerts": open_alerts,
        "locations_out_of_stock": session.scalar(
            select(func.count()).select_from(Inventory).where(Inventory.quantity_on_hand == 0)
        ) or 0,
    }


# How recently the agent must have acted on a site for its verdict to still count
# as current. Beyond this the site reverts to "unassessed" rather than showing a
# stale conclusion as if it were live.
AGENT_VERDICT_TTL_SECONDS = 600


def _apply_agent_status(session: Session, sites: dict[str, Any]) -> None:
    """
    Attach the agent's own verdict to each site.

      escalated   -- the agent raised an alert about this site and it is unresolved
      actioned    -- the agent took some other action on it recently
      assessed_ok -- the agent looked recently and did not act on it
      unassessed  -- no recent agent decision covers this site

    Derived from what the agent DID (alerts, agent-authored events), never from
    the sensor values themselves.
    """
    by_id = {s["warehouse_id"]: s for s in sites.values()}
    for site in sites.values():
        site["agent_status"] = "unassessed"
        site["agent_note"] = None

    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(
        seconds=AGENT_VERDICT_TTL_SECONDS
    )

    # Any recent agent-authored event marks the site as looked-at.
    recent = session.scalars(
        select(Event)
        .where(Event.actor == "agent", Event.occurred_at >= cutoff)
        .order_by(Event.occurred_at.desc())
        .limit(50)
    ).all()
    for ev in recent:
        if ev.entity_type == "warehouse" and ev.entity_id in by_id:
            site = by_id[ev.entity_id]
            if site["agent_status"] == "unassessed":
                site["agent_status"] = "actioned"
                site["agent_note"] = ev.message

    # An unresolved alert outranks everything else.
    open_alerts = session.scalars(
        select(Alert)
        .where(Alert.status != "resolved", Alert.entity_type == "warehouse")
        .order_by(Alert.created_at.desc())
        .limit(50)
    ).all()
    for a in open_alerts:
        if a.entity_id in by_id:
            site = by_id[a.entity_id]
            site["agent_status"] = "escalated"
            site["agent_note"] = a.reason
            site["agent_severity"] = a.severity
