"""
The action layer: everything the agent is allowed to do to the supply chain.

Each public function here is a self-contained tool. In Phase 5 these get exposed
to an LLM, and the docstrings below become the tool descriptions the model reads,
so they are written for a caller with no knowledge of this codebase.

Conventions every tool follows:

  * Plain typed function. No FastAPI, no HTTP -- called in-process by the agent.
  * Returns a JSON-serializable dict describing what changed. Never returns an ORM
    object, so results can be handed straight back to a model.
  * Raises ToolError (with a message meant to be read by the caller) when the
    request cannot be satisfied: unknown id, bad argument, illegal transition.
    The Phase 5 adapter should catch ToolError and return its text as the tool
    result rather than crashing the agent loop.
  * Takes an optional `session`. Pass one to batch several tools into a single
    transaction; omit it and the tool opens, commits and closes its own.
  * Every mutating tool writes an `events` row with actor="agent", so the agent's
    own actions are auditable and distinguishable from simulated conditions.
"""
from __future__ import annotations

import datetime as dt
import logging
from contextlib import contextmanager
from typing import Any, Iterator, Literal

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from db.models import Alert, Event, Inventory, Order, Product, Shipment, Warehouse
from db.session import SessionLocal

log = logging.getLogger("agent.tools")

Severity = Literal["info", "warning", "critical"]
VALID_SEVERITIES: tuple[str, ...] = ("info", "warning", "critical")

ACTOR = "agent"

# Live shipment states the simulation maintains. A shipment that has already been
# delivered is terminal and cannot be rerouted.
ACTIVE_SHIPMENT_STATES = ("PENDING", "IN_TRANSIT", "DELAYED")
DELIVERED = "DELIVERED"

MAX_RESTOCK_UNITS = 100_000


class ToolError(RuntimeError):
    """
    Raised when a tool cannot do what was asked.

    The message is intended to be shown to whoever called the tool (including an
    LLM), so it should say what went wrong and, where possible, what to do about it.
    """


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _aware(value: dt.datetime | None) -> dt.datetime | None:
    """Normalize a DB datetime to UTC-aware; SQLite hands back naive values."""
    if value is None:
        return None
    return value.replace(tzinfo=dt.timezone.utc) if value.tzinfo is None else value


def _iso(value: dt.datetime | None) -> str | None:
    aware = _aware(value)
    return aware.isoformat() if aware else None


@contextmanager
def _use_session(session: Session | None) -> Iterator[Session]:
    """
    Yield a session to work in.

    If the caller supplied one, use it and leave committing to them -- that is how
    several tools get batched into one transaction. Otherwise open a fresh session,
    commit on success, roll back on failure, and always close it.
    """
    if session is not None:
        yield session
        return

    own = SessionLocal()
    try:
        yield own
        own.commit()
    except Exception:
        own.rollback()
        raise
    finally:
        own.close()


def _record_event(
    session: Session,
    *,
    event_type: str,
    severity: str,
    entity_type: str,
    entity_id: int,
    message: str,
    details: dict[str, Any] | None = None,
) -> Event:
    """Write the audit trail row for an agent action."""
    event = Event(
        occurred_at=_now(),
        event_type=event_type,
        severity=severity,
        entity_type=entity_type,
        entity_id=entity_id,
        message=message,
        details=details,
        injected=False,
        actor=ACTOR,
    )
    session.add(event)
    session.flush()  # assign event.id so it can be referenced in the return value
    return event


# ===========================================================================
# Read tools -- the agent should call these before it changes anything.
# ===========================================================================


def get_open_disruptions(
    limit: int = 20,
    min_severity: Severity = "warning",
    session: Session | None = None,
) -> list[dict[str, Any]]:
    """
    List recent problems in the supply chain that nobody has dealt with yet.

    This is the agent's inbox and is normally the first tool to call: it answers
    "what is wrong right now?". It returns events that have not been marked handled,
    newest first. Typical event types are SHIPMENT_DELAY (a shipment will arrive
    later than promised), SENSOR_SPIKE (a warehouse reading such as humidity or
    temperature has gone outside its safe band), and STOCK_OUT (a product has hit
    zero units at a warehouse).

    Use the `entity_type` and `entity_id` on each returned item to look up detail
    with get_shipment_details or get_inventory_status, and pass the event's `id` to
    mark_event_handled once the problem has been dealt with.

    Args:
        limit: Maximum number of disruptions to return, newest first (1-200).
        min_severity: Lowest severity to include. "info" returns everything,
            "warning" skips routine noise, "critical" returns only the most
            serious. Defaults to "warning".
        session: Optional existing database session; leave unset in normal use.

    Returns:
        A list of dicts, each with: id, occurred_at (ISO 8601 UTC), event_type,
        severity, entity_type, entity_id, message, details, age_seconds.
        An empty list means nothing is currently wrong, which is a valid answer.

    Raises:
        ToolError: If min_severity is not one of "info", "warning", "critical",
            or limit is outside 1-200.
    """
    if min_severity not in VALID_SEVERITIES:
        raise ToolError(
            f"min_severity must be one of {VALID_SEVERITIES}, got {min_severity!r}"
        )
    if not 1 <= limit <= 200:
        raise ToolError(f"limit must be between 1 and 200, got {limit}")

    allowed = VALID_SEVERITIES[VALID_SEVERITIES.index(min_severity):]
    now = _now()

    with _use_session(session) as s:
        rows = s.scalars(
            select(Event)
            .where(Event.handled_at.is_(None), Event.severity.in_(allowed))
            .order_by(Event.occurred_at.desc(), Event.id.desc())
            .limit(limit)
        ).all()

        return [
            {
                "id": e.id,
                "occurred_at": _iso(e.occurred_at),
                "event_type": e.event_type,
                "severity": e.severity,
                "entity_type": e.entity_type,
                "entity_id": e.entity_id,
                "message": e.message,
                "details": e.details,
                "age_seconds": round((now - (_aware(e.occurred_at) or now)).total_seconds(), 1),
            }
            for e in rows
        ]


def get_shipment_details(
    shipment_id: int,
    session: Session | None = None,
) -> dict[str, Any]:
    """
    Look up the current state of one shipment.

    Call this before rerouting, to see how late a shipment already is, how far along
    it is, and where it is going. A shipment that is 95% of the way to its
    destination is usually not worth rerouting even if it is delayed.

    Args:
        shipment_id: The shipment's numeric id, as found on a SHIPMENT_DELAY
            disruption's `entity_id`.
        session: Optional existing database session; leave unset in normal use.

    Returns:
        A dict with: shipment_id, order_id, live_status (PENDING, IN_TRANSIT,
        DELAYED or DELIVERED), progress_pct (0-100), eta, original_eta,
        delay_minutes (total accumulated lateness), shipping_mode, current_route,
        destination (city/state/country/market of the order), and
        is_reroutable (False once delivered).

    Raises:
        ToolError: If no shipment with that id exists.
    """
    with _use_session(session) as s:
        ship = s.get(Shipment, shipment_id)
        if ship is None:
            raise ToolError(f"No shipment with id {shipment_id}")

        order = s.get(Order, ship.order_id) if ship.order_id else None

        return {
            "shipment_id": ship.id,
            "order_id": ship.order_id,
            "live_status": ship.live_status,
            "progress_pct": ship.progress_pct,
            "eta": _iso(ship.eta),
            "original_eta": _iso(ship.original_eta),
            "delay_minutes": ship.delay_minutes or 0,
            "shipping_mode": ship.shipping_mode,
            "current_route": ship.current_route,
            "rerouted_at": _iso(ship.rerouted_at),
            "delivered_at": _iso(ship.delivered_at),
            "destination": None if order is None else {
                "city": order.city,
                "state": order.state,
                "country": order.country,
                "market": order.market,
            },
            "is_reroutable": ship.live_status in ACTIVE_SHIPMENT_STATES,
        }


def get_inventory_status(
    product_id: int,
    session: Session | None = None,
) -> dict[str, Any]:
    """
    Show how much of one product is in stock, broken down by warehouse.

    Call this before flagging a product or restocking it, to see whether the
    shortage is everywhere or at a single site. A product that is out at one
    warehouse but well stocked at another usually needs a transfer or a flag rather
    than an emergency restock.

    Args:
        product_id: The product's numeric id, as found on a STOCK_OUT disruption's
            details, or on an order item.
        session: Optional existing database session; leave unset in normal use.

    Returns:
        A dict with: product_id, product_name, low_stock_flagged (whether it is
        already flagged), flagged_at, flag_reason, total_on_hand across all sites,
        total_reserved, is_out_of_stock_everywhere, and `locations` -- one entry per
        warehouse with warehouse_id, warehouse_code, quantity_on_hand,
        quantity_reserved, reorder_point and below_reorder_point.

    Raises:
        ToolError: If no product with that id exists.
    """
    with _use_session(session) as s:
        product = s.get(Product, product_id)
        if product is None:
            raise ToolError(f"No product with id {product_id}")

        rows = s.execute(
            select(Inventory, Warehouse)
            .join(Warehouse, Warehouse.id == Inventory.warehouse_id)
            .where(Inventory.product_id == product_id)
            .order_by(Warehouse.code)
        ).all()

        locations = [
            {
                "warehouse_id": inv.warehouse_id,
                "warehouse_code": wh.code,
                "quantity_on_hand": inv.quantity_on_hand,
                "quantity_reserved": inv.quantity_reserved,
                "reorder_point": inv.reorder_point,
                "below_reorder_point": (
                    inv.reorder_point is not None
                    and inv.quantity_on_hand <= inv.reorder_point
                ),
            }
            for inv, wh in rows
        ]
        total = sum(loc["quantity_on_hand"] for loc in locations)

        return {
            "product_id": product.id,
            "product_name": product.name,
            "low_stock_flagged": product.low_stock_flagged,
            "flagged_at": _iso(product.flagged_at),
            "flag_reason": product.flag_reason,
            "total_on_hand": total,
            "total_reserved": sum(loc["quantity_reserved"] for loc in locations),
            "is_out_of_stock_everywhere": bool(locations) and total == 0,
            "locations": locations,
        }


# ===========================================================================
# Write tools -- these change the state of the supply chain.
# ===========================================================================


def reroute_shipment(
    shipment_id: int,
    new_route: str,
    eta_shift_minutes: int = 0,
    reason: str | None = None,
    session: Session | None = None,
) -> dict[str, Any]:
    """
    Send a shipment by a different route and adjust its estimated arrival.

    Use this to recover a delayed shipment: switching it to a faster lane, a
    different distribution centre, or another carrier. Rerouting does not change
    how far along the shipment already is -- it only changes the recorded route and
    the ETA -- so it is worth doing early in a shipment's journey and rarely worth
    doing once progress_pct is close to 100. Check get_shipment_details first.

    The shipment stays in whatever live_status it had (a DELAYED shipment remains
    DELAYED, because it is still expected to arrive later than originally promised);
    rerouting records the intervention, it does not erase the delay.

    Args:
        shipment_id: The shipment to reroute.
        new_route: Human-readable description of the new path, for example
            "WH2 -> DC1 -> Customer (air freight)". Must be non-empty.
        eta_shift_minutes: How much to move the ETA, in minutes. Negative pulls
            the arrival earlier (the usual case when rerouting to recover time);
            positive pushes it later. 0 leaves the ETA unchanged. Must be between
            -20160 and 20160 (14 days either way).
        reason: Short explanation of why the shipment is being rerouted, stored for
            audit. Strongly recommended.
        session: Optional existing database session; leave unset in normal use.

    Returns:
        A dict with: shipment_id, previous_route, new_route, previous_eta, new_eta,
        eta_shift_minutes, live_status, delay_minutes and event_id (the id of the
        audit event written for this action).

    Raises:
        ToolError: If the shipment does not exist, has already been delivered,
            new_route is blank, or eta_shift_minutes is out of range.
    """
    route = (new_route or "").strip()
    if not route:
        raise ToolError("new_route must be a non-empty description of the new path")
    if len(route) > 255:
        raise ToolError(f"new_route must be 255 characters or fewer (got {len(route)})")
    if not -20_160 <= eta_shift_minutes <= 20_160:
        raise ToolError(
            f"eta_shift_minutes must be between -20160 and 20160, got {eta_shift_minutes}"
        )

    with _use_session(session) as s:
        ship = s.get(Shipment, shipment_id)
        if ship is None:
            raise ToolError(f"No shipment with id {shipment_id}")
        if ship.live_status == DELIVERED:
            raise ToolError(
                f"Shipment {shipment_id} has already been delivered and cannot be rerouted"
            )

        previous_route = ship.current_route
        previous_eta = _aware(ship.eta)

        # Preserve the very first ETA so the effect of interventions stays measurable.
        if ship.original_eta is None:
            ship.original_eta = previous_eta

        new_eta = (
            previous_eta + dt.timedelta(minutes=eta_shift_minutes)
            if previous_eta is not None
            else None
        )

        ship.current_route = route
        ship.eta = new_eta
        ship.rerouted_at = _now()
        ship.reroute_reason = reason

        # Recovering time reduces the recorded lateness, but never below zero.
        if eta_shift_minutes < 0 and ship.delay_minutes:
            ship.delay_minutes = max(0, ship.delay_minutes + eta_shift_minutes)
        elif eta_shift_minutes > 0:
            ship.delay_minutes = (ship.delay_minutes or 0) + eta_shift_minutes

        event = _record_event(
            s,
            event_type="SHIPMENT_REROUTED",
            severity="info",
            entity_type="shipment",
            entity_id=ship.id,
            message=f"Shipment {ship.id} rerouted via {route}",
            details={
                "order_id": ship.order_id,
                "previous_route": previous_route,
                "new_route": route,
                "previous_eta": _iso(previous_eta),
                "new_eta": _iso(new_eta),
                "eta_shift_minutes": eta_shift_minutes,
                "reason": reason,
            },
        )

        log.info("reroute_shipment(%s) -> %s (%+d min)", shipment_id, route, eta_shift_minutes)
        return {
            "shipment_id": ship.id,
            "previous_route": previous_route,
            "new_route": route,
            "previous_eta": _iso(previous_eta),
            "new_eta": _iso(new_eta),
            "eta_shift_minutes": eta_shift_minutes,
            "live_status": ship.live_status,
            "delay_minutes": ship.delay_minutes or 0,
            "event_id": event.id,
        }


def flag_low_inventory(
    product_id: int,
    reason: str | None = None,
    session: Session | None = None,
) -> dict[str, Any]:
    """
    Mark a product as having a stock problem that needs attention.

    Flagging is a bookkeeping action, not a fix: it records that this product is
    short and makes it findable by anyone reviewing inventory, but it does not add
    any units. Use restock_inventory to actually replenish, and escalate_alert if a
    human needs to intervene.

    Flagging an already-flagged product is safe -- it updates the reason and
    timestamp rather than failing -- so it can be called without checking first.

    Args:
        product_id: The product to flag, for example from a STOCK_OUT disruption's
            details.
        reason: Short explanation of why it is being flagged, such as
            "out of stock at WH2, 3 open orders". Stored on the product and in the
            audit event.
        session: Optional existing database session; leave unset in normal use.

    Returns:
        A dict with: product_id, product_name, low_stock_flagged (always True),
        was_already_flagged, flagged_at, reason, total_on_hand across all
        warehouses, locations_out_of_stock, and event_id.

    Raises:
        ToolError: If no product with that id exists.
    """
    with _use_session(session) as s:
        product = s.get(Product, product_id)
        if product is None:
            raise ToolError(f"No product with id {product_id}")

        was_already_flagged = bool(product.low_stock_flagged)

        product.low_stock_flagged = True
        product.flagged_at = _now()
        product.flag_reason = reason

        levels = s.execute(
            select(Inventory.quantity_on_hand).where(Inventory.product_id == product_id)
        ).scalars().all()
        total_on_hand = sum(levels)
        out_of_stock_sites = sum(1 for q in levels if q == 0)

        event = _record_event(
            s,
            event_type="INVENTORY_FLAGGED",
            severity="warning",
            entity_type="product",
            entity_id=product.id,
            message=f"Product {product.id} ({product.name}) flagged for low inventory",
            details={
                "product_name": product.name,
                "reason": reason,
                "total_on_hand": total_on_hand,
                "locations_out_of_stock": out_of_stock_sites,
                "was_already_flagged": was_already_flagged,
            },
        )

        log.info("flag_low_inventory(%s) reason=%r", product_id, reason)
        return {
            "product_id": product.id,
            "product_name": product.name,
            "low_stock_flagged": True,
            "was_already_flagged": was_already_flagged,
            "flagged_at": _iso(product.flagged_at),
            "reason": reason,
            "total_on_hand": total_on_hand,
            "locations_out_of_stock": out_of_stock_sites,
            "event_id": event.id,
        }


def escalate_alert(
    severity: Severity,
    reason: str,
    related_entity_id: int | None = None,
    related_entity_type: str | None = None,
    source_event_id: int | None = None,
    session: Session | None = None,
) -> dict[str, Any]:
    """
    Raise an alert for a human to look at.

    Use this when a situation needs a person: it is outside what the other tools can
    fix, it is repeating, or the cost of getting it wrong is high. An alert is not a
    fix and does not change the supply chain -- it creates a record in an open state
    that stays there until somebody resolves it. Prefer taking a concrete action
    (reroute_shipment, restock_inventory) when one would actually solve the problem,
    and escalate when one would not.

    Severity guide: "info" for something worth knowing but not acting on;
    "warning" for something that needs attention soon; "critical" for something
    that needs attention now, such as spoilage risk or a stock-out blocking orders.

    Args:
        severity: One of "info", "warning" or "critical".
        reason: What is wrong and why a human is needed. Required and must be
            non-empty -- this is the entire content of the alert for whoever reads
            it, so be specific ("WH2 humidity at 79% for 6 cycles, mould risk to
            stored goods" rather than "humidity problem").
        related_entity_id: Numeric id of the thing the alert is about, if any.
        related_entity_type: What that id refers to -- "shipment", "product",
            "warehouse", "inventory" or "order". Should be supplied whenever
            related_entity_id is.
        source_event_id: The id of the disruption that prompted this escalation,
            if it came from one. Links the alert back to its cause.
        session: Optional existing database session; leave unset in normal use.

    Returns:
        A dict with: alert_id, severity, reason, status (always "open"),
        created_at, related_entity_type, related_entity_id, source_event_id
        and event_id.

    Raises:
        ToolError: If severity is not a valid level, or reason is blank.
    """
    if severity not in VALID_SEVERITIES:
        raise ToolError(f"severity must be one of {VALID_SEVERITIES}, got {severity!r}")

    text = (reason or "").strip()
    if not text:
        raise ToolError("reason must be a non-empty explanation of why this is being escalated")

    with _use_session(session) as s:
        alert = Alert(
            created_at=_now(),
            severity=severity,
            reason=text,
            status="open",
            entity_type=related_entity_type,
            entity_id=related_entity_id,
            source_event_id=source_event_id,
            created_by=ACTOR,
            details={"source_event_id": source_event_id},
        )
        s.add(alert)
        s.flush()  # assign alert.id

        event = _record_event(
            s,
            event_type="ALERT_ESCALATED",
            severity=severity,
            entity_type=related_entity_type or "alert",
            entity_id=related_entity_id if related_entity_id is not None else alert.id,
            message=f"Alert {alert.id} raised ({severity}): {text[:160]}",
            details={
                "alert_id": alert.id,
                "severity": severity,
                "reason": text,
                "source_event_id": source_event_id,
            },
        )

        log.info("escalate_alert(%s) alert_id=%s", severity, alert.id)
        return {
            "alert_id": alert.id,
            "severity": alert.severity,
            "reason": alert.reason,
            "status": alert.status,
            "created_at": _iso(alert.created_at),
            "related_entity_type": alert.entity_type,
            "related_entity_id": alert.entity_id,
            "source_event_id": alert.source_event_id,
            "event_id": event.id,
        }


def restock_inventory(
    product_id: int,
    warehouse_id: int,
    quantity: int,
    reason: str | None = None,
    session: Session | None = None,
) -> dict[str, Any]:
    """
    Add units of a product to a warehouse's stock.

    This is the corrective action for a stock-out: flag_low_inventory only records
    the problem, whereas this actually raises the quantity on hand. Use it after
    checking get_inventory_status, and size the order sensibly -- restocking to
    somewhere around the reorder point rather than an arbitrary large number.

    If the product was flagged as low and this restock brings it back above its
    reorder point at every location, the flag is cleared automatically.

    Args:
        product_id: The product to restock.
        warehouse_id: Which warehouse receives the units.
        quantity: How many units to add. Must be a positive whole number no
            greater than 100000.
        reason: Short explanation, such as "emergency replenishment after
            stock-out". Stored in the audit event.
        session: Optional existing database session; leave unset in normal use.

    Returns:
        A dict with: product_id, warehouse_id, warehouse_code, quantity_added,
        quantity_before, quantity_after, reorder_point, still_below_reorder_point,
        flag_cleared and event_id.

    Raises:
        ToolError: If quantity is not positive or exceeds 100000, or if the
            product, the warehouse, or an inventory record linking the two does
            not exist.
    """
    if not isinstance(quantity, int) or isinstance(quantity, bool):
        raise ToolError(f"quantity must be a whole number, got {type(quantity).__name__}")
    if quantity <= 0:
        raise ToolError(f"quantity must be positive, got {quantity}")
    if quantity > MAX_RESTOCK_UNITS:
        raise ToolError(f"quantity must be {MAX_RESTOCK_UNITS} or fewer, got {quantity}")

    with _use_session(session) as s:
        product = s.get(Product, product_id)
        if product is None:
            raise ToolError(f"No product with id {product_id}")

        warehouse = s.get(Warehouse, warehouse_id)
        if warehouse is None:
            raise ToolError(f"No warehouse with id {warehouse_id}")

        row = s.scalars(
            select(Inventory).where(
                Inventory.product_id == product_id,
                Inventory.warehouse_id == warehouse_id,
            )
        ).first()
        if row is None:
            raise ToolError(
                f"Product {product_id} is not stocked at warehouse {warehouse_id} "
                f"({warehouse.code}); there is no inventory record to restock"
            )

        before = row.quantity_on_hand
        row.quantity_on_hand = before + quantity
        row.updated_at = _now()

        still_low = (
            row.reorder_point is not None and row.quantity_on_hand <= row.reorder_point
        )

        # Clear the flag only if every location is now healthy.
        flag_cleared = False
        if product.low_stock_flagged:
            remaining_low = s.scalar(
                select(func.count())
                .select_from(Inventory)
                .where(
                    Inventory.product_id == product_id,
                    Inventory.reorder_point.is_not(None),
                    Inventory.quantity_on_hand <= Inventory.reorder_point,
                )
            ) or 0
            if remaining_low == 0:
                product.low_stock_flagged = False
                product.flagged_at = None
                product.flag_reason = None
                flag_cleared = True

        event = _record_event(
            s,
            event_type="INVENTORY_RESTOCKED",
            severity="info",
            entity_type="inventory",
            entity_id=row.id,
            message=(
                f"Restocked {quantity} units of product {product_id} "
                f"at {warehouse.code}"
            ),
            details={
                "product_id": product_id,
                "product_name": product.name,
                "warehouse_id": warehouse_id,
                "warehouse_code": warehouse.code,
                "quantity_added": quantity,
                "quantity_before": before,
                "quantity_after": row.quantity_on_hand,
                "reason": reason,
                "flag_cleared": flag_cleared,
            },
        )

        log.info("restock_inventory(%s, %s, %d)", product_id, warehouse_id, quantity)
        return {
            "product_id": product_id,
            "warehouse_id": warehouse_id,
            "warehouse_code": warehouse.code,
            "quantity_added": quantity,
            "quantity_before": before,
            "quantity_after": row.quantity_on_hand,
            "reorder_point": row.reorder_point,
            "still_below_reorder_point": still_low,
            "flag_cleared": flag_cleared,
            "event_id": event.id,
        }


def mark_event_handled(
    event_id: int,
    resolution_note: str | None = None,
    session: Session | None = None,
) -> dict[str, Any]:
    """
    Mark a disruption as dealt with so it stops appearing in the inbox.

    Call this as the last step after responding to something from
    get_open_disruptions, whether the response was a fix, an escalation, or a
    decision that no action was needed. Without it the same disruption is returned
    on every poll and will be worked on repeatedly.

    Marking an already-handled event is safe and does not change the original
    handled-at time; the response reports was_already_handled so the caller can tell.

    Args:
        event_id: The id of the event to close out, from get_open_disruptions.
        resolution_note: Short record of what was done about it, for example
            "rerouted via DC1, ETA pulled in 4h" or "no action needed, reading
            returned to normal".
        session: Optional existing database session; leave unset in normal use.

    Returns:
        A dict with: event_id, event_type, handled_at, was_already_handled
        and resolution_note.

    Raises:
        ToolError: If no event with that id exists.
    """
    with _use_session(session) as s:
        event = s.get(Event, event_id)
        if event is None:
            raise ToolError(f"No event with id {event_id}")

        was_already_handled = event.handled_at is not None
        if not was_already_handled:
            event.handled_at = _now()

        if resolution_note:
            details = dict(event.details or {})
            details["resolution_note"] = resolution_note
            details["handled_by"] = ACTOR
            # Reassign rather than mutate: SQLAlchemy does not track in-place
            # changes to a plain JSON dict.
            event.details = details

        log.info("mark_event_handled(%s)", event_id)
        return {
            "event_id": event.id,
            "event_type": event.event_type,
            "handled_at": _iso(event.handled_at),
            "was_already_handled": was_already_handled,
            "resolution_note": resolution_note,
        }


def resolve_alert(
    alert_id: int,
    resolution_note: str,
    session: Session | None = None,
) -> dict[str, Any]:
    """
    Close an alert that has been dealt with.

    Use this when the situation behind an alert raised by escalate_alert no longer
    needs attention -- because it was fixed, or because it turned out not to matter.
    Resolving an alert does not change the supply chain; it only closes the record.

    Args:
        alert_id: The alert to close, as returned by escalate_alert.
        resolution_note: Required record of how it was resolved, for whoever
            reviews the alert history later.
        session: Optional existing database session; leave unset in normal use.

    Returns:
        A dict with: alert_id, status (always "resolved"), severity, reason,
        resolved_at, resolution_note and was_already_resolved.

    Raises:
        ToolError: If no alert with that id exists, or resolution_note is blank.
    """
    note = (resolution_note or "").strip()
    if not note:
        raise ToolError("resolution_note must be a non-empty description of how this was resolved")

    with _use_session(session) as s:
        alert = s.get(Alert, alert_id)
        if alert is None:
            raise ToolError(f"No alert with id {alert_id}")

        was_already_resolved = alert.status == "resolved"
        if not was_already_resolved:
            alert.status = "resolved"
            alert.resolved_at = _now()
        alert.resolution_note = note

        log.info("resolve_alert(%s)", alert_id)
        return {
            "alert_id": alert.id,
            "status": alert.status,
            "severity": alert.severity,
            "reason": alert.reason,
            "resolved_at": _iso(alert.resolved_at),
            "resolution_note": note,
            "was_already_resolved": was_already_resolved,
        }


# Registry for the Phase 5 adapter to enumerate, so tools are declared in one place.
READ_TOOLS = (get_open_disruptions, get_shipment_details, get_inventory_status)
WRITE_TOOLS = (
    reroute_shipment,
    flag_low_inventory,
    escalate_alert,
    restock_inventory,
    mark_event_handled,
    resolve_alert,
)
ALL_TOOLS = READ_TOOLS + WRITE_TOOLS

__all__ = [
    "ToolError",
    "get_open_disruptions",
    "get_shipment_details",
    "get_inventory_status",
    "reroute_shipment",
    "flag_low_inventory",
    "escalate_alert",
    "restock_inventory",
    "mark_event_handled",
    "resolve_alert",
    "READ_TOOLS",
    "WRITE_TOOLS",
    "ALL_TOOLS",
]
