"""
Background worker that makes the database behave like a live system.

One cycle, every SIM_CYCLE_SECONDS:

  1. top up the in-transit pool, advance shipments toward delivery, deliver the
     ones that arrive;
  2. draw inventory down for products on active orders, restocking anything that
     falls near zero;
  3. write a fresh warehouse_sensors reading for every node;
  4. roll for a disruption and, if it hits, inject one and log it to `events`.

Concurrency model: this is an asyncio task, but the database work is synchronous
SQLAlchemy run via asyncio.to_thread, so it shares one pooled engine with the rest
of the app rather than introducing a second async engine and driver. Each cycle
takes exactly one session out of the pool with a context manager and returns it,
whether the cycle succeeds or raises.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import random
import time
from datetime import datetime, timedelta, timezone

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from db.models import Event, Inventory, Order, OrderItem, Shipment, Warehouse, WarehouseSensor
from db.session import SessionLocal

from . import config, sensors

log = logging.getLogger("simulation")

# Order statuses that represent live demand pulling stock.
ACTIVE_ORDER_STATUSES = ("PENDING", "PROCESSING", "PENDING_PAYMENT", "PAYMENT_REVIEW")

PENDING = "PENDING"
IN_TRANSIT = "IN_TRANSIT"
DELAYED = "DELAYED"
DELIVERED = "DELIVERED"

LIVE_ACTIVE = (PENDING, IN_TRANSIT, DELAYED)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _aware(value: datetime | None) -> datetime | None:
    """
    Normalize a datetime read back from the database to UTC-aware.

    Postgres TIMESTAMPTZ round-trips as aware, but SQLite (and any column that
    ends up naive) does not, and mixing the two raises on comparison.
    """
    if value is None:
        return None
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value


class SimulationWorker:
    """Owns the background task and the heartbeat state that /health reports."""

    def __init__(self, on_cycle=None) -> None:
        self._task: asyncio.Task | None = None
        self._stopping = asyncio.Event()
        # Optional async callback invoked after each successful cycle, used to
        # push the fresh twin state to WebSocket clients.
        self._on_cycle = on_cycle

        self.started_at: datetime | None = None
        self.last_run_at: datetime | None = None
        self.last_success_at: datetime | None = None
        self.cycles_completed = 0
        self.cycles_failed = 0
        self.consecutive_failures = 0
        self.last_error: str | None = None
        self.last_cycle_ms: float | None = None
        self.last_cycle_summary: dict | None = None
        self.disruptions_injected = 0
        self.last_disruption_at: datetime | None = None

        # Reading types currently forced into spike, as {(warehouse_id, type): cycles_left}
        self._active_spikes: dict[tuple[int, str], int] = {}

    # -- lifecycle ---------------------------------------------------------

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    def start(self) -> None:
        if self.running:
            return
        self._stopping.clear()
        self.started_at = _now()
        self._task = asyncio.create_task(self._run(), name="simulation-worker")
        log.info(
            "simulation worker started (cycle=%.1fs, disruption_rate=%.3f)",
            config.CYCLE_SECONDS,
            config.DISRUPTION_RATE,
        )

    async def stop(self) -> None:
        """Signal the loop, then wait for the in-flight cycle to finish."""
        if self._task is None:
            return
        self._stopping.set()
        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._task
        self._task = None
        log.info("simulation worker stopped after %d cycles", self.cycles_completed)

    async def _run(self) -> None:
        while not self._stopping.is_set():
            delay = config.CYCLE_SECONDS
            self.last_run_at = _now()
            started = time.perf_counter()
            try:
                # Sync DB work off the event loop. The session is opened and closed
                # inside this call, so nothing is held across an await.
                summary = await asyncio.to_thread(self._cycle)
                self.last_cycle_ms = (time.perf_counter() - started) * 1000
                self.last_success_at = _now()
                self.cycles_completed += 1
                self.consecutive_failures = 0
                self.last_error = None
                self.last_cycle_summary = summary
                log.debug("cycle %d: %s", self.cycles_completed, summary)
                if self._on_cycle is not None:
                    # A broadcast failure must never fail the simulation cycle.
                    with contextlib.suppress(Exception):
                        await self._on_cycle()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - the loop must survive anything
                self.cycles_failed += 1
                self.consecutive_failures += 1
                self.last_error = f"{type(exc).__name__}: {exc}"
                log.exception("simulation cycle failed (%d in a row)", self.consecutive_failures)
                if self.consecutive_failures >= 3:
                    delay = config.FAILURE_BACKOFF_SECONDS

            # Sleep interruptibly so shutdown doesn't wait out a full cycle.
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self._stopping.wait(), timeout=delay)

    # -- one cycle ---------------------------------------------------------

    def _cycle(self) -> dict:
        """Runs on a worker thread. One session, one transaction, always returned."""
        with SessionLocal() as session:
            warehouses = session.scalars(select(Warehouse)).all()
            if not warehouses:
                # Dataset not loaded yet -- idle rather than crash-loop.
                return {"skipped": "no warehouses; run scripts/load_dataset.py"}

            summary: dict = {}
            summary.update(self._advance_shipments(session))
            summary.update(self._draw_inventory(session))
            summary.update(self._write_sensors(session, warehouses))
            summary.update(self._maybe_disrupt(session, warehouses))

            if (
                config.SENSOR_RETENTION_ROWS
                and self.cycles_completed
                and self.cycles_completed % config.SENSOR_TRIM_EVERY_CYCLES == 0
            ):
                summary["sensors_trimmed"] = self._trim_sensors(session)

            session.commit()
            return summary

    # -- 1. shipments ------------------------------------------------------

    def _advance_shipments(self, session: Session) -> dict:
        now = _now()

        # Top the in-flight pool back up from shipments that have never been live.
        in_flight = session.scalar(
            select(func.count()).select_from(Shipment).where(Shipment.live_status.in_(LIVE_ACTIVE))
        ) or 0

        launched = 0
        if in_flight < config.IN_TRANSIT_TARGET:
            wanted = config.IN_TRANSIT_TARGET - in_flight
            fresh = session.scalars(
                select(Shipment).where(Shipment.live_status.is_(None)).limit(wanted)
            ).all()
            for ship in fresh:
                hours = max(1, (ship.days_scheduled or 2)) * 24
                ship.live_status = IN_TRANSIT
                ship.progress_pct = 0
                ship.eta = now + timedelta(hours=hours)
                ship.delay_minutes = 0
                launched += 1

        # Advance a slice of what's in flight.
        moving = session.scalars(
            select(Shipment)
            .where(Shipment.live_status.in_((IN_TRANSIT, DELAYED)))
            .order_by(func.random())
            .limit(config.SHIPMENTS_PER_CYCLE)
        ).all()

        advanced = delivered = 0
        for ship in moving:
            step = random.randint(config.PROGRESS_MIN, config.PROGRESS_MAX)
            ship.progress_pct = min(100, (ship.progress_pct or 0) + step)
            advanced += 1

            if ship.progress_pct >= 100:
                ship.live_status = DELIVERED
                ship.delivered_at = now
                ship.progress_pct = 100
                delivered += 1
                # A delayed shipment landing is worth recording for the agent.
                if (ship.delay_minutes or 0) > 0:
                    self._log_event(
                        session,
                        event_type="SHIPMENT_DELIVERED",
                        severity="info",
                        entity_type="shipment",
                        entity_id=ship.id,
                        message=f"Shipment {ship.id} delivered {ship.delay_minutes} min late",
                        details={"order_id": ship.order_id, "delay_minutes": ship.delay_minutes},
                    )
            # A shipment that has been delayed stays DELAYED until it is delivered.
            # Pushing the ETA out does not un-delay it, so the agent still sees the
            # condition on the next poll.

        return {"launched": launched, "advanced": advanced, "delivered": delivered}

    # -- 2. inventory ------------------------------------------------------

    def _draw_inventory(self, session: Session) -> dict:
        """
        Pull stock for products attached to live demand: items on orders that are
        still open, or on shipments currently in flight.
        """
        product_ids = session.scalars(
            select(OrderItem.product_id)
            .join(Order, Order.id == OrderItem.order_id)
            .outerjoin(Shipment, Shipment.order_id == Order.id)
            .where(
                OrderItem.product_id.is_not(None),
                (Order.status.in_(ACTIVE_ORDER_STATUSES))
                | (Shipment.live_status.in_((IN_TRANSIT, DELAYED))),
            )
            .distinct()
            .limit(200)
        ).all()

        if not product_ids:
            return {"drawn": 0}

        picks = random.sample(
            list(product_ids), min(config.INVENTORY_DRAWS_PER_CYCLE, len(product_ids))
        )

        drawn = restocked = stockouts = 0
        for product_id in picks:
            rows = session.scalars(
                select(Inventory).where(Inventory.product_id == product_id)
            ).all()
            if not rows:
                continue

            row = random.choice(rows)
            units = random.randint(1, 5)

            if row.quantity_on_hand <= 0:
                continue

            if row.quantity_on_hand < units:
                # Demand exceeded stock -- a genuine stock-out, not an injected one.
                units = row.quantity_on_hand
                row.quantity_on_hand = 0
                stockouts += 1
                self._log_event(
                    session,
                    event_type="STOCK_OUT",
                    severity="warning",
                    entity_type="inventory",
                    entity_id=row.id,
                    message=(
                        f"Product {product_id} out of stock at warehouse {row.warehouse_id}"
                    ),
                    details={
                        "product_id": product_id,
                        "warehouse_id": row.warehouse_id,
                        "cause": "demand_exceeded_stock",
                    },
                )
            else:
                row.quantity_on_hand -= units

            row.updated_at = _now()
            drawn += 1

            # Replenish so a long run doesn't grind every product to zero.
            # seed_inventory set reorder_point to 25% of opening stock, so
            # restocking to reorder_point * 4 returns roughly to that opening level.
            reorder = row.reorder_point or 0
            if reorder > 0 and row.quantity_on_hand <= reorder:
                row.quantity_on_hand = max(int(reorder * config.RESTOCK_MULTIPLIER), reorder + 1)
                restocked += 1

        return {"drawn": drawn, "restocked": restocked, "stockouts": stockouts}

    # -- 3. sensors --------------------------------------------------------

    def _write_sensors(self, session: Session, warehouses) -> dict:
        now = _now()
        rows: list[dict] = []
        expired: list[tuple[int, str]] = []

        for wh in warehouses:
            for reading_type, value, unit in sensors.readings_for(wh.kind):
                key = (wh.id, reading_type)
                remaining = self._active_spikes.get(key)
                if remaining:
                    spiked = sensors.spike_value(reading_type)
                    if spiked is not None:
                        value = spiked
                    self._active_spikes[key] = remaining - 1
                    if self._active_spikes[key] <= 0:
                        expired.append(key)

                rows.append({
                    "warehouse_id": wh.id,
                    "reading_type": reading_type,
                    "value": value,
                    "unit": unit,
                    "recorded_at": now,
                })

        for key in expired:
            self._active_spikes.pop(key, None)

        session.bulk_insert_mappings(WarehouseSensor, rows)
        return {"sensor_rows": len(rows), "spikes_active": len(self._active_spikes)}

    def _trim_sensors(self, session: Session) -> int:
        """Keep only the newest SENSOR_RETENTION_ROWS readings."""
        total = session.scalar(select(func.count()).select_from(WarehouseSensor)) or 0
        excess = total - config.SENSOR_RETENTION_ROWS
        if excess <= 0:
            return 0
        cutoff = session.scalar(
            select(WarehouseSensor.id).order_by(WarehouseSensor.id).offset(excess).limit(1)
        )
        if cutoff is None:
            return 0
        session.execute(WarehouseSensor.__table__.delete().where(WarehouseSensor.id < cutoff))
        return excess

    # -- 4. disruptions ----------------------------------------------------

    def _maybe_disrupt(self, session: Session, warehouses) -> dict:
        now = _now()

        if self.last_disruption_at is not None:
            since = (now - self.last_disruption_at).total_seconds()
            if since < config.DISRUPTION_COOLDOWN_SECONDS:
                return {}

        if random.random() >= config.DISRUPTION_RATE:
            return {}

        kinds = list(config.DISRUPTION_WEIGHTS)
        weights = [config.DISRUPTION_WEIGHTS[k] for k in kinds]
        kind = random.choices(kinds, weights=weights, k=1)[0]

        handler = {
            "SHIPMENT_DELAY": self._inject_shipment_delay,
            "SENSOR_SPIKE": self._inject_sensor_spike,
            "STOCK_OUT": self._inject_stock_out,
        }[kind]

        if not handler(session, warehouses, now):
            # Nothing eligible to disrupt this cycle (e.g. no shipments in flight).
            return {}

        self.disruptions_injected += 1
        self.last_disruption_at = now
        log.info("injected disruption: %s", kind)
        return {"disruption": kind}

    def _inject_shipment_delay(self, session: Session, _warehouses, now: datetime) -> bool:
        ship = session.scalars(
            select(Shipment)
            .where(Shipment.live_status.in_((IN_TRANSIT, DELAYED)))
            .order_by(func.random())
            .limit(1)
        ).first()
        if ship is None:
            return False

        minutes = random.randint(config.DELAY_MINUTES_MIN, config.DELAY_MINUTES_MAX)
        previous_eta = _aware(ship.eta)
        ship.eta = (previous_eta or now) + timedelta(minutes=minutes)
        ship.delay_minutes = (ship.delay_minutes or 0) + minutes
        ship.live_status = DELAYED

        self._log_event(
            session,
            event_type="SHIPMENT_DELAY",
            severity="critical" if ship.delay_minutes >= 480 else "warning",
            entity_type="shipment",
            entity_id=ship.id,
            message=f"Shipment {ship.id} delayed by {minutes} min",
            details={
                "order_id": ship.order_id,
                "shipping_mode": ship.shipping_mode,
                "delay_minutes": minutes,
                "cumulative_delay_minutes": ship.delay_minutes,
                "previous_eta": previous_eta.isoformat() if previous_eta else None,
                "new_eta": ship.eta.isoformat(),
                "progress_pct": ship.progress_pct,
            },
            injected=True,
        )
        return True

    def _inject_sensor_spike(self, session: Session, warehouses, now: datetime) -> bool:
        candidates = []
        for wh in warehouses:
            for reading_type, _value, _unit in sensors.readings_for(wh.kind):
                if reading_type in sensors.SPIKEABLE:
                    candidates.append((wh, reading_type))
        if not candidates:
            return False

        wh, reading_type = random.choice(candidates)
        cycles = random.randint(config.SPIKE_CYCLES_MIN, config.SPIKE_CYCLES_MAX)
        self._active_spikes[(wh.id, reading_type)] = cycles

        peak = sensors.spike_value(reading_type)
        band = sensors.normal_range(reading_type, wh.kind or "warehouse")

        self._log_event(
            session,
            event_type="SENSOR_SPIKE",
            severity="critical",
            entity_type="warehouse",
            entity_id=wh.id,
            message=f"{reading_type} anomaly at {wh.code} (~{peak})",
            details={
                "warehouse_code": wh.code,
                "reading_type": reading_type,
                "approx_value": peak,
                "normal_range": band,
                "duration_cycles": cycles,
            },
            injected=True,
        )
        return True

    def _inject_stock_out(self, session: Session, _warehouses, now: datetime) -> bool:
        row = session.scalars(
            select(Inventory)
            .where(Inventory.quantity_on_hand > 0)
            .order_by(func.random())
            .limit(1)
        ).first()
        if row is None:
            return False

        before = row.quantity_on_hand
        row.quantity_on_hand = 0
        row.updated_at = now

        self._log_event(
            session,
            event_type="STOCK_OUT",
            severity="critical",
            entity_type="inventory",
            entity_id=row.id,
            message=(
                f"Stock-out: product {row.product_id} at warehouse {row.warehouse_id}"
            ),
            details={
                "product_id": row.product_id,
                "warehouse_id": row.warehouse_id,
                "quantity_before": before,
                "reorder_point": row.reorder_point,
                "cause": "injected",
            },
            injected=True,
        )
        return True

    # -- events ------------------------------------------------------------

    def _log_event(
        self,
        session: Session,
        *,
        event_type: str,
        severity: str,
        entity_type: str | None = None,
        entity_id: int | None = None,
        message: str | None = None,
        details: dict | None = None,
        injected: bool = False,
    ) -> None:
        session.add(
            Event(
                occurred_at=_now(),
                event_type=event_type,
                severity=severity,
                entity_type=entity_type,
                entity_id=entity_id,
                message=message,
                details=details,
                injected=injected,
            )
        )

    # -- health ------------------------------------------------------------

    def status(self) -> dict:
        now = _now()

        def age(ts: datetime | None) -> float | None:
            return None if ts is None else round((now - ts).total_seconds(), 2)

        stale_after = config.CYCLE_SECONDS * 3
        last_age = age(self.last_success_at)
        healthy = (
            self.running
            and self.consecutive_failures == 0
            and (last_age is not None and last_age <= stale_after)
        )

        return {
            "running": self.running,
            "healthy": healthy,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "last_run_at": self.last_run_at.isoformat() if self.last_run_at else None,
            "last_success_at": (
                self.last_success_at.isoformat() if self.last_success_at else None
            ),
            "seconds_since_last_success": last_age,
            "cycle_seconds": config.CYCLE_SECONDS,
            "cycles_completed": self.cycles_completed,
            "cycles_failed": self.cycles_failed,
            "consecutive_failures": self.consecutive_failures,
            "last_cycle_ms": round(self.last_cycle_ms, 1) if self.last_cycle_ms else None,
            "last_cycle": self.last_cycle_summary,
            "last_error": self.last_error,
            "disruptions_injected": self.disruptions_injected,
            "disruption_rate": config.DISRUPTION_RATE,
            "last_disruption_at": (
                self.last_disruption_at.isoformat() if self.last_disruption_at else None
            ),
            "active_sensor_spikes": len(self._active_spikes),
        }


worker = SimulationWorker()
