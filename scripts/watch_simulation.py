"""
Confirm the database is actually changing while the server runs.

Start the API (uvicorn main:app --reload from backend/), then:

    python scripts/watch_simulation.py              # sample every 5s, 12 times
    python scripts/watch_simulation.py --interval 2 --samples 30

Each sample prints table counts and a few live values; the final section reports
what moved between the first and last sample, and fails loudly if nothing did.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from sqlalchemy import func, select

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from db.models import Event, Inventory, Shipment, WarehouseSensor  # noqa: E402
from db.session import SessionLocal, engine  # noqa: E402


def sample(session) -> dict:
    latest = session.execute(
        select(WarehouseSensor.reading_type, WarehouseSensor.value)
        .order_by(WarehouseSensor.id.desc())
        .limit(6)
    ).all()

    return {
        "sensor_rows": session.scalar(select(func.count()).select_from(WarehouseSensor)) or 0,
        "events": session.scalar(select(func.count()).select_from(Event)) or 0,
        "delivered": session.scalar(
            select(func.count()).select_from(Shipment).where(Shipment.live_status == "DELIVERED")
        ) or 0,
        "in_transit": session.scalar(
            select(func.count()).select_from(Shipment).where(Shipment.live_status == "IN_TRANSIT")
        ) or 0,
        "delayed": session.scalar(
            select(func.count()).select_from(Shipment).where(Shipment.live_status == "DELAYED")
        ) or 0,
        "progress_sum": int(session.scalar(select(func.coalesce(func.sum(Shipment.progress_pct), 0))) or 0),
        "stock_total": int(session.scalar(select(func.coalesce(func.sum(Inventory.quantity_on_hand), 0))) or 0),
        "latest_readings": latest,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=float, default=5.0)
    ap.add_argument("--samples", type=int, default=12)
    args = ap.parse_args()

    first = last = None
    print(f"{'t':>5}  {'sensors':>8} {'events':>7} {'transit':>8} {'delay':>6} "
          f"{'deliv':>6} {'progress':>9} {'stock':>9}")
    print("-" * 72)

    t0 = time.time()
    for i in range(args.samples):
        with SessionLocal() as session:
            snap = sample(session)
        if first is None:
            first = snap
        last = snap
        print(f"{time.time()-t0:>5.0f}  {snap['sensor_rows']:>8,} {snap['events']:>7,} "
              f"{snap['in_transit']:>8,} {snap['delayed']:>6,} {snap['delivered']:>6,} "
              f"{snap['progress_sum']:>9,} {snap['stock_total']:>9,}")
        if i < args.samples - 1:
            time.sleep(args.interval)

    print("\nlatest sensor readings:")
    for reading_type, value in (last or {}).get("latest_readings", []):
        print(f"  {reading_type:<24} {value}")

    with SessionLocal() as session:
        recent = session.scalars(
            select(Event).order_by(Event.occurred_at.desc()).limit(8)
        ).all()
    if recent:
        print("\nrecent events:")
        for e in recent:
            tag = "injected" if e.injected else "organic "
            print(f"  {e.occurred_at:%H:%M:%S} [{tag}] {e.severity:<8} {e.event_type:<20} {e.message}")

    print("\n--- change between first and last sample ---")
    moved = []
    for key in ("sensor_rows", "events", "delivered", "progress_sum", "stock_total"):
        delta = last[key] - first[key]
        flag = "moved" if delta else "NO CHANGE"
        print(f"  {key:<14} {first[key]:>10,} -> {last[key]:>10,}  ({delta:+,})  {flag}")
        if delta:
            moved.append(key)

    engine.dispose()

    if not moved:
        print("\nFAIL: nothing changed. Is the server running with SIM_ENABLED=1?")
        return 1
    print(f"\nOK: {len(moved)} counters moved -- the database is live.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
