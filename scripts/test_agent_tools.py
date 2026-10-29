"""
Exercise every agent tool against the dev database and verify the rows really change.

No LLM involved -- this calls the plain Python functions directly and re-reads the
database independently of the tool's own return value, so a tool that lies about
what it did still fails the check.

    python scripts/test_agent_tools.py
    python scripts/test_agent_tools.py --keep     # don't roll back at the end

By default every change is rolled back, so it is safe to point at a database you
care about. Pass --keep to leave the writes in place and inspect them by hand.
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

from sqlalchemy import func, select

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from agent import tools  # noqa: E402
from agent.tools import ToolError  # noqa: E402
from db.models import Alert, Event, Inventory, Product, Shipment, Warehouse  # noqa: E402
from db.session import SessionLocal, engine  # noqa: E402

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    results.append((PASS if condition else FAIL, name, detail))
    mark = "  ok  " if condition else " FAIL "
    print(f"[{mark}] {name}" + (f"  -- {detail}" if detail else ""))


def expect_error(name: str, fn, *args, **kwargs) -> None:
    """Assert that a tool rejects bad input with ToolError rather than doing it anyway."""
    try:
        fn(*args, **kwargs)
    except ToolError as exc:
        check(name, True, f"raised ToolError: {str(exc)[:70]}")
    except Exception as exc:  # noqa: BLE001
        check(name, False, f"raised {type(exc).__name__} instead of ToolError: {exc}")
    else:
        check(name, False, "no error raised -- invalid input was accepted")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true", help="commit instead of rolling back")
    args = ap.parse_args()

    session = SessionLocal()
    # Everything runs inside one transaction that is rolled back unless --keep.
    # Tools accept this session, so they join the transaction instead of
    # committing on their own.
    try:
        run_all(session)
    finally:
        if args.keep:
            session.commit()
            print("\n(--keep) changes committed")
        else:
            session.rollback()
            print("\nrolled back -- database unchanged")
        session.close()

    # Outside the rolled-back transaction, so the commit path is really exercised.
    run_standalone_session_checks()
    engine.dispose()

    print("\n" + "=" * 62)
    failed = [r for r in results if r[0] == FAIL]
    print(f"{len(results) - len(failed)}/{len(results)} checks passed")
    if failed:
        print("\nfailures:")
        for _, name, detail in failed:
            print(f"  - {name}: {detail}")
        return 1
    print("all agent tools behave correctly against the database")
    return 0


def run_all(s) -> None:
    # ---------------------------------------------------------------- fixtures
    ship = s.scalars(
        select(Shipment).where(Shipment.live_status.in_(tools.ACTIVE_SHIPMENT_STATES)).limit(1)
    ).first()
    delivered = s.scalars(
        select(Shipment).where(Shipment.live_status == "DELIVERED").limit(1)
    ).first()
    inv = s.scalars(select(Inventory).limit(1)).first()

    if ship is None or inv is None:
        print("\n!! Need an active shipment and an inventory row. Start the server for a")
        print("   few cycles (or run scripts/load_dataset.py) before running this test.")
        results.append((FAIL, "fixtures available", "no active shipment / inventory"))
        return

    product = s.get(Product, inv.product_id)
    warehouse = s.get(Warehouse, inv.warehouse_id)
    print(f"\nfixtures: shipment {ship.id} ({ship.live_status}), "
          f"product {product.id}, warehouse {warehouse.code}\n")

    # ============================================================ READ TOOLS
    print("-- read tools " + "-" * 46)

    disruptions = tools.get_open_disruptions(limit=5, session=s)
    check("get_open_disruptions returns a list", isinstance(disruptions, list),
          f"{len(disruptions)} open")
    if disruptions:
        d = disruptions[0]
        check("disruption has the documented shape",
              {"id", "occurred_at", "event_type", "severity", "message", "age_seconds"} <= set(d))
        check("get_open_disruptions returns only unhandled events",
              all(s.get(Event, x["id"]).handled_at is None for x in disruptions))

    crit = tools.get_open_disruptions(limit=50, min_severity="critical", session=s)
    check("min_severity filters correctly",
          all(x["severity"] == "critical" for x in crit), f"{len(crit)} critical")

    details = tools.get_shipment_details(ship.id, session=s)
    check("get_shipment_details returns the right shipment", details["shipment_id"] == ship.id)
    check("get_shipment_details marks active shipment reroutable", details["is_reroutable"] is True)
    if delivered is not None:
        d2 = tools.get_shipment_details(delivered.id, session=s)
        check("delivered shipment reported as not reroutable", d2["is_reroutable"] is False)

    stock = tools.get_inventory_status(product.id, session=s)
    db_total = s.scalar(
        select(func.coalesce(func.sum(Inventory.quantity_on_hand), 0))
        .where(Inventory.product_id == product.id)
    )
    check("get_inventory_status total matches the database",
          stock["total_on_hand"] == db_total, f"tool={stock['total_on_hand']} db={db_total}")
    check("get_inventory_status lists every location",
          len(stock["locations"]) == len(
              s.scalars(select(Inventory).where(Inventory.product_id == product.id)).all()))

    expect_error("get_shipment_details rejects unknown id", tools.get_shipment_details,
                 99_999_999, session=s)
    expect_error("get_inventory_status rejects unknown product", tools.get_inventory_status,
                 99_999_999, session=s)
    expect_error("get_open_disruptions rejects bad severity", tools.get_open_disruptions,
                 min_severity="urgent", session=s)
    expect_error("get_open_disruptions rejects bad limit", tools.get_open_disruptions,
                 limit=0, session=s)

    # ====================================================== reroute_shipment
    print("\n-- reroute_shipment " + "-" * 41)

    before_eta = ship.eta
    before_delay = ship.delay_minutes or 0
    events_before = s.scalar(select(func.count()).select_from(Event))

    out = tools.reroute_shipment(
        ship.id, "WH2 -> DC1 -> Customer (air)", eta_shift_minutes=-120,
        reason="recovering a delay", session=s,
    )
    s.flush()
    s.refresh(ship)

    check("route written to the database", ship.current_route == "WH2 -> DC1 -> Customer (air)",
          f"db={ship.current_route!r}")
    check("rerouted_at stamped", ship.rerouted_at is not None)
    check("reroute_reason stored", ship.reroute_reason == "recovering a delay")
    check("original_eta preserved", ship.original_eta is not None)
    if before_eta is not None:
        moved = round((tools._aware(ship.eta) - tools._aware(before_eta)).total_seconds() / 60)
        check("ETA shifted by exactly -120 minutes", moved == -120, f"moved {moved} min")
    check("delay_minutes reduced but not negative",
          ship.delay_minutes == max(0, before_delay - 120), f"{before_delay} -> {ship.delay_minutes}")

    audit = s.get(Event, out["event_id"])
    check("reroute wrote an audit event", audit is not None and audit.event_type == "SHIPMENT_REROUTED")
    check("audit event attributed to the agent", audit.actor == "agent", f"actor={audit.actor}")
    check("audit event not marked injected", audit.injected is False)
    check("exactly one event added",
          s.scalar(select(func.count()).select_from(Event)) == events_before + 1)

    expect_error("reroute rejects blank route", tools.reroute_shipment, ship.id, "   ", session=s)
    expect_error("reroute rejects absurd eta shift", tools.reroute_shipment, ship.id,
                 "X -> Y", eta_shift_minutes=999_999, session=s)
    expect_error("reroute rejects unknown shipment", tools.reroute_shipment, 99_999_999,
                 "X -> Y", session=s)
    if delivered is not None:
        expect_error("reroute refuses a delivered shipment", tools.reroute_shipment,
                     delivered.id, "X -> Y", session=s)

    # ==================================================== flag_low_inventory
    print("\n-- flag_low_inventory " + "-" * 39)

    s.refresh(product)
    was_flagged = product.low_stock_flagged
    out = tools.flag_low_inventory(product.id, reason="stock-out at WH2", session=s)
    s.flush()
    s.refresh(product)

    check("product marked flagged in the database", product.low_stock_flagged is True)
    check("flagged_at stamped", product.flagged_at is not None)
    check("flag_reason stored", product.flag_reason == "stock-out at WH2")
    check("was_already_flagged reported correctly", out["was_already_flagged"] == was_flagged)
    fev = s.get(Event, out["event_id"])
    check("flag wrote an INVENTORY_FLAGGED event", fev.event_type == "INVENTORY_FLAGGED")

    out2 = tools.flag_low_inventory(product.id, reason="second call", session=s)
    s.flush()
    check("re-flagging is idempotent, not an error", out2["was_already_flagged"] is True)

    expect_error("flag rejects unknown product", tools.flag_low_inventory, 99_999_999, session=s)

    # ======================================================= escalate_alert
    print("\n-- escalate_alert " + "-" * 43)

    alerts_before = s.scalar(select(func.count()).select_from(Alert))
    out = tools.escalate_alert(
        "critical", "WH2 humidity 79% for 6 cycles, mould risk",
        related_entity_id=warehouse.id, related_entity_type="warehouse",
        source_event_id=disruptions[0]["id"] if disruptions else None,
        session=s,
    )
    s.flush()

    alert = s.get(Alert, out["alert_id"])
    check("alert row created", alert is not None)
    check("alert count incremented by one",
          s.scalar(select(func.count()).select_from(Alert)) == alerts_before + 1)
    check("alert severity stored", alert.severity == "critical")
    check("alert opens in 'open' status", alert.status == "open")
    check("alert reason stored", alert.reason.startswith("WH2 humidity"))
    check("alert linked to entity",
          alert.entity_type == "warehouse" and alert.entity_id == warehouse.id)
    check("alert attributed to the agent", alert.created_by == "agent")
    check("escalation wrote an audit event",
          s.get(Event, out["event_id"]).event_type == "ALERT_ESCALATED")

    expect_error("escalate rejects invalid severity", tools.escalate_alert,
                 "catastrophic", "boom", session=s)
    expect_error("escalate rejects blank reason", tools.escalate_alert, "warning", "  ", session=s)

    # ====================================================== restock_inventory
    print("\n-- restock_inventory " + "-" * 40)

    s.refresh(inv)
    qty_before = inv.quantity_on_hand
    out = tools.restock_inventory(product.id, warehouse.id, 250,
                                  reason="emergency replenishment", session=s)
    s.flush()
    s.refresh(inv)

    check("quantity_on_hand increased by exactly 250",
          inv.quantity_on_hand == qty_before + 250, f"{qty_before} -> {inv.quantity_on_hand}")
    check("tool reported the true before value", out["quantity_before"] == qty_before)
    check("tool reported the true after value", out["quantity_after"] == inv.quantity_on_hand)
    check("restock wrote an audit event",
          s.get(Event, out["event_id"]).event_type == "INVENTORY_RESTOCKED")

    expect_error("restock rejects zero quantity", tools.restock_inventory,
                 product.id, warehouse.id, 0, session=s)
    expect_error("restock rejects negative quantity", tools.restock_inventory,
                 product.id, warehouse.id, -50, session=s)
    expect_error("restock rejects absurd quantity", tools.restock_inventory,
                 product.id, warehouse.id, 10_000_000, session=s)
    expect_error("restock rejects non-integer quantity", tools.restock_inventory,
                 product.id, warehouse.id, 12.5, session=s)
    expect_error("restock rejects unknown warehouse", tools.restock_inventory,
                 product.id, 99_999_999, 10, session=s)

    # confirm the negative-stock invariant can't be violated through this path
    check("restock never lowers stock", inv.quantity_on_hand >= qty_before)

    # ===================================================== mark_event_handled
    print("\n-- mark_event_handled " + "-" * 39)

    target = s.scalars(
        select(Event).where(Event.handled_at.is_(None)).order_by(Event.id.desc()).limit(1)
    ).first()
    if target is not None:
        out = tools.mark_event_handled(target.id, "rerouted, ETA pulled in 2h", session=s)
        s.flush()
        s.refresh(target)
        check("handled_at stamped in the database", target.handled_at is not None)
        check("was_already_handled False on first call", out["was_already_handled"] is False)
        check("resolution note persisted into details",
              (target.details or {}).get("resolution_note") == "rerouted, ETA pulled in 2h")

        out2 = tools.mark_event_handled(target.id, "second call", session=s)
        s.flush()
        check("re-handling is idempotent", out2["was_already_handled"] is True)

        still_open = tools.get_open_disruptions(limit=200, min_severity="info", session=s)
        check("handled event no longer appears in the inbox",
              target.id not in {x["id"] for x in still_open})

    expect_error("mark_event_handled rejects unknown id", tools.mark_event_handled,
                 99_999_999, session=s)

    # ========================================================== resolve_alert
    print("\n-- resolve_alert " + "-" * 44)

    out = tools.resolve_alert(alert.id, "dehumidifier restarted, reading normal", session=s)
    s.flush()
    s.refresh(alert)
    check("alert status set to resolved", alert.status == "resolved")
    check("resolved_at stamped", alert.resolved_at is not None)
    check("resolution note stored", alert.resolution_note.startswith("dehumidifier"))
    out2 = tools.resolve_alert(alert.id, "again", session=s)
    check("re-resolving is idempotent", out2["was_already_resolved"] is True)

    expect_error("resolve_alert rejects blank note", tools.resolve_alert, alert.id, "", session=s)
    expect_error("resolve_alert rejects unknown id", tools.resolve_alert,
                 99_999_999, "x", session=s)

    # ============================================ session handling / contract
    print("\n-- contract " + "-" * 49)

    import inspect
    missing_doc = [f.__name__ for f in tools.ALL_TOOLS if not (f.__doc__ or "").strip()]
    check("every tool has a docstring", not missing_doc, str(missing_doc))

    thin = [f.__name__ for f in tools.ALL_TOOLS if len((f.__doc__ or "")) < 300]
    check("docstrings are substantial enough to be tool descriptions", not thin, str(thin))

    unannotated = []
    for f in tools.ALL_TOOLS:
        sig = inspect.signature(f)
        for pname, p in sig.parameters.items():
            if p.annotation is inspect.Parameter.empty:
                unannotated.append(f"{f.__name__}.{pname}")
    check("every tool parameter is type-annotated", not unannotated, str(unannotated))

    no_session = [f.__name__ for f in tools.ALL_TOOLS
                  if "session" not in inspect.signature(f).parameters]
    check("every tool accepts an optional session", not no_session, str(no_session))

    check("all writes are visible in one transaction",
          s.scalar(select(func.count()).select_from(Event)) > events_before)


def run_standalone_session_checks() -> None:
    """
    Verify the no-session path: a tool called without a session must open, commit
    and close its own. This is how the agent will actually call them, and it cannot
    be tested inside the rolled-back transaction above.

    Runs outside that transaction and undoes its own writes.
    """
    print("\n-- standalone session (self-committing) " + "-" * 21)

    with SessionLocal() as probe:
        inv = probe.scalars(select(Inventory).limit(1)).first()
        if inv is None:
            check("standalone fixtures available", False, "no inventory rows")
            return
        product_id = inv.product_id
        warehouse_id = inv.warehouse_id
        qty_before = inv.quantity_on_hand
        was_flagged = probe.get(Product, product_id).low_stock_flagged

    # No session argument -- the tool must manage its own.
    out = tools.restock_inventory(product_id, warehouse_id, 7, reason="standalone commit check")

    with SessionLocal() as verify:
        row = verify.scalars(
            select(Inventory).where(
                Inventory.product_id == product_id, Inventory.warehouse_id == warehouse_id
            )
        ).first()
        check("write committed without an injected session",
              row.quantity_on_hand == qty_before + 7,
              f"{qty_before} -> {row.quantity_on_hand}")
        check("audit event committed too", verify.get(Event, out["event_id"]) is not None)

    # A failing call must roll back its own session and leave nothing behind.
    events_before = None
    with SessionLocal() as probe:
        events_before = probe.scalar(select(func.count()).select_from(Event))
    try:
        tools.restock_inventory(product_id, 99_999_999, 5)
    except ToolError:
        pass
    with SessionLocal() as probe:
        check("failed call leaves no partial write",
              probe.scalar(select(func.count()).select_from(Event)) == events_before)

    # Undo the standalone writes so the database is left as found.
    with SessionLocal() as cleanup:
        row = cleanup.scalars(
            select(Inventory).where(
                Inventory.product_id == product_id, Inventory.warehouse_id == warehouse_id
            )
        ).first()
        row.quantity_on_hand = qty_before
        prod = cleanup.get(Product, product_id)
        prod.low_stock_flagged = was_flagged
        cleanup.execute(Event.__table__.delete().where(Event.id == out["event_id"]))
        cleanup.commit()
    print("     (standalone writes undone)")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(2)
