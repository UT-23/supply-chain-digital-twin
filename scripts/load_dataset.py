"""
One-time load of the DataCo Smart Supply Chain CSV into Postgres.

    python scripts/load_dataset.py                 # load (refuses if tables non-empty)
    python scripts/load_dataset.py --truncate      # wipe the loaded tables first
    python scripts/load_dataset.py --limit 5000    # smoke test on a slice
    python scripts/load_dataset.py --skip-inventory

Assumes `alembic upgrade head` has already created the schema.

Order of insertion respects FKs:
    departments -> categories -> products -> customers -> orders -> order_items
                -> shipments -> warehouses -> inventory

`inventory` is NOT in the DataCo dataset -- the CSV has no stock levels. It is seeded
from observed demand (see seed_inventory) so the twin has plausible stock to render.
`warehouse_sensors` is left empty here; it is live telemetry, not historical data.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pandas as pd
from sqlalchemy import func, select

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from db.models import (  # noqa: E402
    Category,
    Customer,
    Department,
    Inventory,
    Order,
    OrderItem,
    Product,
    Shipment,
    Warehouse,
    WarehouseSensor,
)
from db.session import SessionLocal, engine  # noqa: E402
from load_helpers import (  # noqa: E402
    DatasetError,
    as_bool,
    as_datetime,
    as_decimal,
    as_int,
    as_str,
    as_zip,
    find_dataset,
    read_dataset,
    resolve_columns,
)

CHUNK = 5_000

# Warehouses are not in the dataset either. These are the nodes the 3D scene renders;
# ids are stable so the frontend can reference them.
WAREHOUSE_SEED = [
    {"code": "WH1", "name": "Warehouse 1", "kind": "warehouse",
     "city": "Los Angeles", "country": "USA",
     "latitude": Decimal("34.052200"), "longitude": Decimal("-118.243700")},
    {"code": "WH2", "name": "Warehouse 2", "kind": "warehouse",
     "city": "Chicago", "country": "USA",
     "latitude": Decimal("41.878100"), "longitude": Decimal("-87.629800")},
    {"code": "DC1", "name": "Distribution Center 1", "kind": "distribution_center",
     "city": "Dallas", "country": "USA",
     "latitude": Decimal("32.776700"), "longitude": Decimal("-96.797000")},
    {"code": "RT1", "name": "Retail Outlet 1", "kind": "retail",
     "city": "New York", "country": "USA",
     "latitude": Decimal("40.712800"), "longitude": Decimal("-74.006000")},
]

# Tables this script owns, in delete order (children first).
OWNED_TABLES = [
    WarehouseSensor, Inventory, Warehouse,
    Shipment, OrderItem, Order, Customer, Product, Category, Department,
]


class Stats:
    def __init__(self) -> None:
        self.counts: dict[str, int] = defaultdict(int)
        self.notes: list[str] = []

    def note(self, msg: str) -> None:
        if msg not in self.notes:
            self.notes.append(msg)


def get(row, cols: dict[str, str], logical: str):
    """Fetch a logical field from a row, or None if that column isn't in the file."""
    actual = cols.get(logical)
    return None if actual is None else row.get(actual)


# -- dimension tables ------------------------------------------------------


def load_dimensions(df: pd.DataFrame, cols: dict[str, str], session, stats: Stats) -> None:
    """departments, categories, products, customers -- deduplicated from the flat file."""

    # Departments
    if "department_id" in cols and "department_name" in cols:
        dep = (
            df[[cols["department_id"], cols["department_name"]]]
            .dropna(subset=[cols["department_id"]])
            .drop_duplicates(subset=[cols["department_id"]], keep="first")
        )
        rows = []
        seen_names: dict[str, int] = {}
        for _, r in dep.iterrows():
            did = as_int(r[cols["department_id"]])
            name = as_str(r[cols["department_name"]], 128) or f"Department {did}"
            if did is None:
                continue
            # departments.name is UNIQUE; disambiguate rather than drop the row.
            if name in seen_names:
                stats.note(f"Department name '{name}' reused by ids "
                           f"{seen_names[name]} and {did}; suffixed the second.")
                name = f"{name} ({did})"
            seen_names.setdefault(name, did)
            rows.append({"id": did, "name": name})
        session.bulk_insert_mappings(Department, rows)
        stats.counts["departments"] = len(rows)

    # Categories
    if "category_id" in cols and "category_name" in cols:
        sub = [cols["category_id"], cols["category_name"]]
        if "department_id" in cols:
            sub.append(cols["department_id"])
        cat = (
            df[sub]
            .dropna(subset=[cols["category_id"]])
            .drop_duplicates(subset=[cols["category_id"]], keep="first")
        )
        rows = []
        for _, r in cat.iterrows():
            cid = as_int(r[cols["category_id"]])
            if cid is None:
                continue
            rows.append({
                "id": cid,
                "name": as_str(r[cols["category_name"]], 128) or f"Category {cid}",
                "department_id": as_int(r[cols["department_id"]]) if "department_id" in cols else None,
            })
        session.bulk_insert_mappings(Category, rows)
        stats.counts["categories"] = len(rows)

    # Products
    if "product_id" in cols:
        sub = [c for c in (
            "product_id", "product_name", "product_description", "product_price",
            "product_status", "product_image", "product_category_id", "department_id",
        ) if c in cols]
        prod = (
            df[[cols[c] for c in sub]]
            .dropna(subset=[cols["product_id"]])
            .drop_duplicates(subset=[cols["product_id"]], keep="first")
        )
        rows = []
        for _, r in prod.iterrows():
            pid = as_int(r[cols["product_id"]])
            if pid is None:
                continue
            rows.append({
                "id": pid,
                "name": as_str(get(r, cols, "product_name"), 255) or f"Product {pid}",
                "description": as_str(get(r, cols, "product_description")),
                "price": as_decimal(get(r, cols, "product_price")),
                "status": as_int(get(r, cols, "product_status")),
                "image_url": as_str(get(r, cols, "product_image"), 512),
                "category_id": as_int(get(r, cols, "product_category_id")),
                "department_id": as_int(get(r, cols, "department_id")),
            })
        session.bulk_insert_mappings(Product, rows)
        stats.counts["products"] = len(rows)

    # Customers
    if "customer_id" in cols:
        sub = [c for c in (
            "customer_id", "customer_fname", "customer_lname", "customer_segment",
            "customer_city", "customer_state", "customer_country", "customer_street",
            "customer_zipcode",
        ) if c in cols]
        cust = (
            df[[cols[c] for c in sub]]
            .dropna(subset=[cols["customer_id"]])
            .drop_duplicates(subset=[cols["customer_id"]], keep="first")
        )
        rows = []
        for _, r in cust.iterrows():
            cid = as_int(r[cols["customer_id"]])
            if cid is None:
                continue
            rows.append({
                "id": cid,
                "first_name": as_str(get(r, cols, "customer_fname"), 128),
                "last_name": as_str(get(r, cols, "customer_lname"), 128),
                "segment": as_str(get(r, cols, "customer_segment"), 32),
                "city": as_str(get(r, cols, "customer_city"), 128),
                "state": as_str(get(r, cols, "customer_state"), 64),
                "country": as_str(get(r, cols, "customer_country"), 64),
                "street": as_str(get(r, cols, "customer_street"), 255),
                "zipcode": as_zip(get(r, cols, "customer_zipcode")),
            })
        session.bulk_insert_mappings(Customer, rows)
        stats.counts["customers"] = len(rows)

    session.commit()


# -- fact tables -----------------------------------------------------------


def load_facts(df: pd.DataFrame, cols: dict[str, str], session, stats: Stats) -> None:
    """orders (one per Order Id), order_items (one per row), shipments (one per order)."""
    known_customers = {c for (c,) in session.execute(select(Customer.id))}
    known_products = {p for (p,) in session.execute(select(Product.id))}

    # Orders: first row per Order Id wins; the order-level fields repeat across items.
    order_rows: list[dict] = []
    shipment_rows: list[dict] = []
    seen_orders: set[int] = set()

    for _, r in df.iterrows():
        oid = as_int(get(r, cols, "order_id"))
        if oid is None or oid in seen_orders:
            continue
        seen_orders.add(oid)

        cust = as_int(get(r, cols, "order_customer_id"))
        if cust is not None and cust not in known_customers:
            stats.note(f"Order {oid} references unknown customer {cust}; set to NULL.")
            cust = None

        order_rows.append({
            "id": oid,
            "customer_id": cust,
            "ordered_at": as_datetime(get(r, cols, "order_date")),
            "status": as_str(get(r, cols, "order_status"), 32),
            "order_type": as_str(get(r, cols, "order_type"), 32),
            "market": as_str(get(r, cols, "market"), 32),
            "region": as_str(get(r, cols, "order_region"), 64),
            "country": as_str(get(r, cols, "order_country"), 64),
            "state": as_str(get(r, cols, "order_state"), 64),
            "city": as_str(get(r, cols, "order_city"), 128),
            "zipcode": as_zip(get(r, cols, "order_zipcode")),
            "latitude": as_decimal(get(r, cols, "latitude"), "0.000001"),
            "longitude": as_decimal(get(r, cols, "longitude"), "0.000001"),
            "sales": as_decimal(get(r, cols, "order_sales")),
            "profit": as_decimal(get(r, cols, "order_profit")),
            "benefit_per_order": as_decimal(get(r, cols, "benefit_per_order")),
        })

        actual = as_int(get(r, cols, "days_actual"))
        scheduled = as_int(get(r, cols, "days_scheduled"))
        shipment_rows.append({
            "order_id": oid,
            "shipping_mode": as_str(get(r, cols, "shipping_mode"), 32),
            "delivery_status": as_str(get(r, cols, "delivery_status"), 32),
            "shipped_at": as_datetime(get(r, cols, "shipping_date")),
            "days_actual": actual,
            "days_scheduled": scheduled,
            "days_delta": (actual - scheduled) if (actual is not None and scheduled is not None) else None,
            "late_delivery_risk": as_bool(get(r, cols, "late_delivery_risk")),
        })

    _bulk(session, Order, order_rows)
    stats.counts["orders"] = len(order_rows)

    # Order items: one per CSV row.
    item_rows: list[dict] = []
    seen_items: set[int] = set()
    synthetic_item_id = 0
    dropped_orphans = 0

    for _, r in df.iterrows():
        oid = as_int(get(r, cols, "order_id"))
        if oid is None or oid not in seen_orders:
            dropped_orphans += 1
            continue

        iid = as_int(get(r, cols, "order_item_id"))
        if iid is None or iid in seen_items:
            # Keep the row rather than lose revenue; mint an id above the real range.
            synthetic_item_id += 1
            iid = 900_000_000 + synthetic_item_id
            stats.note("Some rows had a missing/duplicate 'Order Item Id'; "
                       "synthetic ids from 900000001 were assigned.")
        seen_items.add(iid)

        pid = as_int(get(r, cols, "item_product_id"))
        if pid is not None and pid not in known_products:
            stats.note(f"Order item {iid} references unknown product {pid}; set to NULL.")
            pid = None

        item_rows.append({
            "id": iid,
            "order_id": oid,
            "product_id": pid,
            "quantity": as_int(get(r, cols, "item_quantity")),
            "unit_price": as_decimal(get(r, cols, "item_unit_price")),
            "discount": as_decimal(get(r, cols, "item_discount")),
            "discount_rate": as_decimal(get(r, cols, "item_discount_rate"), "0.000001"),
            "sales": as_decimal(get(r, cols, "item_sales")),
            "total": as_decimal(get(r, cols, "item_total")),
            "profit_ratio": as_decimal(get(r, cols, "item_profit_ratio"), "0.000001"),
        })

    if dropped_orphans:
        stats.note(f"{dropped_orphans:,} rows had no usable Order Id and were skipped.")

    _bulk(session, OrderItem, item_rows)
    stats.counts["order_items"] = len(item_rows)

    _bulk(session, Shipment, shipment_rows)
    stats.counts["shipments"] = len(shipment_rows)


def _bulk(session, model, rows: list[dict]) -> None:
    for i in range(0, len(rows), CHUNK):
        session.bulk_insert_mappings(model, rows[i:i + CHUNK])
        session.commit()


# -- warehouses + inventory ------------------------------------------------


def seed_warehouses(session, stats: Stats) -> list[Warehouse]:
    warehouses = [Warehouse(**w) for w in WAREHOUSE_SEED]
    session.add_all(warehouses)
    session.commit()
    stats.counts["warehouses"] = len(warehouses)
    return warehouses


def seed_inventory(session, warehouses: list[Warehouse], stats: Stats) -> None:
    """
    DataCo has no stock levels, so inventory is DERIVED from observed demand:
    for each product, total units ever ordered -> spread across the stocking nodes.

      on_hand      = round(total_qty / n_nodes * 0.30)   (~3.6 months of cover)
      reorder_pt   = round(on_hand * 0.25)
      reserved     = 0

    This is a defensible starting position, not real data. Replace it the moment
    you have actual stock figures.
    """
    stocking = [w for w in warehouses if w.kind in ("warehouse", "distribution_center")]
    if not stocking:
        return

    demand = session.execute(
        select(OrderItem.product_id, func.sum(OrderItem.quantity))
        .where(OrderItem.product_id.is_not(None))
        .group_by(OrderItem.product_id)
    ).all()

    now = datetime.now(timezone.utc)
    rows = []
    for product_id, total_qty in demand:
        per_node = int(round((total_qty or 0) / len(stocking) * 0.30))
        for w in stocking:
            rows.append({
                "warehouse_id": w.id,
                "product_id": product_id,
                "quantity_on_hand": max(per_node, 0),
                "quantity_reserved": 0,
                "reorder_point": max(int(round(per_node * 0.25)), 0),
                # Set explicitly: bulk_insert_mappings bypasses ORM defaults.
                "updated_at": now,
            })

    _bulk(session, Inventory, rows)
    stats.counts["inventory"] = len(rows)
    stats.note("inventory is derived from order demand -- DataCo carries no stock levels.")


# -- orchestration ---------------------------------------------------------


def preflight(session, truncate: bool) -> None:
    populated = []
    for model in OWNED_TABLES:
        n = session.scalar(select(func.count()).select_from(model))
        if n:
            populated.append((model.__tablename__, n))

    if not populated:
        return
    if not truncate:
        detail = ", ".join(f"{t} ({n:,})" for t, n in populated)
        raise DatasetError(
            f"These tables already hold data: {detail}. "
            f"Re-run with --truncate to replace it."
        )

    for model in OWNED_TABLES:
        session.execute(model.__table__.delete())
    session.commit()
    print(f"Truncated: {', '.join(t for t, _ in populated)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Load the DataCo CSV into Postgres.")
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None, help="read only first N rows")
    parser.add_argument("--truncate", action="store_true", help="clear tables before loading")
    parser.add_argument("--skip-inventory", action="store_true")
    args = parser.parse_args()

    path = args.csv or find_dataset()
    print(f"Reading {path} ...")
    df = read_dataset(path, nrows=args.limit)
    print(f"  {len(df):,} rows x {len(df.columns)} columns")

    cols = resolve_columns(df)
    required = ("order_id", "order_item_id", "product_id")
    missing = [c for c in required if c not in cols]
    if missing:
        raise DatasetError(
            f"Required columns not found in the CSV: {missing}.\n"
            f"Headers present: {sorted(df.columns)}\n"
            f"Run scripts/inspect_dataset.py and fix COLUMN_MAP in backend/load_helpers.py."
        )
    unmapped = [c for c in df.columns if c not in set(cols.values())]
    if unmapped:
        print(f"  {len(unmapped)} column(s) present but not loaded: {unmapped}")

    stats = Stats()
    with SessionLocal() as session:
        preflight(session, args.truncate)

        print("Loading dimensions (departments, categories, products, customers) ...")
        load_dimensions(df, cols, session, stats)

        print("Loading facts (orders, order_items, shipments) ...")
        load_facts(df, cols, session, stats)

        print("Seeding warehouses ...")
        warehouses = seed_warehouses(session, stats)

        if args.skip_inventory:
            print("Skipping inventory seed.")
        else:
            print("Deriving inventory from demand ...")
            seed_inventory(session, warehouses, stats)

        # Report actual post-commit row counts, not what we think we inserted.
        print("\n--- row counts ---")
        for model in reversed(OWNED_TABLES):
            n = session.scalar(select(func.count()).select_from(model))
            print(f"  {model.__tablename__:<20} {n:>10,}")

    if stats.notes:
        print("\n--- judgment calls / data issues ---")
        for note in stats.notes:
            print(f"  * {note}")

    engine.dispose()
    print("\nDone.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DatasetError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
