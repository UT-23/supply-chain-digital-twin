"""
Inspect the raw DataCo CSV before loading it.

Run this first. It prints every column with its dtype, null count, cardinality and a
few sample values, then checks the specific assumptions load_dataset.py relies on
(key uniqueness, one-shipment-per-order, date parseability).

    python scripts/inspect_dataset.py
    python scripts/inspect_dataset.py --csv data/raw/SomeOtherName.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from load_helpers import find_dataset, read_dataset  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--rows", type=int, default=None, help="only read first N rows")
    args = parser.parse_args()

    path = args.csv or find_dataset()
    print(f"Reading {path}")
    df = read_dataset(path, nrows=args.rows)
    print(f"\nShape: {df.shape[0]:,} rows x {df.shape[1]} columns\n")

    print(f"{'column':<34} {'dtype':<10} {'nulls':>9} {'uniq':>9}  sample")
    print("-" * 110)
    for col in df.columns:
        s = df[col]
        sample = ", ".join(str(v)[:22] for v in s.dropna().unique()[:3])
        print(f"{col:<34} {str(s.dtype):<10} {s.isna().sum():>9,} {s.nunique():>9,}  {sample[:40]}")

    print("\n--- assumption checks ---")
    checks: list[tuple[str, bool, str]] = []

    def has(col: str) -> bool:
        return col in df.columns

    if has("Order Item Id"):
        dup = int(df["Order Item Id"].duplicated().sum())
        checks.append(("Order Item Id is unique (order_items PK)", dup == 0, f"{dup:,} duplicates"))

    if has("Order Id") and has("Order Customer Id"):
        n = int(df.groupby("Order Id")["Order Customer Id"].nunique().gt(1).sum())
        checks.append(("One customer per Order Id", n == 0, f"{n:,} orders violate"))

    # Shipment facts must be constant within an order for the 1:1 shipments model.
    ship_cols = [
        c
        for c in ("Shipping Mode", "Delivery Status", "shipping date (DateOrders)",
                  "Days for shipping (real)", "Days for shipment (scheduled)")
        if has(c)
    ]
    if has("Order Id") and ship_cols:
        varying = df.groupby("Order Id")[ship_cols].nunique().gt(1).sum()
        for col, n in varying.items():
            checks.append((f"'{col}' constant within order", n == 0, f"{n:,} orders vary"))

    if has("Product Card Id") and has("Product Name"):
        n = int(df.groupby("Product Card Id")["Product Name"].nunique().gt(1).sum())
        checks.append(("Product Card Id -> one name", n == 0, f"{n:,} ids have >1 name"))

    if has("Category Id") and has("Department Id"):
        n = int(df.groupby("Category Id")["Department Id"].nunique().gt(1).sum())
        checks.append(("Category belongs to one department", n == 0, f"{n:,} categories span depts"))

    for col in ("order date (DateOrders)", "shipping date (DateOrders)"):
        if has(col):
            parsed = pd.to_datetime(df[col], errors="coerce")
            bad = int(parsed.isna().sum() - df[col].isna().sum())
            checks.append((f"'{col}' parses as datetime", bad == 0, f"{bad:,} unparseable"))
            if parsed.notna().any():
                print(f"    {col}: {parsed.min()} .. {parsed.max()}")

    for label, ok, detail in checks:
        print(f"  [{'OK ' if ok else 'WARN'}] {label}" + ("" if ok else f"  ({detail})"))

    missing = [c for c in ("Order Id", "Order Item Id", "Product Card Id") if not has(c)]
    if missing:
        print(f"\n!! Expected columns absent: {missing}")
        print("   Column names differ from the standard DataCo release -- update COLUMN_MAP")
        print("   in backend/load_helpers.py before running the loader.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
