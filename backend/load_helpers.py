"""
Shared helpers for locating, reading and normalizing the raw DataCo CSV.

Kept separate from scripts/load_dataset.py so the inspector and the loader agree on
exactly how the file is found, decoded and column-mapped.
"""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

# Names the DataCo file commonly ships under, tried in order.
CANDIDATE_NAMES = (
    "DataCoSupplyChainDataset.csv",
    "DataCo Smart Supply Chain.csv",
    "dataco_supply_chain.csv",
)

# Shipped alongside the dataset in the Kaggle download, and never the file we
# want: a column glossary and an unrelated clickstream log.
IGNORE_NAMES = frozenset({
    "DescriptionDataCoSupplyChain.csv",
    "tokenized_access_logs.csv",
})

# The file is not UTF-8 -- it carries Latin-1 bytes in customer/product names.
ENCODINGS = ("latin-1", "ISO-8859-1", "cp1252", "utf-8")

# Logical name -> the column header(s) it may appear under in the raw CSV.
# Update the right-hand lists if inspect_dataset.py reports different headers.
COLUMN_MAP: dict[str, tuple[str, ...]] = {
    # order
    "order_id": ("Order Id",),
    "order_customer_id": ("Order Customer Id", "Customer Id"),
    "order_date": ("order date (DateOrders)", "order_date_dateorders"),
    "order_status": ("Order Status",),
    "order_type": ("Type",),
    "market": ("Market",),
    "order_region": ("Order Region",),
    "order_country": ("Order Country",),
    "order_state": ("Order State",),
    "order_city": ("Order City",),
    "order_zipcode": ("Order Zipcode",),
    "latitude": ("Latitude",),
    "longitude": ("Longitude",),
    "order_sales": ("Sales",),
    "order_profit": ("Order Profit Per Order",),
    "benefit_per_order": ("Benefit per order",),
    # order item
    "order_item_id": ("Order Item Id",),
    "item_product_id": ("Order Item Cardprod Id", "Product Card Id"),
    "item_quantity": ("Order Item Quantity",),
    "item_unit_price": ("Order Item Product Price",),
    "item_discount": ("Order Item Discount",),
    "item_discount_rate": ("Order Item Discount Rate",),
    "item_total": ("Order Item Total",),
    "item_profit_ratio": ("Order Item Profit Ratio",),
    "item_sales": ("Sales per customer", "Sales"),
    # shipment
    "shipping_date": ("shipping date (DateOrders)", "shipping_date_dateorders"),
    "shipping_mode": ("Shipping Mode",),
    "delivery_status": ("Delivery Status",),
    "days_actual": ("Days for shipping (real)",),
    "days_scheduled": ("Days for shipment (scheduled)",),
    "late_delivery_risk": ("Late_delivery_risk",),
    # product
    "product_id": ("Product Card Id",),
    "product_name": ("Product Name",),
    "product_description": ("Product Description",),
    "product_price": ("Product Price",),
    "product_status": ("Product Status",),
    "product_image": ("Product Image",),
    "product_category_id": ("Product Category Id", "Category Id"),
    # category / department
    "category_id": ("Category Id",),
    "category_name": ("Category Name",),
    "department_id": ("Department Id",),
    "department_name": ("Department Name",),
    # customer
    "customer_id": ("Customer Id",),
    "customer_fname": ("Customer Fname",),
    "customer_lname": ("Customer Lname",),
    "customer_segment": ("Customer Segment",),
    "customer_city": ("Customer City",),
    "customer_state": ("Customer State",),
    "customer_country": ("Customer Country",),
    "customer_street": ("Customer Street",),
    "customer_zipcode": ("Customer Zipcode",),
}

# Deliberately never loaded: the dataset ships plaintext credentials and emails.
PII_DROP = ("Customer Email", "Customer Password")


class DatasetError(RuntimeError):
    pass


def find_dataset(raw_dir: Path | None = None) -> Path:
    """Locate the DataCo CSV in data/raw/, by known name or by falling back to any CSV."""
    raw_dir = raw_dir or RAW_DIR
    if not raw_dir.exists():
        raise DatasetError(f"{raw_dir} does not exist. Create it and drop the DataCo CSV in.")

    # Kaggle's download extracts into a subdirectory (typically "archive/"), so
    # look at the top level first and then recursively. The recursive pass also
    # has to be name-driven: the DataCo download ships two decoys alongside the
    # dataset -- DescriptionDataCoSupplyChain.csv (a column glossary) and
    # tokenized_access_logs.csv (a ~95MB clickstream log, a different dataset).
    for name in CANDIDATE_NAMES:
        candidate = raw_dir / name
        if candidate.exists():
            return candidate

    for name in CANDIDATE_NAMES:
        nested = sorted(raw_dir.rglob(name))
        if nested:
            return nested[0]

    csvs = sorted(p for p in raw_dir.rglob("*.csv") if p.name not in IGNORE_NAMES)
    if len(csvs) == 1:
        return csvs[0]
    if not csvs:
        raise DatasetError(
            f"No CSV found in {raw_dir}. Download the DataCo Smart Supply Chain dataset "
            f"and place DataCoSupplyChainDataset.csv there."
        )
    raise DatasetError(f"Multiple CSVs in {raw_dir}: {[p.name for p in csvs]}. Pass --csv.")


def read_dataset(path: Path, nrows: int | None = None) -> pd.DataFrame:
    """Read the CSV, trying each known encoding. Headers are whitespace-stripped."""
    last: Exception | None = None
    for enc in ENCODINGS:
        try:
            df = pd.read_csv(path, encoding=enc, nrows=nrows, low_memory=False)
            break
        except UnicodeDecodeError as exc:
            last = exc
    else:
        raise DatasetError(f"Could not decode {path} with any of {ENCODINGS}: {last}")

    df.columns = [str(c).strip() for c in df.columns]
    return df.drop(columns=[c for c in PII_DROP if c in df.columns], errors="ignore")


def resolve_columns(df: pd.DataFrame) -> dict[str, str]:
    """Map logical names to the actual headers present. Absent ones are omitted."""
    resolved: dict[str, str] = {}
    for logical, candidates in COLUMN_MAP.items():
        for actual in candidates:
            if actual in df.columns:
                resolved[logical] = actual
                break
    return resolved


# -- value coercion --------------------------------------------------------


def as_int(value) -> int | None:
    """Coerce to int, tolerating '3.0', ' 3 ', NaN, '' and 'NA'."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, str):
        value = value.strip()
        if value in ("", "NA", "N/A", "nan", "NaN", "-"):
            return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(f) else int(round(f))


def as_decimal(value, places: str = "0.01") -> Decimal | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, str):
        value = value.strip().replace(",", "").replace("$", "")
        if value in ("", "NA", "N/A", "nan", "NaN", "-"):
            return None
    try:
        d = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None
    if not d.is_finite():
        return None
    try:
        return d.quantize(Decimal(places))
    except InvalidOperation:
        return None


def as_str(value, max_len: int | None = None) -> str | None:
    """Trim, collapse empties to NULL, and truncate to the column width."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    s = str(value).strip()
    if s in ("", "NA", "N/A", "nan", "NaN"):
        return None
    return s[:max_len] if max_len else s


def as_zip(value) -> str | None:
    """
    Zip codes arrive as floats ('95758.0') because the column has nulls. Render the
    integer form and keep them as text -- they are identifiers, not numbers.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    s = str(value).strip()
    if s in ("", "NA", "nan", "NaN"):
        return None
    if s.replace(".", "", 1).isdigit() and s.endswith(".0"):
        s = s[:-2]
    return s[:16]


def as_bool(value) -> bool | None:
    """Late_delivery_risk is 0/1; be liberal about other truthy spellings."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in ("1", "1.0", "true", "yes", "y"):
        return True
    if s in ("0", "0.0", "false", "no", "n"):
        return False
    return None


def as_datetime(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    ts = pd.to_datetime(value, errors="coerce")
    return None if pd.isna(ts) else ts.to_pydatetime()
