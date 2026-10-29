"""
Tunables for the live simulation worker. Everything here is overridable by env
var so behaviour can be changed without a code edit.
"""
from __future__ import annotations

import os


def _f(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _i(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


# -- cycle ------------------------------------------------------------------

CYCLE_SECONDS = _f("SIM_CYCLE_SECONDS", 5.0)
ENABLED = os.getenv("SIM_ENABLED", "1").lower() not in ("0", "false", "no")

# Back off to this interval after repeated failures, instead of hammering a
# database that is down.
FAILURE_BACKOFF_SECONDS = _f("SIM_FAILURE_BACKOFF_SECONDS", 30.0)

# -- shipments --------------------------------------------------------------

# How many shipments to keep in flight at once, and how many to advance per cycle.
IN_TRANSIT_TARGET = _i("SIM_IN_TRANSIT_TARGET", 40)
SHIPMENTS_PER_CYCLE = _i("SIM_SHIPMENTS_PER_CYCLE", 12)

# A shipment gains this much progress per cycle (percentage points), so a normal
# delivery takes roughly 100/8 ~= 13 cycles ~= 65s at a 5s cycle. Fast enough to
# watch, slow enough that the board isn't churning.
PROGRESS_MIN = _i("SIM_PROGRESS_MIN", 5)
PROGRESS_MAX = _i("SIM_PROGRESS_MAX", 11)

# -- inventory --------------------------------------------------------------

# Products drawn down per cycle.
INVENTORY_DRAWS_PER_CYCLE = _i("SIM_INVENTORY_DRAWS_PER_CYCLE", 8)

# A row at or below its reorder point is replenished to reorder_point * this.
# seed_inventory set reorder_point to 25% of opening stock, so 4.0 restores
# roughly the original level.
RESTOCK_MULTIPLIER = _f("SIM_RESTOCK_MULTIPLIER", 4.0)

# -- disruptions ------------------------------------------------------------

# Probability of injecting a disruption on any given cycle. At the 5s default
# that is one roughly every 100s of simulated operation.
DISRUPTION_RATE = _f("SIM_DISRUPTION_RATE", 0.05)

# Minimum gap between injected disruptions, so a run of lucky rolls cannot
# produce a burst that looks like a system-wide failure.
DISRUPTION_COOLDOWN_SECONDS = _f("SIM_DISRUPTION_COOLDOWN_SECONDS", 30.0)

# Relative weights for which disruption gets injected.
DISRUPTION_WEIGHTS = {
    "SHIPMENT_DELAY": _f("SIM_WEIGHT_SHIPMENT_DELAY", 0.45),
    "SENSOR_SPIKE": _f("SIM_WEIGHT_SENSOR_SPIKE", 0.35),
    "STOCK_OUT": _f("SIM_WEIGHT_STOCK_OUT", 0.20),
}

# Added to a delayed shipment's ETA.
DELAY_MINUTES_MIN = _i("SIM_DELAY_MINUTES_MIN", 90)
DELAY_MINUTES_MAX = _i("SIM_DELAY_MINUTES_MAX", 720)

# How many cycles a sensor spike persists once injected. A one-cycle blip is
# easy for a consumer to miss between polls.
SPIKE_CYCLES_MIN = _i("SIM_SPIKE_CYCLES_MIN", 3)
SPIKE_CYCLES_MAX = _i("SIM_SPIKE_CYCLES_MAX", 8)

# -- retention --------------------------------------------------------------

# warehouse_sensors grows ~10 rows/cycle forever; trim it so a long-running dev
# server doesn't fill the disk. 0 disables trimming.
SENSOR_RETENTION_ROWS = _i("SIM_SENSOR_RETENTION_ROWS", 50_000)
SENSOR_TRIM_EVERY_CYCLES = _i("SIM_SENSOR_TRIM_EVERY_CYCLES", 200)
