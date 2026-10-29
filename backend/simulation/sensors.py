"""
Synthetic warehouse telemetry.

Sensor data legitimately does not come from the DataCo dataset, so this keeps the
sine + random-noise approach from the retired generate_data.py, including its
baselines. The differences from the old script:

  * phase is driven by wall-clock time, not a row index, so the series is
    continuous across server restarts;
  * there are no hard-coded crisis windows -- breaches now come from disruptions
    injected at random by the worker.

Baselines and the bands they oscillate in are chosen to sit clear of the
frontend's alert thresholds (humidity 68%, temperature 28C, DC capacity 88%)
during normal operation, so a breach means something actually happened.
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass

# Seconds per full sine cycle. The old script advanced 0.1 "t" per tick and used
# frequencies of 0.3-0.7, i.e. a period of ~90-210 ticks; at a 5s cycle that maps
# to roughly 8-18 minutes of wall clock.
PERIOD_SECONDS = 900.0


@dataclass(frozen=True)
class SensorSpec:
    reading_type: str
    unit: str
    base: float          # centre of the oscillation
    amplitude: float     # sine swing either side of base
    noise: float         # uniform jitter either side
    phase: float         # offset so metrics don't move in lockstep
    freq: float          # multiplier on the base period
    low: float           # hard clamp
    high: float
    decimals: int = 1


# Which metrics each kind of node reports.
SPECS_BY_KIND: dict[str, tuple[SensorSpec, ...]] = {
    "warehouse": (
        SensorSpec("humidity", "%", 52.0, 10.0, 2.0, 0.0, 0.7, 30.0, 85.0),
        SensorSpec("temperature", "C", 22.0, 5.0, 0.5, 1.1, 0.3, 15.0, 35.0),
    ),
    "distribution_center": (
        SensorSpec("humidity", "%", 48.0, 8.0, 2.0, 1.0, 0.5, 30.0, 75.0),
        SensorSpec("temperature", "C", 18.0, 3.0, 1.0, 0.4, 0.3, 10.0, 30.0),
        SensorSpec("capacity_pct", "%", 55.0, 20.0, 3.0, 2.0, 0.4, 20.0, 98.0),
        SensorSpec("throughput", "units/h", 350.0, 70.0, 30.0, 0.7, 0.6, 120.0, 600.0, 0),
    ),
    "retail": (
        SensorSpec("humidity", "%", 45.0, 6.0, 2.0, 0.3, 0.5, 30.0, 70.0),
        SensorSpec("temperature", "C", 20.0, 3.0, 0.6, 1.7, 0.35, 12.0, 30.0),
    ),
}

# Robot arms live on warehouse nodes. Reported as 1/0 in the numeric value column.
ROBOT_ARMS_PER_WAREHOUSE = 2
ARM_FAILURE_CHANCE = 0.01  # matches the old script's 1% baseline

# Where a spike pushes each metric. Chosen to clear the alert thresholds, mirroring
# the targets the old script's crisis scenarios ramped toward.
SPIKE_TARGETS: dict[str, tuple[float, float]] = {
    "humidity": (70.0, 80.0),      # threshold 68
    "temperature": (29.0, 33.0),   # threshold 28
    "capacity_pct": (90.0, 97.0),  # threshold 88
    "throughput": (40.0, 110.0),   # a throughput *collapse* is the anomaly
}

SPIKEABLE = tuple(SPIKE_TARGETS)


def _wave(spec: SensorSpec, now: float) -> float:
    t = (now / PERIOD_SECONDS) * 2 * math.pi * spec.freq
    value = spec.base + spec.amplitude * math.sin(t + spec.phase)
    value += random.uniform(-spec.noise, spec.noise)
    value = max(spec.low, min(spec.high, value))
    return round(value, spec.decimals) if spec.decimals else float(round(value))


def readings_for(kind: str | None, now: float | None = None) -> list[tuple[str, float, str]]:
    """
    Produce this moment's readings for a node of the given kind.
    Returns (reading_type, value, unit) triples.
    """
    now = time.time() if now is None else now
    out: list[tuple[str, float, str]] = []

    for spec in SPECS_BY_KIND.get(kind or "warehouse", SPECS_BY_KIND["warehouse"]):
        out.append((spec.reading_type, _wave(spec, now), spec.unit))

    if kind == "warehouse":
        for arm in range(1, ROBOT_ARMS_PER_WAREHOUSE + 1):
            active = random.random() > ARM_FAILURE_CHANCE
            out.append((f"robot_arm_{arm}_active", 1.0 if active else 0.0, "bool"))

    return out


def spike_value(reading_type: str) -> float | None:
    """The value a spiked metric should report, or None if it isn't spikeable."""
    bounds = SPIKE_TARGETS.get(reading_type)
    if bounds is None:
        return None
    return round(random.uniform(*bounds), 1)


def normal_range(reading_type: str, kind: str) -> tuple[float, float] | None:
    """The band this metric normally occupies, for event context."""
    for spec in SPECS_BY_KIND.get(kind, ()):  # noqa: B007
        if spec.reading_type == reading_type:
            return (
                round(spec.base - spec.amplitude - spec.noise, 1),
                round(spec.base + spec.amplitude + spec.noise, 1),
            )
    return None
