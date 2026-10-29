"""
Test whether the agent is actually exercising judgement, or is a threshold system
with an API call in the middle.

    python scripts/agent_judgment_probe.py
    python scripts/agent_judgment_probe.py --repeats 3

This does NOT touch the live database. It feeds the model hand-built state
snapshots and records what it decides, with every tool call intercepted rather
than executed. Nothing is mutated.

## The experiment

The probe is built around matched pairs. Each pair holds the *numeric reading*
constant and varies only the surrounding context. A threshold system must
respond identically within a pair, because it only sees the number. A system
exercising judgement may respond differently.

  pair A -- humidity 71.2% at a site holding spoilable stock
         vs humidity 71.2% at a site that is empty
  pair B -- a shipment 4h late at 12% progress (recoverable)
         vs a shipment 4h late at 96% progress (nearly delivered)
  pair C -- a single humidity sample at 69.1%, flat trend
         vs six consecutive rising samples ending at 69.1%
  pair D -- a stock-out of a product with no open demand
         vs a stock-out of a product with in-flight shipments depending on it

Plus two controls:
  clear-normal   -- everything comfortably within band (should produce no action)
  clear-critical -- unambiguous multi-signal emergency (should produce action)

## Reading the result

  * Controls wrong  -> the agent is not working; fix that before judging nuance.
  * Pairs identical -> it is behaving like a threshold system. The numbers drive
    it and the context does not. That is a real and reportable finding.
  * Pairs differ, and the reasoning cites the contextual difference -> it is
    doing something a threshold cannot.

Divergence alone is not proof of judgement -- an LLM is stochastic and will vary
between identical calls. That is what --repeats is for: run each scenario
several times and compare within-scenario variation against across-pair
variation. If they are the same magnitude, the differences are noise.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

# Windows consoles default to cp1252, which cannot encode much of what a model
# writes (dashes, arrows, degree signs). Force UTF-8 so printing results never
# crashes the run after the API calls have already been paid for.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

BAND_HUM_WH = {"low": 40.0, "high": 64.0}
BAND_TEMP_WH = {"low": 16.5, "high": 27.5}


def _site(code, kind, humidity, trend=None, temp=22.0, inventory=0):
    return {
        "warehouse_id": {"WH1": 1, "WH2": 2, "DC1": 3, "RT1": 4}[code],
        "code": code,
        "name": code,
        "kind": kind,
        "inventory_units_on_site": inventory,
        "readings": [
            {
                "metric": "humidity",
                "current": humidity,
                "unit": "%",
                "recent_values_oldest_first": trend or [humidity] * 6,
                "normal_band": BAND_HUM_WH,
                "last_recorded_at": "2026-09-13T16:00:00+00:00",
            },
            {
                "metric": "temperature",
                "current": temp,
                "unit": "C",
                "recent_values_oldest_first": [temp] * 6,
                "normal_band": BAND_TEMP_WH,
                "last_recorded_at": "2026-09-13T16:00:00+00:00",
            },
        ],
    }


def _base(sites, shipments=None, inventory=None, events=None):
    return {
        "observed_at": "2026-09-13T16:00:00+00:00",
        "sites": sites,
        "shipments": shipments or {"counts_by_status": {"IN_TRANSIT": 30}, "notable_active": []},
        "inventory": inventory or {"locations_out_of_stock": 0, "lowest_stock": []},
        "unhandled_events": events or [],
        "open_alerts": [],
        "recent_agent_actions": [],
    }


def _ship(sid, progress, delay_min, product_note=""):
    return {
        "shipment_id": sid,
        "order_id": 1000 + sid,
        "status": "DELAYED",
        "progress_pct": progress,
        "delay_minutes": delay_min,
        "eta": "2026-09-14T04:00:00+00:00",
        "hours_until_eta": 12.0,
        "shipping_mode": "Standard Class",
        "current_route": None,
        "already_rerouted": False,
        "note": product_note,
    }


def _event(eid, etype, severity, entity_type, entity_id, message, details=None):
    return {
        "event_id": eid,
        "occurred_at": "2026-09-13T15:59:00+00:00",
        "age_seconds": 60.0,
        "type": etype,
        "severity": severity,
        "entity_type": entity_type,
        "entity_id": entity_id,
        "message": message,
        "details": details or {},
    }


# -- the scenarios ---------------------------------------------------------

RISING = [61.0, 63.2, 65.0, 66.4, 68.0, 69.1]
FLAT = [68.9, 69.0, 68.8, 69.2, 69.0, 69.1]

SCENARIOS: dict[str, dict] = {
    # Controls
    "control_normal": _base({
        "WH1": _site("WH1", "warehouse", 51.2, inventory=4200),
        "WH2": _site("WH2", "warehouse", 47.8, inventory=3100),
    }),
    "control_critical": _base(
        {
            "WH1": _site("WH1", "warehouse", 79.4, trend=[70, 73, 75, 77, 78, 79.4],
                         temp=31.8, inventory=5200),
            "WH2": _site("WH2", "warehouse", 48.0, inventory=3000),
        },
        inventory={"locations_out_of_stock": 3, "lowest_stock": [
            {"product_id": 365, "product_name": "Perfect Fitness Rip Deck", "warehouse_id": 1,
             "warehouse_code": "WH1", "quantity_on_hand": 0, "reorder_point": 40,
             "already_flagged": False},
        ]},
        events=[
            _event(901, "SENSOR_SPIKE", "critical", "warehouse", 1,
                   "humidity anomaly at WH1 (~79.4)"),
            _event(902, "STOCK_OUT", "critical", "inventory", 11,
                   "Product 365 out of stock at WH1"),
        ],
    ),

    # Pair A -- same humidity, different consequence
    "A1_humid_full_warehouse": _base(
        {"WH1": _site("WH1", "warehouse", 71.2, inventory=5200)},
        events=[_event(910, "SENSOR_SPIKE", "critical", "warehouse", 1,
                       "humidity anomaly at WH1 (~71.2)")],
    ),
    "A2_humid_empty_warehouse": _base(
        {"WH1": _site("WH1", "warehouse", 71.2, inventory=0)},
        events=[_event(910, "SENSOR_SPIKE", "critical", "warehouse", 1,
                       "humidity anomaly at WH1 (~71.2)")],
    ),

    # Pair B -- same lateness, different recoverability
    "B1_late_early_journey": _base(
        {"WH1": _site("WH1", "warehouse", 52.0, inventory=3000)},
        shipments={"counts_by_status": {"DELAYED": 1, "IN_TRANSIT": 29},
                   "notable_active": [_ship(4101, 12, 240)]},
        events=[_event(920, "SHIPMENT_DELAY", "warning", "shipment", 4101,
                       "Shipment 4101 delayed by 240 min")],
    ),
    "B2_late_nearly_delivered": _base(
        {"WH1": _site("WH1", "warehouse", 52.0, inventory=3000)},
        shipments={"counts_by_status": {"DELAYED": 1, "IN_TRANSIT": 29},
                   "notable_active": [_ship(4102, 96, 240)]},
        events=[_event(921, "SHIPMENT_DELAY", "warning", "shipment", 4102,
                       "Shipment 4102 delayed by 240 min")],
    ),

    # Pair C -- same current value, different trajectory
    "C1_borderline_flat": _base(
        {"WH1": _site("WH1", "warehouse", 69.1, trend=FLAT, inventory=3000)},
    ),
    "C2_borderline_rising": _base(
        {"WH1": _site("WH1", "warehouse", 69.1, trend=RISING, inventory=3000)},
    ),

    # Pair D -- same stock-out, different demand
    "D1_stockout_no_demand": _base(
        {"WH1": _site("WH1", "warehouse", 52.0, inventory=3000)},
        inventory={"locations_out_of_stock": 1, "lowest_stock": [
            {"product_id": 502, "product_name": "Nike Dri-FIT Polo", "warehouse_id": 1,
             "warehouse_code": "WH1", "quantity_on_hand": 0, "reorder_point": 30,
             "already_flagged": False, "open_orders_for_product": 0},
        ]},
        events=[_event(930, "STOCK_OUT", "warning", "inventory", 21,
                       "Product 502 out of stock at WH1",
                       {"product_id": 502, "open_orders_for_product": 0})],
    ),
    "D2_stockout_with_demand": _base(
        {"WH1": _site("WH1", "warehouse", 52.0, inventory=3000)},
        shipments={"counts_by_status": {"IN_TRANSIT": 30},
                   "notable_active": [_ship(4200, 30, 0, "carries product 502")]},
        inventory={"locations_out_of_stock": 1, "lowest_stock": [
            {"product_id": 502, "product_name": "Nike Dri-FIT Polo", "warehouse_id": 1,
             "warehouse_code": "WH1", "quantity_on_hand": 0, "reorder_point": 30,
             "already_flagged": False, "open_orders_for_product": 14},
        ]},
        events=[_event(931, "STOCK_OUT", "warning", "inventory", 21,
                       "Product 502 out of stock at WH1",
                       {"product_id": 502, "open_orders_for_product": 14}),
                ],
    ),
}

PAIRS = [("A1_humid_full_warehouse", "A2_humid_empty_warehouse"),
         ("B1_late_early_journey", "B2_late_nearly_delivered"),
         ("C1_borderline_flat", "C2_borderline_rising"),
         ("D1_stockout_no_demand", "D2_stockout_with_demand")]


async def run_scenario(name: str, state: dict, repeats: int) -> list[dict]:
    """Run one scenario N times with tool execution stubbed out."""
    from agent import service as svc

    out = []
    for _ in range(repeats):
        agent = svc.AgentService()
        record = {
            "trigger": "probe", "input_state": state, "reasoning": None,
            "tool_calls": [], "tool_results": [], "model": svc.MODEL,
            "error": None, "stop_reason": None, "input_tokens": 0, "output_tokens": 0,
        }
        try:
            agent._client = svc._build_client()
            await agent._reason_and_act(state, "events", record)
        except Exception as exc:  # noqa: BLE001
            record["error"] = f"{type(exc).__name__}: {exc}"
        out.append(record)
    return out


def summarize(name: str, runs: list[dict]) -> dict:
    tools = [sorted({c["tool"] for c in r["tool_calls"]}) for r in runs]
    return {
        "scenario": name,
        "errors": [r["error"] for r in runs if r["error"]],
        "tool_sets": tools,
        "acted": [bool(t) for t in tools],
        "reasoning": [r["reasoning"] for r in runs],
    }


async def main_async(repeats: int) -> int:
    # NOTE: do NOT set AGENT_READ_ONLY here. It is read at import time by
    # service.py and restricts the tool list to the three read tools, so the
    # model is never offered flag/escalate/restock/reroute and cannot possibly
    # take an action -- which silently guarantees a "threshold-like" verdict.
    # Safety comes from stubbing _execute_tool below: no tool body ever runs,
    # so nothing can touch the database whatever the model asks for.

    from agent import service as svc

    try:
        svc._build_client()
    except svc.AgentUnavailable as exc:
        print(f"Cannot run the probe: {exc}\n")
        print("This probe is the evidence for whether the agent reasons or just")
        print("thresholds. It cannot be simulated -- it needs real model calls.")
        print("Set GROQ_API_KEY in backend/.env and re-run.")
        return 2

    # Intercept tool execution: record the call, return a plausible result,
    # never touch the database.
    def fake_execute(name: str, payload: dict) -> dict:
        return {"ok": True, "result": {"stubbed": True, "tool": name, "input": payload}}

    svc.AgentService._execute_tool = staticmethod(fake_execute)

    results: dict[str, dict] = {}
    for name, state in SCENARIOS.items():
        print(f"running {name} x{repeats} ...")
        runs = await run_scenario(name, state, repeats)
        results[name] = summarize(name, runs)

    # A failed API call is not a decision to take no action. Counting it as one
    # manufactures a "threshold-like" verdict out of an infrastructure problem,
    # so refuse to score a run that largely failed.
    failed = sum(len(r["errors"]) for r in results.values())
    total = sum(len(r["tool_sets"]) for r in results.values())
    if failed:
        print(f"\n!! {failed}/{total} run(s) FAILED:")
        for r in results.values():
            for e in r["errors"]:
                print(f"   {r['scenario']}: {str(e)[:160]}")
        if failed * 2 >= total:
            print("\nREFUSING TO RENDER A VERDICT: too many runs failed for these")
            print("results to mean anything. Fix the failures and re-run.")
            out = ROOT / "agent_probe_results.json"
            out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
            print(f"\nfull transcripts: {out}")
            return 3
        print("\nTreat the results below as provisional.")

    print("\n" + "=" * 72)
    print("CONTROLS")
    print("=" * 72)
    for name in ("control_normal", "control_critical"):
        r = results[name]
        expect = "no action" if name == "control_normal" else "action"
        print(f"\n{name}  (expected: {expect})")
        for i, (acted, tools) in enumerate(zip(r["acted"], r["tool_sets"])):
            print(f"  run {i}: {'ACTED' if acted else 'no action'}  {tools}")
        if r["reasoning"] and r["reasoning"][0]:
            print(f"  reasoning: {r['reasoning'][0][:300]}")

    print("\n" + "=" * 72)
    print("MATCHED PAIRS -- same numbers, different context")
    print("=" * 72)
    diverged = 0
    for left, right in PAIRS:
        lr, rr = results[left], results[right]
        lsets = {tuple(t) for t in lr["tool_sets"]}
        rsets = {tuple(t) for t in rr["tool_sets"]}
        same = lsets == rsets and len(lsets) == 1
        if not same:
            diverged += 1
        print(f"\n{left}")
        print(f"   tools: {lr['tool_sets']}")
        if lr["reasoning"] and lr["reasoning"][0]:
            print(f"   says:  {lr['reasoning'][0][:260]}")
        print(f"{right}")
        print(f"   tools: {rr['tool_sets']}")
        if rr["reasoning"] and rr["reasoning"][0]:
            print(f"   says:  {rr['reasoning'][0][:260]}")
        print(f"   => {'IDENTICAL response' if same else 'DIFFERENT response'}")

    print("\n" + "=" * 72)
    print(f"VERDICT: {diverged}/{len(PAIRS)} pairs produced a different response")
    print("=" * 72)
    if diverged == 0:
        print("The agent responded identically to every matched pair. On this")
        print("evidence it is behaving like a threshold system: the numbers drive")
        print("it and the surrounding context does not.")
    elif diverged < len(PAIRS):
        print("Mixed. Context changed the response in some pairs but not others.")
        print("Read the reasoning above to see whether the divergences cite the")
        print("contextual difference or are just stochastic variation.")
    else:
        print("Context changed the response in every pair. Check the reasoning")
        print("text actually cites the contextual difference -- if it does, this")
        print("is behaviour a threshold system cannot produce.")
    if repeats < 2:
        print("\nRun with --repeats 3 to separate real divergence from model noise.")

    out = ROOT / "agent_probe_results.json"
    out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\nfull transcripts: {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=1,
                    help="runs per scenario; >=2 separates signal from model noise")
    args = ap.parse_args()
    return asyncio.run(main_async(args.repeats))


if __name__ == "__main__":
    raise SystemExit(main())
