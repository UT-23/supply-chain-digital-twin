import asyncio
import logging
from datetime import datetime, timezone
from contextlib import asynccontextmanager

from fastapi import Body, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select, text

import twin_state
from agent import service as agent_module
from agent.service import agent_service, serialize_decision
from db.models import AgentDecision, Event, Warehouse
from db.session import SessionLocal, engine, pool_status
from simulation import config as sim_config
from simulation.worker import worker
from ws import hub

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("app")

# How many past decisions a newly connected client receives, so the AI log is
# populated immediately instead of staying blank until the next decision.
DECISION_BACKLOG = 8


async def broadcast_twin_state() -> None:
    """Push the current twin snapshot. Called after each simulation cycle."""
    if hub.client_count == 0:
        return
    snapshot = await asyncio.to_thread(_read_snapshot)
    await hub.broadcast({"type": "twin_state", "state": snapshot})


def _read_snapshot() -> dict:
    with SessionLocal() as session:
        return twin_state.build_snapshot(session)


def _read_recent_decisions(limit: int = DECISION_BACKLOG) -> list[dict]:
    with SessionLocal() as session:
        rows = session.scalars(
            select(AgentDecision).order_by(AgentDecision.id.desc()).limit(limit)
        ).all()
        return [serialize_decision(r) for r in rows][::-1]


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Wire the simulation's post-cycle hook and the agent's decision hook to the
    # WebSocket hub.
    worker._on_cycle = broadcast_twin_state
    agent_service._broadcast = hub.broadcast

    if sim_config.ENABLED:
        worker.start()
    else:
        log.warning("simulation worker disabled (SIM_ENABLED=0)")

    if agent_module.ENABLED:
        agent_service.start()
    else:
        log.warning("agent service disabled (AGENT_ENABLED=0)")

    try:
        yield
    finally:
        await agent_service.stop()
        await worker.stop()
        engine.dispose()
        log.info("engine disposed")


app = FastAPI(title="Supply Chain Digital Twin", lifespan=lifespan)

# The frontend dev server runs on a different origin (localhost:3000) from the
# API (localhost:8000), so the browser sends a CORS preflight before POSTing to
# /api/demo/inject. Without this the preflight OPTIONS got a bare 405 and every
# demo injection button failed in the browser -- while curl/python clients, which
# send no preflight, worked fine. Credentials are off: there is no auth to leak.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    """
    Live feed for the dashboard.

    On connect the client receives the current twin state plus the most recent
    agent decisions, then streams `twin_state` and `agent_decision` messages as
    they happen.
    """
    await hub.connect(websocket)
    try:
        snapshot = await asyncio.to_thread(_read_snapshot)
        await hub.send_to(websocket, {"type": "twin_state", "state": snapshot})

        for decision in await asyncio.to_thread(_read_recent_decisions):
            await hub.send_to(websocket, {"type": "agent_decision", "decision": decision})

        # Hold the connection open. The client is not expected to send anything;
        # receive() is how we notice it has gone away.
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    except Exception:  # noqa: BLE001
        log.debug("ws connection closed", exc_info=True)
    finally:
        await hub.disconnect(websocket)


# Disruption kinds the demo buttons can inject. Each writes a REAL row into
# `events`, which the agent picks up on its next poll exactly like a
# simulation-generated disruption -- there is no separate demo code path.
DEMO_DISRUPTIONS = {
    "humidity": {
        "event_type": "SENSOR_SPIKE", "severity": "warning", "reading": "humidity",
        "value": 71.5, "band": [40.0, 64.0],
    },
    "temp": {
        "event_type": "SENSOR_SPIKE", "severity": "critical", "reading": "temperature",
        "value": 31.2, "band": [16.5, 27.5],
    },
    "armfail": {
        "event_type": "SENSOR_SPIKE", "severity": "warning", "reading": "robot_arm_1_active",
        "value": 0.0, "band": [1.0, 1.0],
    },
    "dc": {
        "event_type": "SENSOR_SPIKE", "severity": "critical", "reading": "capacity_pct",
        "value": 94.0, "band": [20.0, 98.0],
    },
}


@app.post("/api/demo/inject")
async def demo_inject(payload: dict = Body(...)):
    """
    Inject a real disruption for a live demo.

    This is a genuine write to the `events` table, not a UI-only effect: the
    agent's normal poll finds it as an unhandled event and reasons about it like
    any other. That is the whole point -- a viva demo should exercise the real
    pipeline.

    NOTE: unauthenticated, and CORS is wide open. Fine for a local demo, not for
    anything exposed.
    """
    kind = str(payload.get("kind", "")).strip()
    spec = DEMO_DISRUPTIONS.get(kind)
    if spec is None:
        raise HTTPException(
            status_code=400,
            detail=f"unknown kind {kind!r}; expected one of {sorted(DEMO_DISRUPTIONS)}",
        )

    def _write() -> dict:
        with SessionLocal() as session:
            code = "DC1" if kind == "dc" else "WH1"
            wh = session.scalars(select(Warehouse).where(Warehouse.code == code)).first()
            if wh is None:
                wh = session.scalars(select(Warehouse).order_by(Warehouse.id)).first()
            if wh is None:
                raise HTTPException(503, "no warehouses loaded; run scripts/load_dataset.py")

            event = Event(
                occurred_at=datetime.now(timezone.utc),
                event_type=spec["event_type"],
                severity=spec["severity"],
                entity_type="warehouse",
                entity_id=wh.id,
                message=(
                    f"{spec['reading']} anomaly at {wh.code} (~{spec['value']}) "
                    f"[operator-injected]"
                ),
                details={
                    "warehouse_code": wh.code,
                    "reading_type": spec["reading"],
                    "approx_value": spec["value"],
                    "normal_range": spec["band"],
                    "source": "demo_injection",
                },
                injected=True,
                actor="system",
            )
            session.add(event)
            session.commit()
            return {
                "event_id": event.id,
                "warehouse": wh.code,
                "event_type": event.event_type,
                "severity": event.severity,
                "message": event.message,
            }

    written = await asyncio.to_thread(_write)
    log.info("demo injection: %s -> event %s", kind, written["event_id"])

    # Push the refreshed snapshot immediately so the UI reflects it without
    # waiting for the next simulation tick.
    await broadcast_twin_state()
    return {"ok": True, "injected": written,
            "note": "the agent will pick this up on its next poll"}


@app.get("/health")
def health():
    """Liveness of the process, database, simulation worker and agent service."""
    db: dict = {"connected": False}
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        db["connected"] = True
    except Exception as exc:  # noqa: BLE001 - health must never raise
        db["error"] = f"{type(exc).__name__}: {exc}"

    db["pool"] = pool_status()
    sim = worker.status()
    agent = agent_service.status()

    healthy = db["connected"] and (sim["healthy"] or not sim_config.ENABLED)
    return {
        "status": "ok" if healthy else "degraded",
        "database": db,
        "simulation": sim,
        "agent": agent,
        "websocket": {"clients": hub.client_count},
    }
