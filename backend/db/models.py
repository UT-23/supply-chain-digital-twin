"""
Normalized schema for the supply chain digital twin.

Source of truth for orders / order_items / shipments / products / customers is the
DataCo Smart Supply Chain dataset, which arrives as a single flat CSV at
order-*item* grain. The models below split that flat file into proper entities.

`inventory` and `warehouse_sensors` are NOT in the DataCo dataset -- they back the
warehouse/telemetry side of the twin and are seeded separately.
"""
from __future__ import annotations

import datetime as dt
from decimal import Decimal

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base

# Money: DataCo prices/sales are 2-decimal USD. Ratios get 6 places.
Money = Numeric(12, 2)
Ratio = Numeric(10, 6)


# -- Reference / lookup tables ---------------------------------------------


class Department(Base):
    __tablename__ = "departments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    name: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)

    categories: Mapped[list["Category"]] = relationship(back_populates="department")
    products: Mapped[list["Product"]] = relationship(back_populates="department")


class Category(Base):
    __tablename__ = "categories"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    department_id: Mapped[int | None] = mapped_column(
        ForeignKey("departments.id", ondelete="RESTRICT"), nullable=True, index=True
    )

    department: Mapped["Department | None"] = relationship(back_populates="categories")
    products: Mapped[list["Product"]] = relationship(back_populates="category")


class Customer(Base):
    __tablename__ = "customers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    first_name: Mapped[str | None] = mapped_column(String(128))
    last_name: Mapped[str | None] = mapped_column(String(128))
    segment: Mapped[str | None] = mapped_column(String(32), index=True)
    city: Mapped[str | None] = mapped_column(String(128))
    state: Mapped[str | None] = mapped_column(String(64))
    country: Mapped[str | None] = mapped_column(String(64))
    street: Mapped[str | None] = mapped_column(String(255))
    zipcode: Mapped[str | None] = mapped_column(String(16))

    orders: Mapped[list["Order"]] = relationship(back_populates="customer")


# -- Products --------------------------------------------------------------


class Product(Base):
    __tablename__ = "products"

    # DataCo "Product Card Id" -- the stable product key.
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    description: Mapped[str | None] = mapped_column(Text)
    price: Mapped[Decimal | None] = mapped_column(Money)
    status: Mapped[int | None] = mapped_column(SmallInteger)  # 0 = in stock, 1 = discontinued
    image_url: Mapped[str | None] = mapped_column(String(512))

    # -- set by the agent action layer (backend/agent/tools.py) --
    low_stock_flagged: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    flagged_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    flag_reason: Mapped[str | None] = mapped_column(Text)

    category_id: Mapped[int | None] = mapped_column(
        ForeignKey("categories.id", ondelete="RESTRICT"), index=True
    )
    department_id: Mapped[int | None] = mapped_column(
        ForeignKey("departments.id", ondelete="RESTRICT"), index=True
    )

    category: Mapped["Category | None"] = relationship(back_populates="products")
    department: Mapped["Department | None"] = relationship(back_populates="products")
    order_items: Mapped[list["OrderItem"]] = relationship(back_populates="product")
    inventory: Mapped[list["Inventory"]] = relationship(back_populates="product")

    __table_args__ = (
        CheckConstraint("price IS NULL OR price >= 0", name="ck_products_price_nonneg"),
    )


# -- Orders ----------------------------------------------------------------


class Order(Base):
    __tablename__ = "orders"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    customer_id: Mapped[int | None] = mapped_column(
        ForeignKey("customers.id", ondelete="RESTRICT"), index=True
    )

    ordered_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=False), index=True)
    status: Mapped[str | None] = mapped_column(String(32), index=True)
    order_type: Mapped[str | None] = mapped_column(String(32))  # DataCo "Type": payment method

    # Destination (order ship-to), distinct from the customer's own address.
    market: Mapped[str | None] = mapped_column(String(32), index=True)
    region: Mapped[str | None] = mapped_column(String(64))
    country: Mapped[str | None] = mapped_column(String(64))
    state: Mapped[str | None] = mapped_column(String(64))
    city: Mapped[str | None] = mapped_column(String(128))
    zipcode: Mapped[str | None] = mapped_column(String(16))
    latitude: Mapped[Decimal | None] = mapped_column(Numeric(9, 6))
    longitude: Mapped[Decimal | None] = mapped_column(Numeric(9, 6))

    # Order-level money (repeated on every CSV row; collapsed to one value here).
    sales: Mapped[Decimal | None] = mapped_column(Money)
    profit: Mapped[Decimal | None] = mapped_column(Money)
    benefit_per_order: Mapped[Decimal | None] = mapped_column(Money)

    customer: Mapped["Customer | None"] = relationship(back_populates="orders")
    items: Mapped[list["OrderItem"]] = relationship(
        back_populates="order", cascade="all, delete-orphan"
    )
    shipment: Mapped["Shipment | None"] = relationship(
        back_populates="order", cascade="all, delete-orphan", uselist=False
    )


class OrderItem(Base):
    """One CSV row = one order item. DataCo's 'Order Item Id' is globally unique."""

    __tablename__ = "order_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    order_id: Mapped[int] = mapped_column(
        ForeignKey("orders.id", ondelete="CASCADE"), nullable=False, index=True
    )
    product_id: Mapped[int | None] = mapped_column(
        ForeignKey("products.id", ondelete="RESTRICT"), index=True
    )

    quantity: Mapped[int | None] = mapped_column(Integer)
    unit_price: Mapped[Decimal | None] = mapped_column(Money)
    discount: Mapped[Decimal | None] = mapped_column(Money)
    discount_rate: Mapped[Decimal | None] = mapped_column(Ratio)
    sales: Mapped[Decimal | None] = mapped_column(Money)  # gross, before discount
    total: Mapped[Decimal | None] = mapped_column(Money)  # net, after discount
    profit_ratio: Mapped[Decimal | None] = mapped_column(Ratio)

    order: Mapped["Order"] = relationship(back_populates="items")
    product: Mapped["Product | None"] = relationship(back_populates="order_items")

    __table_args__ = (
        CheckConstraint("quantity IS NULL OR quantity >= 0", name="ck_order_items_qty_nonneg"),
    )


class Shipment(Base):
    """
    One shipment per order. DataCo carries shipping facts on every item row, but they
    are constant within an order, so this is modelled 1:1 with `orders`.
    """

    __tablename__ = "shipments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    order_id: Mapped[int] = mapped_column(
        ForeignKey("orders.id", ondelete="CASCADE"), nullable=False, unique=True
    )

    shipping_mode: Mapped[str | None] = mapped_column(String(32), index=True)
    # Historical DataCo outcome label ("Late delivery", "Shipping on time", ...).
    # The simulation never overwrites this -- it is loaded fact.
    delivery_status: Mapped[str | None] = mapped_column(String(32), index=True)
    shipped_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=False), index=True)

    days_scheduled: Mapped[int | None] = mapped_column(SmallInteger)
    days_actual: Mapped[int | None] = mapped_column(SmallInteger)
    # Positive = late. Derived: days_actual - days_scheduled.
    days_delta: Mapped[int | None] = mapped_column(SmallInteger, index=True)
    late_delivery_risk: Mapped[bool | None] = mapped_column(Boolean, index=True)

    # -- live simulation state (written by the background worker) --
    # Kept separate from delivery_status so historical data stays intact.
    live_status: Mapped[str | None] = mapped_column(String(24), index=True)
    eta: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    progress_pct: Mapped[int | None] = mapped_column(SmallInteger)
    delivered_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    delay_minutes: Mapped[int | None] = mapped_column(Integer)

    # -- set by the agent action layer (backend/agent/tools.py) --
    # Free-text lane description, e.g. "WH2 -> DC1 -> Customer (air)".
    current_route: Mapped[str | None] = mapped_column(String(255))
    rerouted_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    reroute_reason: Mapped[str | None] = mapped_column(Text)
    # ETA before the first reroute, so the effect of the change stays measurable.
    original_eta: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))

    order: Mapped["Order"] = relationship(back_populates="shipment")

    __table_args__ = (
        CheckConstraint(
            "progress_pct IS NULL OR (progress_pct >= 0 AND progress_pct <= 100)",
            name="ck_shipments_progress_range",
        ),
    )


# -- Warehouses, inventory, telemetry (not from DataCo) --------------------


class Warehouse(Base):
    __tablename__ = "warehouses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    code: Mapped[str] = mapped_column(String(32), nullable=False, unique=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    kind: Mapped[str | None] = mapped_column(String(32))  # warehouse | distribution_center | retail
    city: Mapped[str | None] = mapped_column(String(128))
    country: Mapped[str | None] = mapped_column(String(64))
    latitude: Mapped[Decimal | None] = mapped_column(Numeric(9, 6))
    longitude: Mapped[Decimal | None] = mapped_column(Numeric(9, 6))

    inventory: Mapped[list["Inventory"]] = relationship(back_populates="warehouse")
    sensor_readings: Mapped[list["WarehouseSensor"]] = relationship(back_populates="warehouse")


class Inventory(Base):
    """Stock position of one product at one warehouse."""

    __tablename__ = "inventory"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    warehouse_id: Mapped[int] = mapped_column(
        ForeignKey("warehouses.id", ondelete="CASCADE"), nullable=False
    )
    product_id: Mapped[int] = mapped_column(
        ForeignKey("products.id", ondelete="CASCADE"), nullable=False
    )

    quantity_on_hand: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    quantity_reserved: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    reorder_point: Mapped[int | None] = mapped_column(Integer)
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    warehouse: Mapped["Warehouse"] = relationship(back_populates="inventory")
    product: Mapped["Product"] = relationship(back_populates="inventory")

    __table_args__ = (
        UniqueConstraint("warehouse_id", "product_id", name="uq_inventory_warehouse_product"),
        CheckConstraint("quantity_on_hand >= 0", name="ck_inventory_on_hand_nonneg"),
        CheckConstraint("quantity_reserved >= 0", name="ck_inventory_reserved_nonneg"),
    )


class WarehouseSensor(Base):
    """
    Narrow time-series table for the telemetry the 3D twin visualizes
    (humidity, temperature, robot-arm state, conveyor throughput, ...).

    Kept as a tall (reading_type, value) table rather than one column per metric so
    new sensor types need no migration. `value` is numeric; boolean-ish readings
    (robot arm active) are stored as 0/1.
    """

    __tablename__ = "warehouse_sensors"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    warehouse_id: Mapped[int] = mapped_column(
        ForeignKey("warehouses.id", ondelete="CASCADE"), nullable=False
    )
    reading_type: Mapped[str] = mapped_column(String(48), nullable=False)
    value: Mapped[Decimal] = mapped_column(Numeric(12, 4), nullable=False)
    unit: Mapped[str | None] = mapped_column(String(16))
    recorded_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    warehouse: Mapped["Warehouse"] = relationship(back_populates="sensor_readings")

    __table_args__ = (
        # The read pattern is "latest N readings of type X for warehouse Y".
        Index("ix_warehouse_sensors_lookup", "warehouse_id", "reading_type", "recorded_at"),
    )


# -- Events ----------------------------------------------------------------


class Event(Base):
    """
    Append-only log of things that happened in the live system: injected
    disruptions, plus organically-occurring conditions worth reacting to.

    This is the queue the agent reads in Phase 4-5. `handled_at` is left NULL by
    the simulation; the agent claims an event by stamping it.
    """

    __tablename__ = "events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    occurred_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )

    # e.g. SHIPMENT_DELAY, SENSOR_SPIKE, STOCK_OUT, SHIPMENT_DELIVERED
    event_type: Mapped[str] = mapped_column(String(48), nullable=False, index=True)
    severity: Mapped[str] = mapped_column(String(16), nullable=False, default="info", index=True)

    # Loose reference to whatever the event is about. Not an FK: events outlive
    # the rows they describe, and they can point at different tables.
    entity_type: Mapped[str | None] = mapped_column(String(32), index=True)
    entity_id: Mapped[int | None] = mapped_column(Integer, index=True)

    message: Mapped[str | None] = mapped_column(Text)
    details: Mapped[dict | None] = mapped_column(JSON)

    # True = deliberately injected by the simulator; False = arose from normal flow.
    injected: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    # Who wrote this row: "simulation" | "agent" | "system". Lets the agent's own
    # actions be told apart from the conditions it was reacting to.
    actor: Mapped[str] = mapped_column(
        String(16), nullable=False, default="simulation", index=True
    )
    handled_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        # "unhandled events, newest first" -- the agent's polling query.
        Index("ix_events_unhandled", "handled_at", "occurred_at"),
    )


# -- Alerts ----------------------------------------------------------------


class Alert(Base):
    """
    Escalations raised for human attention.

    Distinct from `events`: an event is "something happened", an alert is
    "somebody needs to look at this". Alerts have a lifecycle (open ->
    acknowledged -> resolved); events are append-only and never mutate except
    for being marked handled.
    """

    __tablename__ = "alerts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )

    severity: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="open", index=True
    )

    # Loose reference, same rationale as Event: alerts outlive what they describe.
    entity_type: Mapped[str | None] = mapped_column(String(32), index=True)
    entity_id: Mapped[int | None] = mapped_column(Integer, index=True)

    # The event that triggered this escalation, when there was one.
    source_event_id: Mapped[int | None] = mapped_column(Integer, index=True)

    details: Mapped[dict | None] = mapped_column(JSON)
    created_by: Mapped[str] = mapped_column(String(16), nullable=False, default="agent")

    acknowledged_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    resolved_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    resolution_note: Mapped[str | None] = mapped_column(Text)

    __table_args__ = (
        CheckConstraint(
            "severity IN ('info', 'warning', 'critical')", name="ck_alerts_severity"
        ),
        CheckConstraint(
            "status IN ('open', 'acknowledged', 'resolved')", name="ck_alerts_status"
        ),
        # "open alerts, most severe first" -- the dashboard query.
        Index("ix_alerts_open", "status", "created_at"),
    )


# -- Agent decisions -------------------------------------------------------


class AgentDecision(Base):
    """
    One pass of the LLM agent: what it saw, what it concluded, what it did.

    Written by backend/agent/service.py on every decision cycle, including
    cycles where the model deliberately chose to do nothing -- a decision not to
    act is still a decision, and hiding those would make the log look more
    decisive than the agent actually is.
    """

    __tablename__ = "agent_decisions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )

    # Why this cycle ran: "events" (new unhandled events) or "heartbeat" (timer).
    trigger: Mapped[str] = mapped_column(String(16), nullable=False, index=True)

    # The exact perceived state sent to the model, so a decision can be replayed
    # and audited against what it actually had in front of it.
    input_state: Mapped[dict | None] = mapped_column(JSON)

    # The model's own words. This is what the UI shows as agent reasoning.
    reasoning: Mapped[str | None] = mapped_column(Text)

    # [{tool, input, ok, result|error}, ...] in the order they were executed.
    tool_calls: Mapped[list | None] = mapped_column(JSON)
    tool_results: Mapped[list | None] = mapped_column(JSON)

    action_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # True when the model looked and chose to take no action.
    no_action: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    model: Mapped[str | None] = mapped_column(String(64))
    input_tokens: Mapped[int | None] = mapped_column(Integer)
    output_tokens: Mapped[int | None] = mapped_column(Integer)
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    stop_reason: Mapped[str | None] = mapped_column(String(32))

    # Set when the cycle failed (API error, timeout). reasoning stays NULL.
    error: Mapped[str | None] = mapped_column(Text)

    # Set once the outcome scorer has evaluated this decision, so it is not
    # re-scanned forever. NULL = still inside its observation window.
    outcome_scored_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))

    # How many past cases were retrieved and shown to the model for this decision.
    # 0 means it reasoned without memory -- the control condition for any A/B.
    retrieved_case_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    retrieved_case_ids: Mapped[list | None] = mapped_column(JSON)
    # Whether the reasoning text actually REFERS to the retrieved cases, as
    # opposed to merely having been given them. This is a text heuristic, not
    # ground truth -- see agent.service._cites_memory. Kept distinct from
    # retrieved_case_count because "memory was supplied" and "memory was used"
    # are different claims and the UI must not conflate them.
    cites_memory: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    # The structured situation fingerprint used for retrieval, stored so past
    # decisions can be matched without recomputing from the full snapshot.
    signature: Mapped[dict | None] = mapped_column(JSON)

    outcomes: Mapped[list["DecisionOutcome"]] = relationship(
        back_populates="decision", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_agent_decisions_recent", "created_at", "no_action"),
        Index("ix_agent_decisions_pending_scoring", "outcome_scored_at", "created_at"),
    )


class DecisionOutcome(Base):
    """
    What actually happened after a decision, judged from database state.

    One row per condition the decision was reacting to, so a cycle that saw three
    disruptions gets three outcome rows.

    IMPORTANT -- these labels are correlational, not causal. The database can say
    "the condition cleared"; it cannot say "the condition cleared *because* the
    agent acted". `attribution` records whether the agent had any tool capable of
    influencing this condition type at all, which is the honest way to read a
    "resolved after action" label.
    """

    __tablename__ = "decision_outcomes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    decision_id: Mapped[int] = mapped_column(
        ForeignKey("agent_decisions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    scored_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    window_seconds: Mapped[int] = mapped_column(Integer, nullable=False)

    # The condition being followed up.
    subject_type: Mapped[str] = mapped_column(String(48), nullable=False, index=True)
    subject_event_id: Mapped[int | None] = mapped_column(Integer, index=True)
    subject_entity_type: Mapped[str | None] = mapped_column(String(32))
    subject_entity_id: Mapped[int | None] = mapped_column(Integer)

    # cleared | persisted | worsened | unknown
    condition_status: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    agent_acted: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    # self_resolved | missed | resolved_after_action | action_ineffective | unknown
    label: Mapped[str] = mapped_column(String(32), nullable=False, index=True)

    # agent_influenceable | self_healing | unknown -- whether any tool could have
    # affected this condition type. See outcomes.ATTRIBUTION.
    attribution: Mapped[str] = mapped_column(String(24), nullable=False, default="unknown")

    time_to_clear_seconds: Mapped[int | None] = mapped_column(Integer)
    detail: Mapped[dict | None] = mapped_column(JSON)

    decision: Mapped["AgentDecision"] = relationship(back_populates="outcomes")
