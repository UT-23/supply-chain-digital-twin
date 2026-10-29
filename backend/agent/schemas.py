"""
JSON schemas for the Phase 4 tools, in OpenAI function-calling format.

The agent runs against Groq's OpenAI-compatible endpoint, which expects

    {"type": "function", "function": {"name", "description", "parameters"}}

rather than Anthropic's {"name", "description", "input_schema"}. The underlying
functions in tools.py are unchanged -- only how they are described to the model.

Descriptions are taken verbatim from each function's docstring -- that was the
point of writing them carefully. The `session` parameter is deliberately not
exposed: it is an implementation detail of how the tool talks to the database and
has no meaning to the model.
"""
from __future__ import annotations

import inspect
import os
from typing import Any, Callable

from . import tools

# Per-parameter descriptions. The docstrings carry these in prose; the API wants
# them per-field, and repeating them here keeps the schema self-contained.
PARAM_DOCS: dict[str, dict[str, str]] = {
    "get_open_disruptions": {
        "limit": "Maximum number of disruptions to return, newest first (1-200).",
        "min_severity": (
            'Lowest severity to include. "info" returns everything, "warning" skips '
            'routine noise, "critical" returns only the most serious.'
        ),
    },
    "get_shipment_details": {
        "shipment_id": (
            "The shipment's numeric id, as found on a SHIPMENT_DELAY disruption's entity_id."
        ),
    },
    "get_inventory_status": {
        "product_id": (
            "The product's numeric id, from a STOCK_OUT disruption's details or an order item."
        ),
    },
    "reroute_shipment": {
        "shipment_id": "The shipment to reroute.",
        "new_route": (
            'Human-readable description of the new path, e.g. "WH2 -> DC1 -> Customer '
            '(air freight)". Must be non-empty.'
        ),
        "eta_shift_minutes": (
            "How much to move the ETA, in minutes. Negative pulls arrival earlier (the "
            "usual case when recovering time); positive pushes it later; 0 leaves it "
            "unchanged. Between -20160 and 20160."
        ),
        "reason": "Short explanation of why the shipment is being rerouted, stored for audit.",
    },
    "flag_low_inventory": {
        "product_id": "The product to flag.",
        "reason": (
            'Short explanation of why it is being flagged, e.g. "out of stock at WH2, '
            '3 open orders".'
        ),
    },
    "escalate_alert": {
        "severity": 'One of "info", "warning" or "critical".',
        "reason": (
            "What is wrong and why a human is needed. Be specific -- this is the entire "
            "content of the alert for whoever reads it."
        ),
        "related_entity_id": "Numeric id of the thing the alert is about, if any.",
        "related_entity_type": (
            'What that id refers to: "shipment", "product", "warehouse", "inventory" '
            'or "order".'
        ),
        "source_event_id": (
            "The id of the disruption event that prompted this escalation, if it came from one."
        ),
    },
    "restock_inventory": {
        "product_id": "The product to restock.",
        "warehouse_id": "Which warehouse receives the units.",
        "quantity": "How many units to add. Positive whole number, at most 100000.",
        "reason": 'Short explanation, e.g. "emergency replenishment after stock-out".',
    },
    "mark_event_handled": {
        "event_id": "The id of the event to close out, from the disruptions list.",
        "resolution_note": (
            'Short record of what was done about it, e.g. "rerouted via DC1, ETA pulled '
            'in 4h" or "no action needed, reading returned to normal".'
        ),
    },
    "resolve_alert": {
        "alert_id": "The alert to close.",
        "resolution_note": "Required record of how it was resolved.",
    },
}

# OpenAI-style strict schema adherence. Groq accepts the field but support
# varies by model, so this defaults OFF here (it defaulted ON for Anthropic).
# Turn it on with AGENT_STRICT_TOOLS=1 if your chosen model honours it.
STRICT = os.getenv("AGENT_STRICT_TOOLS", "0").lower() in ("1", "true", "yes")

ENUMS: dict[tuple[str, str], list[str]] = {
    ("get_open_disruptions", "min_severity"): ["info", "warning", "critical"],
    ("escalate_alert", "severity"): ["info", "warning", "critical"],
    ("escalate_alert", "related_entity_type"): [
        "shipment", "product", "warehouse", "inventory", "order",
    ],
}


def _json_type(annotation: Any) -> str:
    text = str(annotation)
    if "int" in text and "str" not in text:
        return "integer"
    if "bool" in text:
        return "boolean"
    return "string"


def _summary(fn: Callable) -> str:
    """
    The tool description the model reads.

    Uses the whole docstring minus the Args/Returns/Raises machinery: the prose
    explains what the tool does and when to use it, which is what matters for
    tool selection, while the per-argument detail lives on the schema fields.
    """
    doc = inspect.getdoc(fn) or ""
    out: list[str] = []
    for line in doc.splitlines():
        if line.strip() in ("Args:", "Returns:", "Raises:"):
            break
        out.append(line)
    return "\n".join(out).strip()


def build_parameters(fn: Callable) -> dict[str, Any]:
    """JSON Schema for one tool's arguments, excluding `session`."""
    name = fn.__name__
    sig = inspect.signature(fn)
    properties: dict[str, Any] = {}
    required: list[str] = []

    for pname, param in sig.parameters.items():
        if pname == "session":  # implementation detail, not part of the tool contract
            continue

        prop: dict[str, Any] = {"type": _json_type(param.annotation)}
        enum = ENUMS.get((name, pname))
        if enum:
            prop["enum"] = enum
        desc = PARAM_DOCS.get(name, {}).get(pname)
        if desc:
            prop["description"] = desc

        # Optional params are typed as `X | None`; let the model omit them.
        if param.default is inspect.Parameter.empty:
            required.append(pname)

        properties[pname] = prop

    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def build_tool_schema(fn: Callable) -> dict[str, Any]:
    """
    One tool in OpenAI function-calling format, as Groq expects.

    Note the shape difference from Anthropic: the name/description/parameters are
    nested under a "function" key, and the schema key is "parameters", not
    "input_schema".
    """
    function: dict[str, Any] = {
        "name": fn.__name__,
        "description": _summary(fn),
        "parameters": build_parameters(fn),
    }
    if STRICT:
        function["strict"] = True

    return {"type": "function", "function": function}


# name -> callable, for dispatching what the model asks for.
TOOL_REGISTRY: dict[str, Callable] = {fn.__name__: fn for fn in tools.ALL_TOOLS}

# Read-only subset, for running the agent in a mode that cannot change anything.
READ_ONLY_NAMES = frozenset(fn.__name__ for fn in tools.READ_TOOLS)


# Keys the OpenAI schema carries that Gemini's OpenAPI-subset validator rejects
# (or silently mishandles) on its OpenAI-compatibility endpoint.
GEMINI_UNSUPPORTED_KEYS = ("additionalProperties", "strict", "$schema")


def _to_gemini(tool: dict[str, Any]) -> dict[str, Any]:
    """
    Adapt one OpenAI tool definition for Gemini.

    Gemini accepts OpenAI's {"type": "function", "function": {...}} envelope
    verbatim, so this is a subtraction rather than a translation: the name,
    description and parameter schema are passed through byte-identically, which
    is what keeps a Gemini run comparable with a Groq run of the same probe.
    Only the JSON-Schema keys Gemini's validator refuses are stripped.
    """
    def strip(node: Any) -> Any:
        if isinstance(node, dict):
            return {
                k: strip(v)
                for k, v in node.items()
                if k not in GEMINI_UNSUPPORTED_KEYS
            }
        if isinstance(node, list):
            return [strip(v) for v in node]
        return node

    return strip(tool)


def tool_schemas(read_only: bool = False, dialect: str = "openai") -> list[dict[str, Any]]:
    """
    Tool definitions in a stable order so the prompt caches.

    dialect="openai" suits Groq (and OpenAI itself); dialect="gemini" additionally
    strips the schema keys Gemini's compatibility endpoint will not accept.
    """
    fns = tools.READ_TOOLS if read_only else tools.ALL_TOOLS
    defs = [build_tool_schema(fn) for fn in sorted(fns, key=lambda f: f.__name__)]
    if dialect == "gemini":
        return [_to_gemini(d) for d in defs]
    return defs
