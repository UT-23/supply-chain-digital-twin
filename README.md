# Supply Chain Digital Twin — LLM Agent over a Live Database

A 3D supply chain digital twin backed by a real Postgres database, with a
background process that keeps that database changing, and an LLM agent that
observes the state, decides whether anything warrants action, and acts through a
typed tool layer.

Every decision the agent makes — its reasoning text and the tools it called — is
stored and pushed to the browser over a WebSocket. The "AI log" in the UI shows
what a model actually wrote, not canned strings.

---

## What this is

**A real database, not a CSV.** 14 normalized tables under Alembic migrations
(5 revisions). Order, shipment, product, customer and inventory data comes from
the DataCo Smart Supply Chain dataset; warehouse telemetry and disruptions are
generated, because that dataset has no sensor readings or stock levels.

**A simulation that makes it live.** A background asyncio worker runs every few
seconds: advances in-transit shipments, draws inventory down against open
orders, writes fresh sensor readings, and on a low randomized probability
injects one disruption (shipment delay, sensor spike, or stock-out) into an
`events` table.

**An LLM agent with tool use.** On each cycle the agent is given the current
state — raw sensor values with their normal operating bands, shipments,
inventory, unhandled events — and decides what to do. It has nine typed tools
(three read, six write) and can choose to do nothing. The system prompt contains
no numeric thresholds, and the state it sees contains no pre-computed
"is this a breach" flags; both would make it a threshold system with an API call
in the middle.

The agent is a **LangGraph state graph** — `perceive → reason → act`, with a
conditional edge that loops back to `reason` only while the model keeps asking
for tools and stops as soon as it answers in plain text:

```
START ─> perceive ─> reason ─┬─ tool calls? ─> act ─┐
                             │                      │
                             └─ no ─> END           └─> reason
```

`perceive` queries the database and retrieves comparable past cases; `reason` is
the model call with tools bound via `bind_tools`; `act` executes them. The tool
schemas handed to `bind_tools` are the same OpenAI-format dicts generated from
the Phase 4 docstrings, so the model sees byte-identical tool descriptions to
those used in the documented experiments.

It runs on **Groq's free tier** via `langchain-groq`. The reason is cost,
not capability: this is a loop that reasons on every disruption and on a
heartbeat besides, so a paid per-token API would bill continuously for a project
that is demonstrated rather than operated. Default model is
`openai/gpt-oss-120b`. Note that Groq's own docs recommend
`llama-3.3-70b-versatile`, but that model returns **404 model_not_found** on this
account — always confirm with `client.models.list()` rather than trusting the
docs. Swap models with `AGENT_MODEL`.

### Which model ran which experiment

The main agent runs on **Groq** (`openai/gpt-oss-120b`). The **memory A/B probe
can also run on Google Gemini** (`gemini-3.5-flash`, AI Studio free tier) via
`--provider gemini`.

That second provider exists because **one A/B run very nearly fills Groq's daily
free-tier budget.** Each call costs ~4,200 tokens (over half of it the nine tool
docstrings) and a 3-repeat run consumed **86,782 tokens across 22 requests** —
about 43% of Groq's 200,000/day. Attempting the judgment probe and the A/B on the
same day exhausts it, which aborted four earlier attempts mid-run. The result
reported below was finally obtained on Groq on a day with a clean budget.

Gemini remains wired up as the fallback, but its own free tier is the tighter
constraint: **5 RPM / 20 requests per day** (confirmed from the AI Studio
dashboard). A 3-repeat run needs ~18 requests, past the 80% guard ceiling of 16,
so Gemini can only host a reduced-scope run. `qwen/qwen3.8-27b` is unsuitable for
a different reason: its output-tokens-per-minute ceiling of 1,000 rejects a
single `max_tokens=4096` request outright.

Both providers are driven by the same LangGraph graph — Groq through
`langchain-groq`, Gemini through `langchain-openai` pointed at its
OpenAI-compatible endpoint — and both receive the same tool dicts via
`bind_tools`, so tool names, descriptions and parameter schemas are
**byte-identical** across providers; only JSON-Schema keys Gemini's validator
refuses (`additionalProperties`, `strict`) are stripped. That is what keeps a
Gemini run comparable with a Groq run.

**A single run never mixes providers.** `--provider` is applied before the agent
module is imported, so all three arms use one provider and one model; mixing
them would confound the effect being measured with the model answering. Every
stored transcript records the provider and model that produced it.

**Case memory.** After a fixed delay, each past decision is scored against what
actually happened: did the condition clear, persist, or worsen. Before each new
reasoning call, past decisions with a similar situation fingerprint are
retrieved and included in the prompt along with their outcomes.

**WebSocket push.** One connection carries both the twin state (for the 3D
scene) and agent decisions (for the log panel). No polling.

---

## What this is NOT

Stated plainly, because several of these are easy to overclaim:

- **Not multi-agent.** One agent, one loop. No orchestration, no delegation, no
  planner/executor split.
- **Not reinforcement learning, and no model training of any kind.** The memory
  is retrieval-augmented prompting: past cases are looked up and pasted into the
  prompt. Model weights never change. The accurate term is *case-based* or
  *experience-conditioned* decision-making, not "self-evolving."
- **Not connected to real IoT devices.** Sensor telemetry is generated with a
  sine-plus-noise model. It is synthetic, and labelled as such in the code.
- **Not using Azure Digital Twins**, despite an earlier `ADT_URL` variable that
  is referenced nowhere and should be deleted from your `.env`.
- **Not production code.** No authentication, CORS is wide open, no rate
  limiting, no deployment configuration, no test coverage of the HTTP layer.
- **Not proven to adapt.** The memory A/B shifted behaviour in the predicted
  direction with a large effect, but at n=3 per arm it is not statistically
  significant (p=0.10, the floor at that sample size). See *Memory A/B result*.

---

## Current status

Honest accounting of what has been executed versus what has only been built:

| Component | Status |
|---|---|
| Schema + 5 migrations | Verified — round-trips up/down/up, and **applied to live Supabase Postgres 17.6** (14 tables, 61 indexes, 7 CHECK constraints) |
| Dataset loader | **Verified against the real Kaggle CSV** — 180,519 rows loaded into live Supabase Postgres: 65,752 orders, 180,519 order_items, 65,752 shipments, 20,652 customers, 118 products. 0 orphan rows, 0 duplicate shipments |
| Simulation worker | Verified — rows demonstrably change over time, no connection leaks |
| WebSocket | Verified — twin state and decisions stream to a real client |
| Agent tool layer | Verified — 74 automated checks against the database |
| Memory scoring + retrieval | Verified — 35 automated checks |
| Groq request/response format | Verified against a mock OpenAI-compatible server: schema shape, role sequence, tool_call_id round-trip, parallel calls, malformed-JSON and unknown-tool handling |
| **LLM reasoning** | **Executed.** Judgment probe run against Groq `gpt-oss-120b`: 6/10 scenarios emitted a write action, 2/4 matched pairs diverged with reasoning citing the contextual difference (rerouted a shipment at 12% progress, declined at 96%; flagged a no-demand stock-out, restocked one with 14 open orders). **Different setup from the A/B below — see *Two experiments, two questions*** |
| **Whether memory changes behaviour** | **Measured, directionally strong, not statistically significant.** 3-arm A/B on Groq `gpt-oss-120b`, 9/9 valid runs: action rate **A (no memory) 100%, B (benign history) 0%, C (serious history) 67%**. All 3 arm-B runs cited the retrieved case as their stated reason for not acting; 0 arm-A runs referenced any case. Fisher exact A vs B p=0.10 — the floor at n=3/arm. See *Memory A/B result* |

Everything that does not require a model call has been run and tested against
the live database, and the agent's reasoning has been observed directly.

---

## Two experiments, two questions

The two model-facing results in the status table are **separate experiments run
under different conditions**. They are not two runs of one test, and their
numbers should not be compared with each other.

| | Judgment probe | Memory A/B probe |
|---|---|---|
| Question | Does the agent reason *contextually*, or is it a threshold system with an API call in the middle? | Does *retrieved case history* change what the agent does? |
| Script | `scripts/agent_judgment_probe.py` | `scripts/memory_ab_probe.py` |
| Fixture | **10 hand-built scenarios**, incl. 4 matched pairs holding a number constant while varying context | **1 scenario**, held byte-identical across 3 arms while only the case history varies |
| Scenario tuning | **Never retuned.** 4 of its scenarios still carry a hardcoded `severity="critical"` | **Retuned to the decision boundary** (66.1% humidity, `severity="warning"`, 180 units) after an earlier version pinned every arm at 100% |
| Repeats | 1 run per scenario | 3 runs per arm |
| Result | 6/10 emitted a write action; 2/4 pairs diverged | A 100% / B 0% / C 67%; p=0.10 |

Both ran on Groq `openai/gpt-oss-120b`, and **both ran after the
`AGENT_READ_ONLY` tool-visibility bug was fixed** — that bug had restricted the
model to the three read-only tools, pinning write actions at 0/10 and making any
"agent won't act" reading an artefact of the harness rather than the model. The
figures reported here are all post-fix.

The one condition that genuinely differs is the **fixture**: the A/B scenario was
deliberately moved to the decision boundary so a memory effect would have room to
show, whereas the judgment probe's scenarios were left as originally written,
several of them unambiguous by design (its two controls are *meant* to be
clear-cut). That is appropriate for what each measures, but it means the
judgment probe's action rate and the A/B's action rates are **not** comparable
quantities.

---

## Memory A/B result

The controlled experiment behind the status-table row above. One situation, held
byte-identical across three arms; only the retrieved case history varies. Tool
execution is stubbed, so nothing is mutated. Run on Groq `openai/gpt-oss-120b`,
`--repeats 3`, 9/9 runs valid (no rate-limit exclusions), 86,782 tokens.

Situation: humidity 66.1% against a 40–64% band, flat trend, `severity=warning`,
180 units on site — deliberately placed near the decision boundary.

| Arm | History shown | Valid runs | Acted | Cited a retrieved case |
|---|---|---|---|---|
| A | none (control) | 3/3 | **3/3 (100%)** | 0/3 |
| B | condition self-resolved 3× | 3/3 | **0/3 (0%)** | **3/3** |
| C | condition persisted/worsened 3× | 3/3 | **2/3 (67%)** | 1/3 |

Every arm-B run named the retrieved history as its reason for *not* acting,
unprompted:

> "…the same pattern has **self-resolved in three recent, similar cases**."

> "**Similar humidity spikes at this site have resolved themselves in the past**
> without intervention…"

No arm-A run referenced any past case, because none were retrieved for it.

### What this does and does not establish

**It establishes** that memory reaches the model's reasoning: retrieval,
outcome-scoring and prompt assembly work end to end, and the model reads the
cases and reasons from them explicitly. The direction of the behavioural shift
matches the prediction (benign history suppresses action), and the prose and the
action agree in every one of the nine runs.

**It does not establish statistical significance.** Fisher's exact two-tailed on
A (3/3) vs B (0/3) gives **p = 0.10** — and at n=3 per arm that is the *floor*,
so even perfect separation cannot reach p<0.05. Arm A's own baseline is unstable:
it measured 33% on this identical fixture in an earlier pilot versus 100% here.

The accurate claim is therefore: **memory demonstrably reaches the model's
reasoning and is associated with a large behavioural shift.** Not "proven
adaptive", and not reinforcement learning — see *What this is NOT*. Confirming
the effect statistically would need roughly 8 repeats per arm (~230k tokens,
more than one day of Groq's free-tier budget); that was not run.

Arm C is the weak arm by construction: with A already at 100%, a "condition
persisted" history had no headroom to push action higher. Only arm B could move,
and it moved fully. A two-sided test would need a baseline nearer 50%.

Reproduce with:

```bash
python scripts/memory_ab_probe.py --provider groq --repeats 3 \
  --rpd 1000 --tpd 200000 --rpm 1.4
```

---

## Quick start

```bash
pip install -r backend/requirements.txt
```

Set `DATABASE_URL` and `GROQ_API_KEY` in `backend/.env` (see
`backend/.env.example`; get a free Groq key at
<https://console.groq.com/keys>), put `DataCoSupplyChainDataset.csv` in
`data/raw/`, then:

```bash
cd backend && python -m alembic upgrade head
```

```bash
python scripts/inspect_dataset.py && python scripts/load_dataset.py
```

```bash
cd backend && python -m uvicorn main:app --reload --port 8000
```

```bash
cd frontend && npm install && npm start
```

Open `http://localhost:3000`. `GET /health` reports database, simulation and
agent liveness.

---

## Architecture

```
DataCo CSV ──> load_dataset.py ──> Postgres (14 tables)
                                       ^  |
                    simulation worker ──┘  |  every few seconds:
                    (shipments, sensors,   |  advance, draw down, emit events
                     injected disruptions) |
                                           v
                                    agent service
              perceive ─> retrieve past cases ─> Groq LLM (tool use)
                                           |
                                           v
                          execute tools ──> write agent_decisions
                                           |
                                           v
                                    WebSocket /ws
                                           |
                                           v
                              React + React Three Fiber
                         (3D scene + live agent reasoning log)
```

Outcome scoring runs on the same loop: decisions past their observation window
are re-checked against database state and labelled.

---

## Agent tools

Read: `get_open_disruptions`, `get_shipment_details`, `get_inventory_status`
Write: `reroute_shipment`, `flag_low_inventory`, `escalate_alert`,
`restock_inventory`, `mark_event_handled`, `resolve_alert`

Each is a plain typed Python function whose docstring becomes the tool
description the model reads. Every write logs an `events` row with
`actor="agent"`, so the agent's own actions are distinguishable from the
conditions it was reacting to.

---

## An important measurement caveat

The agent has no tool that can influence a sensor reading, and simulated sensor
spikes expire on a timer after 3–8 cycles. **Every `SENSOR_SPIKE` therefore
resolves regardless of what the agent does.** A "resolved after action" label on
one means nothing causally.

Outcome rows carry an `attribution` field (`self_healing` vs
`agent_influenceable`) recording this, and the prompt surfaces it to the model.
Of the three disruption types, only stock-outs and shipment delays are genuinely
affected by agent action. If you want memory to visibly matter, the simulation
needs a disruption that persists until something intervenes.

---

## Verification scripts

| Script | Needs API key | What it does |
|---|---|---|
| `scripts/inspect_dataset.py` | no | Profiles the raw CSV and checks schema assumptions |
| `scripts/load_dataset.py` | no | One-time load into Postgres |
| `scripts/watch_simulation.py` | no | Proves DB rows change while the server runs |
| `scripts/test_agent_tools.py` | no | 74 checks on the tool layer (rolls back) |
| `scripts/test_agent_memory.py` | no | 35 checks on outcome scoring and retrieval |
| `scripts/memory_report.py` | no | Early-vs-late analysis over a recorded run |
| `scripts/run_agent_once.py` | **yes** | One decision cycle, full transcript |
| `scripts/agent_judgment_probe.py` | **yes** | Matched-pair test: judgement or thresholds? |
| `scripts/memory_ab_probe.py` | **yes** | Controlled A/B: does memory change behaviour? `--provider groq\|gemini` |

The last two are the ones that decide whether the project's central claims hold.
Both print an explicit null-result verdict if the answer is no.

---

## Project structure

```
backend/
  agent/          tools, prompts, perception, memory, outcomes, service
  db/             SQLAlchemy models, session, pool config
  simulation/     background worker, sensor model, config
  alembic/        5 migrations
  main.py         FastAPI app: lifespan, /health, /ws
  ws.py           WebSocket hub
  twin_state.py   snapshot for the 3D scene
frontend/src/
  App.js          3D scene, dashboard, agent decision log
scripts/          loaders, tests, probes
data/raw/         drop the DataCo CSV here (gitignored)
```

---

## Tech stack

React 18 · React Three Fiber · Three.js · Chart.js · FastAPI · SQLAlchemy 2 ·
Alembic · PostgreSQL 17.6 on Supabase (psycopg 3) · pandas · LangGraph ·
langchain-groq · Groq
(OpenAI-compatible API, `openai/gpt-oss-120b`) · optional Google Gemini
(`gemini-3.5-flash`) for the memory A/B probe

---

## Course & institution

B.Tech CSE (AI & ML Specialization)
Course: 21CSP302L — Research Project
SRM Institute of Science and Technology, Kattankulathur

## License

MIT
