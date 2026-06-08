# Supply-Chain-Digital-Twin-Real-Time-3D-Visualization-Event-Driven-Crisis-Response
# Supply Chain Digital Twin — Event-Driven 3D Visualization

A proof-of-concept digital twin for textile supply chain operations. Built with React Three Fiber (3D visualization), FastAPI (backend), and synthetic data simulation.

## What This Is

- **Concept demonstration** of digital twin architecture
- **Event-driven system** that polls data every 2.5 seconds
- **Rule-based crisis detection** with 7 embedded scenarios
- **Autonomous response** that stops affected equipment and logs actions
- **Complete demo cycle** from crisis detection to 10-second recovery

## What This Is NOT

- Not connected to real IoT devices (uses synthetic CSV)
- Not machine learning (yet) — pure rule-based logic
- Not production code — college project
- Not truly self-evolving (aspirational title for Phase 2)

## Quick Start

**Backend:**
```bash
cd backend
python -m uvicorn main:app --reload --port 8000
```

**Frontend:**
```bash
cd frontend
npm install
npm start
```

Open `http://localhost:3000`

## Architecture
CSV Dataset (720 ticks)
↓
FastAPI Backend (serves 16 parameters)
↓
React Frontend (polls every 2.5s)
↓
Crisis Detection (rule-based)
↓
3D Scene + Dashboard Update
↓
10-second autonomous recovery
## Monitored Parameters

- WH1: humidity, temperature, inventory
- WH2: humidity, temperature, inventory  
- Factory: temperature, running status
- DC: capacity, throughput
- Equipment: arm status, truck cargo, mold detection
- Retail: stock level

## Crisis Types (Priority Order)

1. **Mold** → Stops WH1, Factory, Truck2, Arm1
2. **Humidity** → Stops Truck2, Arm1
3. **Temperature** → Stops Factory
4. **Arm Failure** → Stops both arms, Factory
5. **DC Overflow** → Stops Truck3, Conveyor
6. **Supply Shortage** → Logging only

## Tech Stack

- **Frontend:** React 18, React Three Fiber, Three.js, Chart.js
- **Backend:** FastAPI, Python, Uvicorn
- **Data:** Synthetic CSV (720 ticks)
- **Architecture:** Event-driven polling

## Project Structure
├── backend/
│   ├── main.py              # FastAPI server
│   ├── generate_data.py     # CSV generator
│   ├── synthetic_data.csv   # 720-tick dataset
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.js           # Main React component
│   │   └── App.css
│   ├── package.json
│   └── .gitignore
└── README.md
## Key Features

✅ Browser-based 3D visualization (WebGL)
✅ Real-time state propagation across 9 supply chain nodes
✅ Automatic equipment shutdown during crises
✅ AI-logged responses (timestamped)
✅ 10-second autonomous recovery cycle
✅ Supply Chain Health Score (0-100%)
✅ Demo buttons for manual crisis triggers
✅ Live telemetry chart (humidity, temperature)
✅ Equipment status panel

## Next Phases

**Phase 2:** ML anomaly detection on synthetic data
**Phase 3:** Real IoT integration (replace CSV with live telemetry)

## Course & Institution

B.Tech CSE (AI & ML Specialization)
Course: 21CSP302L — Research Project
SRM Institute of Science and Technology, Kattankulathur

## License

MIT
