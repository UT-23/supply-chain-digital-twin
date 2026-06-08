from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import csv
import os

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load CSV once at startup ──────────────────────────────────────
DATA_FILE = os.path.join(os.path.dirname(__file__), "synthetic_data.csv")

def load_csv():
    rows = []
    with open(DATA_FILE, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                'tick':                 int(row['tick']),
                'wh1_humidity':         float(row['wh1_humidity']),
                'wh1_temperature':      float(row['wh1_temperature']),
                'wh1_inventory':        int(row['wh1_inventory']),
                'wh2_humidity':         float(row['wh2_humidity']),
                'wh2_temperature':      float(row['wh2_temperature']),
                'wh2_inventory':        int(row['wh2_inventory']),
                'dc_capacity':          float(row['dc_capacity']),
                'dc_throughput':        int(row['dc_throughput']),
                'shipment_humidity':    float(row['shipment_humidity']),
                'shipment_temperature': float(row['shipment_temperature']),
                'arm1_active':          row['arm1_active'] == 'True',
                'arm2_active':          row['arm2_active'] == 'True',
                'truck_cargo':          int(row['truck_cargo']),
                'mold_detected':        row['mold_detected'] == 'True',
                'retail_stock':         int(row['retail_stock']),
            })
    return rows

DATA = load_csv()
current_tick = 0

print(f"Loaded {len(DATA)} ticks from synthetic_data.csv")

# ── Main endpoint ─────────────────────────────────────────────────
@app.get("/api/supplychain")
def get_supply_chain():
    global current_tick

    row = DATA[current_tick % len(DATA)]
    current_tick += 1

    return {
        "warehouse1": {
            "humidity":    row['wh1_humidity'],
            "temperature": row['wh1_temperature'],
            "inventory":   row['wh1_inventory'],
        },
        "warehouse2": {
            "humidity":    row['wh2_humidity'],
            "temperature": row['wh2_temperature'],
            "inventory":   row['wh2_inventory'],
        },
        "distCenter": {
            "capacity":   row['dc_capacity'],
            "throughput": row['dc_throughput'],
        },
        "shipment1": {
            "humidity":    row['shipment_humidity'],
            "temperature": row['shipment_temperature'],
        },
        "arm1": { "active": row['arm1_active'] },
        "arm2": { "active": row['arm2_active'] },
        "extras": {
            "truck1_cargo":  row['truck_cargo'],
            "mold_detected": row['mold_detected'],
            "retail_stock":  row['retail_stock'],
            "factory_temp":  row['wh1_temperature'],
            "current_tick":  row['tick'],
        }
    }

@app.get("/health")
def health():
    return {
        "status": "ok",
        "current_tick": current_tick,
        "total_ticks": len(DATA),
        "data_source": "synthetic_csv"
    }