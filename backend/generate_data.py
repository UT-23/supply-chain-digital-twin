import csv
import math
import random

random.seed(42)

TICKS = 720
OUTPUT_FILE = "synthetic_data.csv"

def generate_data():
    rows = []

    wh1_humidity_base = 52.0
    wh2_humidity_base = 48.0
    factory_temp_base = 22.0
    dc_capacity_base = 55.0
    wh1_inventory = 800
    wh2_inventory = 600
    retail_stock = 400

    # Define crisis scenarios by tick range
    # Each scenario: (start_tick, end_tick, type)
    scenarios = [
        (80,  120,  'gradual_humidity'),
        (200, 210,  'sudden_mold'),
        (300, 340,  'gradual_temp'),
        (400, 420,  'arm_fail'),
        (500, 540,  'dc_overflow'),
        (600, 650,  'gradual_humidity'),
        (680, 700,  'sudden_mold'),
    ]

    def in_scenario(tick, stype):
        for s, e, t in scenarios:
            if s <= tick <= e and t == stype:
                return True, (tick - s) / (e - s)
        return False, 0.0

    for tick in range(1, TICKS + 1):
        t = tick * 0.1

        # ── WH1 Humidity ──────────────────────────────────────────
        in_grad_hum, progress = in_scenario(tick, 'gradual_humidity')
        in_mold, mold_progress = in_scenario(tick, 'sudden_mold')

        if in_grad_hum:
            # Gradually rises from baseline to 74% over the scenario
            wh1_humidity = round(
                52 + (22 * progress) + random.uniform(-1, 1), 1
            )
        elif in_mold:
            # Sudden spike — jumps immediately to 72-78%
            wh1_humidity = round(
                72 + random.uniform(0, 6), 1
            )
        else:
            # Normal oscillation
            wh1_humidity = round(
                wh1_humidity_base +
                10 * math.sin(t * 0.7) +
                random.uniform(-2, 2), 1
            )
        wh1_humidity = max(30.0, min(85.0, wh1_humidity))

        # ── WH2 Humidity ──────────────────────────────────────────
        wh2_humidity = round(
            wh2_humidity_base +
            8 * math.sin(t * 0.5 + 1) +
            random.uniform(-2, 2), 1
        )
        wh2_humidity = max(30.0, min(75.0, wh2_humidity))

        # ── Factory Temperature ────────────────────────────────────
        in_temp, temp_progress = in_scenario(tick, 'gradual_temp')
        if in_temp:
            # Gradually rises to 31°C
            factory_temp = round(
                22 + (10 * temp_progress) + random.uniform(-0.5, 0.5), 1
            )
        else:
            factory_temp = round(
                factory_temp_base +
                5 * math.sin(t * 0.3) +
                random.uniform(-0.5, 0.5), 1
            )
        factory_temp = max(15.0, min(35.0, factory_temp))

        # ── DC Capacity ────────────────────────────────────────────
        in_dc, dc_progress = in_scenario(tick, 'dc_overflow')
        if in_dc:
            # Rises to 95% during overflow scenario
            dc_capacity = round(
                55 + (40 * dc_progress) + random.uniform(-2, 2), 1
            )
        else:
            dc_capacity = round(
                dc_capacity_base +
                20 * math.sin(t * 0.4 + 2) +
                random.uniform(-3, 3), 1
            )
        dc_capacity = max(20.0, min(98.0, dc_capacity))

        # ── WH1 Inventory ──────────────────────────────────────────
        wh1_inventory = max(
            100,
            wh1_inventory - random.randint(1, 3) + random.randint(0, 1)
        )

        # ── WH2 Inventory ──────────────────────────────────────────
        wh2_inventory = max(
            100,
            wh2_inventory - random.randint(0, 2) + random.randint(0, 1)
        )

        # ── Retail Stock ───────────────────────────────────────────
        retail_stock = max(
            200,
            retail_stock - random.randint(0, 2) + random.randint(0, 1)
        )

        # ── Mold Detected ──────────────────────────────────────────
        in_mold_scenario, _ = in_scenario(tick, 'sudden_mold')
        if in_mold_scenario:
            mold_detected = random.random() < 0.85  # 85% chance during mold scenario
        else:
            mold_detected = random.random() < 0.01  # 1% chance normally

        # ── Arm Failure ────────────────────────────────────────────
        in_arm, _ = in_scenario(tick, 'arm_fail')
        if in_arm:
            arm_active = random.random() > 0.80  # 80% chance of failure during scenario
        else:
            arm_active = random.random() > 0.01  # 1% chance normally

        # ── Truck Cargo ────────────────────────────────────────────
        truck_cargo = tick % 11

        # ── DC Throughput ──────────────────────────────────────────
        throughput = random.randint(280, 420)

        # ── Shipment values ────────────────────────────────────────
        shipment_humidity = round(wh1_humidity + random.uniform(-3, 3), 1)
        shipment_temp = round(factory_temp + random.uniform(-1, 2), 1)

        # ── WH2 Temperature ────────────────────────────────────────
        wh2_temp = round(18 + random.uniform(-2, 3), 1)

        rows.append({
            'tick': tick,
            'wh1_humidity': wh1_humidity,
            'wh1_temperature': factory_temp,
            'wh1_inventory': wh1_inventory,
            'wh2_humidity': wh2_humidity,
            'wh2_temperature': wh2_temp,
            'wh2_inventory': wh2_inventory,
            'dc_capacity': dc_capacity,
            'dc_throughput': throughput,
            'shipment_humidity': shipment_humidity,
            'shipment_temperature': shipment_temp,
            'arm1_active': arm_active,
            'arm2_active': arm_active,
            'truck_cargo': truck_cargo,
            'mold_detected': mold_detected,
            'retail_stock': retail_stock,
        })

    return rows

def main():
    rows = generate_data()

    fieldnames = [
        'tick', 'wh1_humidity', 'wh1_temperature', 'wh1_inventory',
        'wh2_humidity', 'wh2_temperature', 'wh2_inventory',
        'dc_capacity', 'dc_throughput',
        'shipment_humidity', 'shipment_temperature',
        'arm1_active', 'arm2_active',
        'truck_cargo', 'mold_detected', 'retail_stock'
    ]

    with open(OUTPUT_FILE, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated {len(rows)} ticks of synthetic data")
    print(f"Saved to: {OUTPUT_FILE}")
    print()
    print("Crisis scenarios embedded:")
    print("  Ticks 080-120  : Gradual humidity rise → breach 68%")
    print("  Ticks 200-210  : Sudden mold outbreak")
    print("  Ticks 300-340  : Gradual temperature rise → breach 28°C")
    print("  Ticks 400-420  : Robot arm failure")
    print("  Ticks 500-540  : DC overflow → breach 88%")
    print("  Ticks 600-650  : Second humidity rise")
    print("  Ticks 680-700  : Second mold outbreak")

if __name__ == "__main__":
    main()