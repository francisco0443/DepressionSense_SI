import json
import math
import numpy as np
import matplotlib.pyplot as plt

# === CONFIGURATION ===
FILENAME = r"C:\Users\pc\Desktop\FEUP\Others\Summer Internship\Documentation\Dataset\Datasets\ULSTMAD\03-07-25\behavioral_data.json"

PAUSE_THRESHOLD_MS = 250
HOVER_RADIUS_PX = 10
HOVER_DURATION_MS = 1000

def euclidean_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# === LOAD JSON ===
with open(FILENAME, "r", encoding="utf-8") as f:
    sessions = json.load(f)

# === STORAGE ===
global_pauses = []
global_hovers = []
hover_candidates_global = []

# === PROCESS FIRST 40 SESSIONS ===
for session in sessions[:40]:
    for module_name, module_data in session.items():
        if not module_name.startswith("behavior_"):
            continue

        mouse_moves = module_data.get("mouseMovements", [])
        click_details = module_data.get("clicks", {}).get("clickDetails", [])

        # Combine mouse moves and clicks, sorted by timestamp
        combined_events = sorted(mouse_moves + click_details, key=lambda e: e["timestamp"])
        if len(combined_events) < 2:
            continue

        hover_start_time = None
        hover_start_pos = None

        for i in range(1, len(combined_events)):
            prev = combined_events[i - 1]
            curr = combined_events[i]

            time_diff = curr["timestamp"] - prev["timestamp"]
            dx = curr["x"] - prev["x"]
            dy = curr["y"] - prev["y"]
            dist = euclidean_distance(curr["x"], curr["y"], prev["x"], prev["y"])

            # Track pauses (all time differences)
            global_pauses.append(time_diff)

            # Hover detection logic
            if dist < HOVER_RADIUS_PX:
                if hover_start_time is None:
                    hover_start_time = prev["timestamp"]
                    hover_start_pos = (prev["x"], prev["y"])
                
                # Check if current event is a click (presence of 'node' means click)
                if "node" in curr and hover_start_time is not None:
                    dwell_time = curr["timestamp"] - hover_start_time
                    hover_candidates_global.append(dwell_time)
                    if dwell_time >= HOVER_DURATION_MS:
                        global_hovers.append(dwell_time)
                    hover_start_time = None
                    hover_start_pos = None

            else:
                # Hover ended due to movement
                if hover_start_time is not None:
                    dwell_time = prev["timestamp"] - hover_start_time
                    hover_candidates_global.append(dwell_time)
                    if dwell_time >= HOVER_DURATION_MS:
                        global_hovers.append(dwell_time)
                    hover_start_time = None
                    hover_start_pos = None

        # Handle hover ongoing at end of module
        if hover_start_time is not None:
            dwell_time = combined_events[-1]["timestamp"] - hover_start_time
            hover_candidates_global.append(dwell_time)
            if dwell_time >= HOVER_DURATION_MS:
                global_hovers.append(dwell_time)
                hover_candidates_global.append(dwell_time)

"""# === VALIDATE THRESHOLDS ===
print("\n--- Pause Threshold Validation ---")
for pause_thr in range(100, 1100, 100):
    count_long_pauses = len([p for p in global_pauses if p > pause_thr])
    print(f"Pauses > {pause_thr} ms: {count_long_pauses}")

print("\n--- Hover Duration Threshold Validation ---")
for hover_thr in range(500, 2500, 250):
    count_valid_hovers = len([h for h in global_hovers if h >= hover_thr])
    print(f"Hovers >= {hover_thr} ms: {count_valid_hovers}")"""

# === PERCENTILES ===
def print_percentiles(values, label):
    if not values:
        print(f"\nNo {label} data available")
        return

    print(f"\n--- {label} Percentiles (ms) ---")
    for p in [50, 75, 90, 95, 99]:
        val = np.percentile(values, p)
        print(f"{p}th percentile: {val:.2f} ms")

print_percentiles(global_pauses, "All Pauses")
print_percentiles(hover_candidates_global, "All Valid Hovers")
print(len(global_hovers)/len(hover_candidates_global))
