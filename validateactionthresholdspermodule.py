from collections import defaultdict

import json
import math
import matplotlib.pyplot as plt
import numpy as np

# === CONFIGURATION ===
FILENAME = r"C:\Users\pc\Desktop\FEUP\Others\Summer Internship\Documentation\Dataset\Datasets\ULSTMAD\03-07-25\behavioral_data.json"
PAUSE_THRESHOLD_MS = 2000
HOVER_RADIUS_PX = 10
HOVER_DURATION_MS = 1000

def euclidean_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# === LOAD JSON ===
with open(FILENAME, "r", encoding="utf-8") as f:
    sessions = json.load(f)

# === STORAGE ===
session_results = []  # Store results per session
global_pauses = []
global_hovers = []

# Store pause and hover percentiles for each module
pause_percentiles_by_module = defaultdict(list)
hover_percentiles_by_module = defaultdict(list)

for session in sessions[:40]:
    for module_name, module_data in session.items():
        if not module_name.startswith("behavior_"):
            continue

        mouse_moves = module_data.get("mouseMovements", [])
        click_details = module_data.get("clicks", {}).get("clickDetails", [])
        global_pauses = []
        global_hovers = []

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
                    if dwell_time >= HOVER_DURATION_MS:
                        global_hovers.append(dwell_time)
                    hover_start_time = None
                    hover_start_pos = None

            else:
                # Hover ended due to movement
                if hover_start_time is not None:
                    dwell_time = prev["timestamp"] - hover_start_time
                    if dwell_time >= HOVER_DURATION_MS:
                        global_hovers.append(dwell_time)
                    hover_start_time = None
                    hover_start_pos = None

        # Handle hover ongoing at end of module
        if hover_start_time is not None:
            dwell_time = combined_events[-1]["timestamp"] - hover_start_time
            if dwell_time >= HOVER_DURATION_MS:
                global_hovers.append(dwell_time)

        if global_pauses:
            pause_percentiles = [np.percentile(global_pauses, p) for p in [50, 75, 90, 95, 99]]
            pause_percentiles_by_module[module_name].append(pause_percentiles)

        if global_hovers:
            hover_percentiles = [np.percentile(global_hovers, p) for p in [50, 75, 90, 95, 99]]
            hover_percentiles_by_module[module_name].append(hover_percentiles)

# === Print Averaged Percentiles by Module ===
def print_avg_module_percentiles(percentiles_by_module, label):
    print(f"\n===== AVERAGED {label} PERCENTILES BY MODULE =====")
    for module_name, percentiles_list in percentiles_by_module.items():
        if not percentiles_list:
            continue
        averages = np.mean(percentiles_list, axis=0)
        print(f"\nModule {module_name}:")
        for p, avg in zip([50, 75, 90, 95, 99], averages):
            print(f"  {p}th percentile: {avg:.2f} ms")

print_avg_module_percentiles(pause_percentiles_by_module, "PAUSE")
print_avg_module_percentiles(hover_percentiles_by_module, "HOVER")
print(len(sessions))
