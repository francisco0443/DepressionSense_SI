
import json
import pandas as pd
import math

# Parameters
HOVER_RADIUS_PX = 10
HOVER_DURATION_MS = 1000
PAUSE_THRESHOLD_MS=250

def euclidean_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def group_events_from_module(events, session_id, module_name):
    actions = []
    last_event = None
    hover_start = None
    hover_anchor = None
    active_movement = None  # Track current continuous movement
    last_pause = None 

    for event in events:
        # Ensure event has required fields
        if not all(key in event for key in ['timestamp', 'x', 'y']):
            continue

        # Initialize last_event on first iteration
        if last_event is None:
            last_event = event
            continue

        # Calculate movement metrics
        time_diff = event["timestamp"] - last_event["timestamp"]
        dist = euclidean_distance(event["x"], event["y"], last_event["x"], last_event["y"])

        # --- PAUSE DETECTION (must come first) ---
        if time_diff >= PAUSE_THRESHOLD_MS and dist<3 and "node" not in event:
            # If previous action is a pause and adjacent in time, merge this pause
            if last_pause and last_pause['end'] == last_event['timestamp']:
                # Merge by extending end time, duration and distance
                last_pause['end'] = event['timestamp']
                last_pause['duration'] += time_diff
                last_pause['distance'] += dist
            else:
                # Create new pause action
                pause_action = {
                    'type': 'pause',
                    'start': last_event['timestamp'],
                    'end': event['timestamp'],
                    'distance': dist,
                    'duration': time_diff,
                    'x': last_event['x'],
                    'y': last_event['y'],
                    'module': module_name,
                    'sessionId': session_id
                }
                actions.append(pause_action)
                last_pause = pause_action  # Update last_pause reference

            active_movement = None  # Reset movement tracking after pause
            hover_start= None
            hover_anchor = None
            last_event = event
            continue

        else:
            # If current event is not a pause, reset last_pause tracker
            last_pause = None
  
        # --- CLICK PROCESSING ---
        if 'node' in event:  # Click identification
            # Finalize any ongoing hover
            if hover_start is not None:
                dwell_time = event["timestamp"] - hover_start["timestamp"]
                if dwell_time >= HOVER_DURATION_MS:
                    actions.append({
                        'type': 'hover',
                        'start': hover_start["timestamp"],
                        'end': event["timestamp"],
                        'duration': dwell_time,
                        'x': hover_start["x"],
                        'y': hover_start["y"],
                        'module': module_name,
                            'sessionId': session_id
                        })
                hover_start = None
                hover_anchor = None
            
            # Register the click
            actions.append({
                'type': 'click',
                'timestamp': event['timestamp'],
                'x': event['x'],
                'y': event['y'],
                'node': event.get('node'),
                'module': module_name,
                'sessionId': session_id
            })
            active_movement = None  # Clicks break movement continuity
            last_event = event
            hover_start= None
            hover_anchor = None
            continue

        # --- HOVER DETECTION ---
        if hover_start is None:
            if dist < HOVER_RADIUS_PX:
                hover_start = {
                    'timestamp': last_event['timestamp'],
                    'x': last_event['x'],
                    'y': last_event['y']
                }
                hover_anchor = (last_event['x'], last_event['y'])
        else:
            anchor_dist = euclidean_distance(event["x"], event["y"], hover_anchor[0], hover_anchor[1])
            if anchor_dist > HOVER_RADIUS_PX:
                dwell_time = last_event["timestamp"] - hover_start["timestamp"]
                if dwell_time >= HOVER_DURATION_MS:
                    actions.append({
                        'type': 'hover',
                        'start': hover_start["timestamp"],
                        'end': last_event["timestamp"],
                        'duration': dwell_time,
                        'x': hover_start["x"],
                        'y': hover_start["y"],
                        'module': module_name,
                        'sessionId': session_id
                    })
                hover_start = None
                hover_anchor = None
                


        # --- MOVEMENT TRACKING ---
        if dist > 0:
            if active_movement and actions[-1]['type'] == 'mouse_movement':
                # Update existing movement
                actions[-1].update({
                    'end': event['timestamp'],
                    'duration': actions[-1]['duration'] + time_diff,
                    'distance': actions[-1]['distance'] + dist,
                    'path': actions[-1]['path'] + [(event['x'], event['y'], event['timestamp'])],
                    'x': event['x'],
                    'y': event['y'],
                    'timestamp': event['timestamp']
                })
            else:
                # Start new movement segment
                new_movement = {
                    'type': 'mouse_movement',
                    'start': last_event['timestamp'],
                    'end': event['timestamp'],
                    'duration': time_diff,
                    'distance': dist,
                    'path': [(last_event['x'], last_event['y'],last_event['timestamp']), (event['x'], event['y'],event['timestamp'])],
                    'module': module_name,
                    'sessionId': session_id,
                    'x': event['x'],
                    'y': event['y'],
                    'timestamp': event['timestamp']
                }
                actions.append(new_movement)
                active_movement = True

        last_event = event

    # Flush hover that reaches the end of the module.
    if hover_start is not None and last_event is not None:
        dwell_time = last_event["timestamp"] - hover_start["timestamp"]
        if dwell_time >= HOVER_DURATION_MS:
            actions.append({
                'type': 'hover',
                'start': hover_start["timestamp"],
                'end': last_event["timestamp"],
                'duration': dwell_time,
                'x': hover_start["x"],
                'y': hover_start["y"],
                'module': module_name,
                'sessionId': session_id
            })

    return actions


def extract_and_group_all_sessions(input_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        sessions = json.load(f)

    all_actions = []
    for session in sessions:
        participant = session.get("processNumber", "unknown")  # Extract participant ID
        session_id = session.get("sessionId", "unknown")
        
        for module_name in ['behavior_WAI_sr', 'behavior_EADS', 'behavior_RESS']:
            module = session.get(module_name)
            if module:
                # Combine and sort events
                raw_events = sorted(
                    module.get("mouseMovements", []) + 
                    module.get("clicks", {}).get("clickDetails", []),
                    key=lambda e: e["timestamp"]
                )
                # Group events into actions and add participant/session IDs
                actions = group_events_from_module(raw_events, session_id, module_name)
                for action in actions:
                    action["processNumber"] = participant  # Add participant ID
                all_actions.extend(actions)
    
    return pd.DataFrame(all_actions)


def _collapse_duplicate_hovers(actions):
    """
    Collapse consecutive duplicate hover actions generated by legacy logic.
    Keeps the hover with the largest end/duration for each duplicate run.
    """
    cleaned = []
    i = 0
    removed = 0
    while i < len(actions):
        current = actions[i]
        if current.get("type") != "hover":
            cleaned.append(current)
            i += 1
            continue

        best = current
        j = i + 1
        while j < len(actions):
            nxt = actions[j]
            if nxt.get("type") != "hover":
                break
            same_anchor = (
                nxt.get("start") == current.get("start")
                and nxt.get("x") == current.get("x")
                and nxt.get("y") == current.get("y")
            )
            if same_anchor:
                if float(nxt.get("end", -1)) >= float(best.get("end", -1)):
                    best = nxt
                removed += 1
            else:
                break
            j += 1

        cleaned.append(best)
        i = j

    return cleaned, removed


def repair_hover_duplicates_in_json(input_json_path, output_json_path=None):
    """
    Repair hover duplication directly on action_sequences JSON.
    Useful when raw behavioral_data.json is not available.
    """
    if output_json_path is None:
        output_json_path = input_json_path

    with open(input_json_path, "r", encoding="utf-8") as f:
        sequence_dict = json.load(f)

    total_removed = 0
    hover_before = 0
    hover_after = 0

    for key, actions in sequence_dict.items():
        hover_before += sum(1 for a in actions if a.get("type") == "hover")
        cleaned, removed = _collapse_duplicate_hovers(actions)
        total_removed += removed
        hover_after += sum(1 for a in cleaned if a.get("type") == "hover")
        sequence_dict[key] = cleaned

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(sequence_dict, f, indent=2)

    print("\n=== Hover Repair Summary ===")
    print(f"Input JSON: {input_json_path}")
    print(f"Output JSON: {output_json_path}")
    print(f"Hover before: {hover_before}")
    print(f"Hover after: {hover_after}")
    print(f"Removed duplicated hover rows: {total_removed}")


# ====== Run this part ======
if __name__ == "__main__":
    INPUT_JSON = r"C:\Users\pc\Desktop\FEUP\Others\Summer Internship\Documentation\Dataset\Datasets\ULSTMAD\03-07-25\behavioral_data.json"
    df = extract_and_group_all_sessions(INPUT_JSON)
    print(df.head())
    print(df[df['type'] == 'hover'].head(10))
    print(df[df['type'] == 'click'].head(10))
    print(df[df['type'] == 'mouse_movement'].head(10))
    print(df[df['type'] == 'pause'].head(10))
    
    # Save as CSV for inspection or traditional ML
    # CSV without paths
    df.drop(columns=['path']).to_csv("actions_flat.csv", index=False)


    # Group into session-wise sequences for LSTM
    sequence_dict = {}
    grouped = df.groupby(['processNumber', 'sessionId',"module"])

    for (process_number, session_id, module_name), group in grouped:
        key = f"{process_number}|{session_id}|{module_name}"  # Include all grouping fields
        sequence_dict[key] = group.to_dict(orient='records')
    
    # Save sequences as JSON
    with open("action_sequences.json", "w", encoding='utf-8') as f:
        json.dump(sequence_dict, f, indent=2)
    
    print(f"\n=== Output Summary ===")
    print(f"Saved: {len(df)} actions → 'actions_flat.csv'")
    print(f"Saved: {len(sequence_dict)} sequences → 'action_sequences.json'")
