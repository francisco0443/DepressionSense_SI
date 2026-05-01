import json
import math
import os
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup

EPS = 1e-6


def euclidean_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def angle_between(x1, y1, x2, y2):
    dot = x1 * x2 + y1 * y2
    norm1 = math.sqrt(x1**2 + y1**2)
    norm2 = math.sqrt(x2**2 + y2**2)
    if norm1 * norm2 == 0:
        return 0
    cos_theta = max(min(dot / (norm1 * norm2), 1), -1)
    return math.acos(cos_theta)

def count_self_intersections(xs, ys):
    def segments_intersect(a, b, c, d):
        def ccw(p1, p2, p3):
            return (p3[1]-p1[1]) * (p2[0]-p1[0]) > (p2[1]-p1[1]) * (p3[0]-p1[0])
        return ccw(a, c, d) != ccw(b, c, d) and ccw(a, b, c) != ccw(a, b, d)
    count = 0
    segments = list(zip(xs, ys))
    for i in range(len(segments)-1):
        for j in range(i+2, len(segments)-1):
            if segments_intersect(segments[i], segments[i+1], segments[j], segments[j+1]):
                count += 1
    return count

def get_dom_tag(html_str):
    if not html_str or not isinstance(html_str, str):
        return None
    soup = BeautifulSoup(html_str, "html.parser")
    return soup.find().name if soup.find() else None


def _safe_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def build_session_xy_stats(actions):
    xs, ys = [], []
    for action in actions:
        action_type = str(action.get("type", "")).lower()

        if action_type == "mouse_movement":
            traj = action.get("path", [])
            for pt in traj:
                if not isinstance(pt, (list, tuple)) or len(pt) < 2:
                    continue
                x = _safe_float(pt[0])
                y = _safe_float(pt[1])
                if x is None or y is None:
                    continue
                xs.append(x)
                ys.append(y)
        else:
            x = _safe_float(action.get("x"))
            y = _safe_float(action.get("y"))
            if x is None or y is None:
                continue
            xs.append(x)
            ys.append(y)

    if not xs or not ys:
        return {"x_center": 0.0, "y_center": 0.0, "x_scale": 1.0, "y_scale": 1.0}

    arr_x = np.asarray(xs, dtype=np.float64)
    arr_y = np.asarray(ys, dtype=np.float64)

    x_center = float(np.median(arr_x))
    y_center = float(np.median(arr_y))
    x_iqr = float(np.percentile(arr_x, 75) - np.percentile(arr_x, 25))
    y_iqr = float(np.percentile(arr_y, 75) - np.percentile(arr_y, 25))
    x_range = float(np.max(arr_x) - np.min(arr_x))
    y_range = float(np.max(arr_y) - np.min(arr_y))

    x_scale = x_iqr if x_iqr > EPS else (x_range if x_range > EPS else 1.0)
    y_scale = y_iqr if y_iqr > EPS else (y_range if y_range > EPS else 1.0)

    return {
        "x_center": x_center,
        "y_center": y_center,
        "x_scale": float(x_scale),
        "y_scale": float(y_scale),
    }


def normalize_xy(x_raw, y_raw, xy_stats):
    x = _safe_float(x_raw)
    y = _safe_float(y_raw)
    if x is None or y is None:
        return np.nan, np.nan
    x_n = (x - xy_stats["x_center"]) / xy_stats["x_scale"]
    y_n = (y - xy_stats["y_center"]) / xy_stats["y_scale"]
    return float(x_n), float(y_n)


def compute_features_for_movement(action, xy_stats):
    traj = action.get('path', [])
    if not traj or len(traj) < 2:
        return None

    xs_raw = []
    ys_raw = []
    times = []
    for pt in traj:
        if not isinstance(pt, (list, tuple)) or len(pt) < 3:
            continue
        x_raw = _safe_float(pt[0])
        y_raw = _safe_float(pt[1])
        t = _safe_float(pt[2])
        if x_raw is None or y_raw is None or t is None:
            continue
        xs_raw.append(float(x_raw))
        ys_raw.append(float(y_raw))
        times.append(float(t))

    if len(xs_raw) < 2:
        return None

    norm_points = [normalize_xy(x, y, xy_stats) for x, y in zip(xs_raw, ys_raw)]
    xs = [p[0] for p in norm_points]
    ys = [p[1] for p in norm_points]



    dt = [(t2 - t1) for t1, t2 in zip(times[:-1], times[1:])]
    distances = [euclidean_distance(x1, y1, x2, y2)
                 for x1, y1, x2, y2 in zip(xs[:-1], ys[:-1], xs[1:], ys[1:])]
    
    total_distance = sum(distances)
    total_time = times[-1] - times[0] if times[-1] > times[0] else 1

    speeds = [d / dt[i] if dt[i] > 0 else 0 for i, d in enumerate(distances)]
    accel = [s2 - s1 for s1, s2 in zip(speeds[:-1], speeds[1:])]

    angles = [
        angle_between(xs[i] - xs[i - 1], ys[i] - ys[i - 1], xs[i + 1] - xs[i], ys[i + 1] - ys[i])
        for i in range(1, len(xs) - 1)
    ]

    return {
        'cursor_speed': total_distance / total_time,
        'acceleration_mean': np.mean(accel) if accel else 0,
        'reciprocal_acceleration': 1/np.mean(accel) if accel and abs(np.mean(accel)) > 1e-6 else 0,
        'jitter': np.std(distances) if len(distances) > 1 else 0,
        'direction_changes': sum(1 for a in angles if a > math.pi / 4),
        'curvature_mean': np.mean(angles) if angles else 0,  # NEW
        'rate_of_curvature': sum(angles) / total_distance if total_distance != 0 else 0,
        'distance_travelled': total_distance,
        'movement_offset': euclidean_distance(xs[0], ys[0], xs[-1], ys[-1]),
        'straightness': euclidean_distance(xs[0], ys[0], xs[-1], ys[-1]) / total_distance if total_distance else 0,
        'self_intersections': count_self_intersections(xs, ys),
        'x': xs[-1],
        'y': ys[-1],
        'x_raw': xs_raw[-1],
        'y_raw': ys_raw[-1],
        'module_id': action.get('module', 'unknown'),
        'action_type': action.get('type', 'unknown'),
        'dom_element_type': get_dom_tag(action.get('node')),
        'timestamp': action.get('timestamp', times[-1]),
        'action_duration':total_time

    }

def compute_click_duration(action, xy_stats):
    x_raw = _safe_float(action.get('x'))
    y_raw = _safe_float(action.get('y'))
    x_norm, y_norm = normalize_xy(x_raw, y_raw, xy_stats)
    return {
        #'action_duration': action.get('end', 0) - action.get('start', 0),
        'module_id': action.get('module', 'unknown'),
        'action_type': action.get('type', 'click'),
        'dom_element_type': get_dom_tag(action.get('node')),
        'timestamp': action.get('timestamp', None),
        'x': x_norm,
        'y': y_norm,
        'x_raw': x_raw,
        'y_raw': y_raw,
    }

def another_actions(action, xy_stats):
    x_raw = _safe_float(action.get('x'))
    y_raw = _safe_float(action.get('y'))
    x_norm, y_norm = normalize_xy(x_raw, y_raw, xy_stats)
    return {
        'action_duration': action.get('end', 0) - action.get('start', 0),
        'module_id': action.get('module', 'unknown'),
        'action_type': action.get('type', ''),
        'timestamp': action.get('end', None),
        'x': x_norm,
        'y': y_norm,
        'x_raw': x_raw,
        'y_raw': y_raw,
        }      
        

def extract_lstm_features_from_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    rows = []
    for session_key, actions in data.items():  # Handle each session block
        xy_stats = build_session_xy_stats(actions)
        prev_timestamp = None
        for action in actions:
            action_type = action.get('type', '').lower()

            if action_type == 'mouse_movement':
                features = compute_features_for_movement(action, xy_stats)
            elif action_type == 'click':
                features = compute_click_duration(action, xy_stats)
            else:
                features = another_actions(action, xy_stats)

            if features:
                curr_timestamp = features.get('timestamp', None)
                if curr_timestamp is not None and prev_timestamp is not None:
                    features['time_since_last_action'] = curr_timestamp - prev_timestamp
                else:
                    features['time_since_last_action'] = 0
                prev_timestamp = curr_timestamp
                
                features.pop('timestamp', None)
                features['session_key'] = session_key
                rows.append(features)
                
    
    print(f"[INFO] Processed {len(rows)} actions into LSTM features.")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    input_path = r"C:\Users\franc\Desktop\2nd Year\Article Project\Feature Extraction\action_sequences.json"
    output_csv = os.path.splitext(input_path)[0] + "_lstm_features.csv"

    df = extract_lstm_features_from_json(input_path)
    print(df.head())
    df.to_csv(output_csv, sep=';', index=False, encoding='utf-8')
    print(f"[INFO] Saved LSTM-ready features to {output_csv}")
    
