from collections import defaultdict
import json
import math

# === CONFIGURATION ===
FILENAME = r"C:\Users\pc\Desktop\FEUP\Others\Summer Internship\Documentation\Dataset\Datasets\ULSTMAD\03-07-25\behavioral_data.json"


def euclidean_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


# === LOAD JSON ===
with open(FILENAME, "r", encoding="utf-8") as f:
    sessions = json.load(f)


def count_sessions():
    return len(sessions)

def list_unique_participants():
    participants = set()
    for session in sessions:
        if "processNumber" in session:
            participants.add(session["processNumber"])
        # Also check inside modules if participant ID is nested there
        for module in session.values():
            if isinstance(module, dict) and "userInfo" in module:
                if "participantId" in module["userInfo"]:
                    participants.add(module["userInfo"]["participantId"])
    return participants

def participant_session_map():
    mapping = defaultdict(list)
    for session_idx, session in enumerate(sessions):
        participant = session.get("processNumber", "unknown")
        mapping[participant].append(session_idx)
    return dict(mapping)

def count_modules_per_participant():
    module_counts = defaultdict(set)
    for session in sessions:
        participant = session.get("processNumber", "unknown")
        for module_name in ['behavior_WAI_sr', 'behavior_EADS', 'behavior_RESS']:
            if module_name in session:
                module_counts[participant].add(module_name)
    return {k: len(v) for k, v in module_counts.items()}

def count_completed_questionnaires(sessions):
    completed_counts = defaultdict(int)
    for session in sessions:
        for module_name in ['behavior_WAI_sr', 'behavior_EADS', 'behavior_RESS']:
            if module_name in session:
                if "clicks" in session[module_name]:
                    completed_counts[module_name] += 1
    return completed_counts

def count_total_clickstreams(sessions):
    count = 0
    for session in sessions:
        for module_name in ['behavior_WAI_sr', 'behavior_EADS', 'behavior_RESS']:
            if module_name in session:
                if "clicks" in session[module_name]:
                    count += 1
    return count


# === Execute analysis ===
print("Total sessions:", count_sessions())
print("Unique participants:", len(list_unique_participants()))
print("Participant session map:", participant_session_map())
print("Modules per participant:", count_modules_per_participant())
print("Modules Count (by type):", count_completed_questionnaires(sessions))
print("Total individual clickstreams (questionnaires):", count_total_clickstreams(sessions))
