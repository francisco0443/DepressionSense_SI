
import json

with open('action_sequences.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Track any non-increasing timestamps
non_monotonic = []

for session_id, actions in data.items():
    prev_t = None
    for i, action in enumerate(actions):
        curr_t = action.get('timestamp')
        if curr_t is None:
            curr_t = action.get('end')  
        if prev_t is not None and curr_t < prev_t:
            non_monotonic.append((session_id, i, prev_t, curr_t))
        prev_t = curr_t

# Print results
if non_monotonic:
    print(f"⚠️ Found {len(non_monotonic)} non-monotonic timestamp(s):\n")
    for session_id, idx, prev, curr in non_monotonic:
        print(f"Session '{session_id}', action #{idx}:")
        print(f"  Previous timestamp: {prev}")
        print(f"  Current timestamp : {curr}")
        print(f"  Difference         : {curr - prev}\n")
else:
    print("✅ All action timestamps are in increasing order.")
