import json

with open("wmn_annotations.json") as f:
    data = json.load(f)

def span_overlaps(a, b):
    # Converts span dicts into sortable tuples
    a_start = (a["start_index"], a["start_offset"])
    a_end = (a["end_index"], a["end_offset"])
    b_start = (b["start_index"], b["start_offset"])
    b_end = (b["end_index"], b["end_offset"])
    # No overlap if one ends before the other starts
    return not (a_end <= b_start or b_end <= a_start)

overlaps_by_wmn = {}

for entry in data:
    if not entry['labels']: continue
    wmn_id = entry["wmn_id"]
    labels = entry.get("labels", [])

    overlaps = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            if span_overlaps(labels[i], labels[j]):
                overlaps.append((labels[i], labels[j]))

    if overlaps:
        overlaps_by_wmn[wmn_id] = overlaps

# === Output overlapping cases ===
print("\n=== Overlapping Label Spans by WMN ===")
for wmn_id, pairs in overlaps_by_wmn.items():
    print(f"\nWMN: {wmn_id}")
    for a, b in pairs:
        print("  Overlap:")
        print(f"    A: {a}")
        print(f"    B: {b}")
