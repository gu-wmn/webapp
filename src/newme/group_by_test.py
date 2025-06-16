import json
from collections import defaultdict, Counter

# Load JSON
with open("wmn_annotations.json") as f:
    data = json.load(f)

# Normalize excerpts (case-insensitive)
def norm(text):
    return text.strip().lower()

# === 1. Group by WMN ID ===
wmn_to_excerpts = defaultdict(list)

for entry in data:
    wmn_id = entry["wmn_id"]
    if not entry.get("labels"):
        continue
    for label in entry.get("labels"):
        if label.get("name") != "Trigger":
            continue
        excerpt = label.get("excerpt", "").strip()
        wmn_to_excerpts[wmn_id].append(excerpt)

# Count and deduplicate within each WMN
grouped_by_wmn = {}

for wmn_id, excerpts in wmn_to_excerpts.items():
    counter = Counter([norm(e) for e in excerpts])
    seen = set()
    final = []
    for excerpt in excerpts:
        normed = norm(excerpt)
        if normed in seen:
            continue
        seen.add(normed)
        count = counter[normed]
        if count > 1:
            final.append(f"{excerpt}({count})")
        else:
            final.append(excerpt)
    grouped_by_wmn[wmn_id] = final

# === 2. Group by Excerpt ===
excerpt_to_wmns = defaultdict(set)

for entry in data:
    wmn_id = entry["wmn_id"]
    if not entry.get("labels"):
        continue
    for label in entry.get("labels"):
        if label.get("name") != "Trigger":
            continue
        excerpt = label.get("excerpt", "").strip()
        excerpt_to_wmns[norm(excerpt)].add(wmn_id)

# Build sorted view
grouped_by_label = []

for normed_excerpt in sorted(excerpt_to_wmns.keys(), key=str.lower):
    wmns = sorted(excerpt_to_wmns[normed_excerpt])
    if len(wmns) > 1:
        display = f"{normed_excerpt} ({', '.join(wmns)})"
    else:
        display = f"{normed_excerpt} ({wmns[0]})"
    grouped_by_label.append(display)

# === OUTPUT ===

print("\n=== Grouped by WMN ID ===")
for wmn_id, excerpts in grouped_by_wmn.items():
    print(f"\n{wmn_id}:")
    for e in excerpts:
        print(f"  - {e}")

print("\n=== Grouped by Label (Excerpt) ===")
for e in grouped_by_label:
    print(f"  - {e}")
