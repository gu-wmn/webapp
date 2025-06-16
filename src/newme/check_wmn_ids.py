#!/usr/bin/env python

import json

with open("./wmn_annotations.json") as f:
    annotations = json.load(f)

wmn_ids = {}

for wmn in annotations:
    if not wmn['wmn_id'] in wmn_ids.keys():
        wmn_ids[wmn['wmn_id']] = 1
    else:
        wmn_ids[wmn['wmn_id']] += 1

for wmn_id, wmn_sum in wmn_ids.items():
    print(wmn_id, wmn_sum)
    if wmn_sum > 1:
        print("more than one wmn_id.", wmn_id, wmn_sum)

print(len(wmn_ids.keys()))
