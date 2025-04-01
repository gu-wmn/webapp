import json

with open('wmn_annotations.json') as file:
    data = json.load(file)

filenames = []

for seq in data['wmn_sequences']:
    if seq['corpus'] == 'bnc':
        if not seq['corpus_id'] in filenames:
            filenames.append(seq['corpus_id'])

for name in filenames:
    print(name)

print("Sum:", len(filenames))
