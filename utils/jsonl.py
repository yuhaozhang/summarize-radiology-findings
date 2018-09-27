"""
Use jsonl in place of json format.
"""
import json

def load(infile):
    data = []
    for line in infile:
        line = line.strip()
        if len(line) == 0:
            continue
        d = json.loads(line)
        data += [d]
    return data

def dump(data, outfile):
    for d in data:
        print(json.dumps(d), outfile)
    return

