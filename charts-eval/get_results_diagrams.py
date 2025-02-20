import json
from eval_diagrams import calculate_scrm
import matplotlib.pyplot as plt
path = "results/ain7b_diagrams.json"

with open(path, "r") as f:
    data = json.load(f)
s = set()
vals = []
tot = 0
for i, d in enumerate(data):
    gt = d['ground_truth']
    pred = d['predictions']
    try:
        scrm, eth, J = calculate_scrm(pred, gt)
    except Exception as e:
        print(f"Data {i} was not evaluated")
        scrm = 0 
    tot += scrm
    # vals.append(round(J, 1))
    # vals.append(scrum)
    s.add(scrm)

print(f"SCRM: {tot / len(data)}")
