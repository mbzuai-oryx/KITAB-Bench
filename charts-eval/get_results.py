import json
# from eval import calculate_scrm, calculate_chartex
from eval_diagrams import compute_json_similarity, calculate_scrm
import matplotlib.pyplot as plt
from tqdm import tqdm
path = "results/ain7b_diagrams.json"

from concurrent.futures import ThreadPoolExecutor, TimeoutError

timeout_sec = 5  # Set your timeout in seconds

with open(path, "r") as f:
    data = json.load(f)
s = set()
J_vals = []
eth_vals = []
tot = 0
failed = 0
for idx, d in tqdm(enumerate(data), total=len(data)):
    gt = d['ground_truth']
    pred = d['predictions']
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(calculate_scrm, pred, gt)
        try:
            scrum, eth, J = future.result(timeout=timeout_sec)
        except Exception as e:
            scrum = 1.0
    tot += scrum
    # J_vals.append(min(40, J))
    # eth_vals.append(eth)
    # tot += calculate_chartex(pred, gt)

    # try:
    #     gt_data = eval(gt['data'])
    #     pred_data = eval(pred['data'])
    #     tot += calculate_scrm(gt_data, pred_data)
    # except:
    #     failed += 1
    #     pass
    # vals.append(min(5, round(eth, 1)))
    # vals.append(scrum)
    # s.add(scrum)

tot /= (len(data) - failed)
print(f"Score: {tot*100:.2f}")

plt.hist(J_vals)
plt.savefig("fig.png")
# print(s)
