from argparse import ArgumentParser
import json
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from tqdm import tqdm

import matplotlib.pyplot as plt
from chartex_metric import calculate_chartex
from scrm_metric import calculate_scrm

timeout_sec = 5  

def main(args):
    path = f"results/{args.model_name}.json"
    with open(path, "r") as f:
        data = json.load(f)

    tot_scrm, tot_chartex = 0, 0
    for idx, d in tqdm(enumerate(data), total=len(data)):
        gt, pred = d['ground_truth'], d['predictions']
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(calculate_scrm, pred, gt)
            try:
                scrum_score = future.result(timeout=timeout_sec)
            except Exception as e:
                scrum_score = 1.0
        tot_scrm += scrum_score
        tot_chartex += calculate_chartex(pred, gt)
    tot_scrm = tot_scrm * 20 / len(data) # scale to percentage, multiply by 100 / 5 
    tot_chartex /= len(data)
    print(f"SCRM Score: {tot_scrm:.2f}%")
    print(f"ChartEx Score: {tot_chartex:.2f}%")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gpt-4o-mini")
    args = parser.parse_args()
    main(args)
