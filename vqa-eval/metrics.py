import json
from glob import glob
from pprint import pprint
from argparse import ArgumentParser

def eval_mtvqa(gt, pred):
    gt = gt.lower().strip().replace(".", "")
    pred = pred.lower().strip().replace(".", "")
    return int(gt in pred)

def eval_mcq(gt: str, pred: str):
    gt = gt.strip()
    pred = pred.strip()
    if len(pred) > 0: pred = pred[0]
    arabic_to_english = {
        "أ": "A",
        "ب": "B",
        "ج": "C",
        "د": "D"
    }
    table = str.maketrans(arabic_to_english)
    gt = gt.translate(table).lower()
    pred = pred.translate(table).lower()
    return int(gt == pred)

def main(args):
    results = {}
    print(f"Getting results for {args.model_name}")
    files = glob(f"results/{args.model_name}_*.json")
    for file in files:
        ds_name = file.split("/")[-1].replace(f"{args.model_name}_", "").replace(".json", "")
        with open(file, "r") as f:
            data = json.load(f)
        tot_correct = 0
        tot = 0
        for d in data:
            gts = d['gt']
            for idx, pred in enumerate(d['pred']):
                tot += 1
                gt = gts[idx]
                tot_correct += eval_mtvqa(gt, pred) if "mtvqa" in ds_name else eval_mcq(gt, pred)
        results[ds_name] = {
                "correct": tot_correct,
                "total": tot,
                "accuracy": round(tot_correct * 100 / tot, 2)
            }
    pprint(results)
    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gemini")
    args = parser.parse_args()
    main(args)
    