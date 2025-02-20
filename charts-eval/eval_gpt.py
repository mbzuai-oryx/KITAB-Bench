import datasets
import os
from tqdm import tqdm
import json
import multiprocessing as mp
from gpt_model import GPTModel
import ast
from eval import TYPE_PROMPT, TOPIC_PROMPT, DATA_PROMPT


RESULTS_DIR = "results"
NUM_WORKERS = 32

os.makedirs(RESULTS_DIR, exist_ok=True)

model_name = "gpt4o"

def process_sample(args):
    idx, sample = args
    model = GPTModel()
    img = sample["image"]
    pred_type = ""
    pred_topic = ""
    pred_data = ""
    metadata = ast.literal_eval(sample['metadata'])
    type_chart = metadata.get('figure_type')
    gt = {
        'type': type_chart,
        'topic': sample['topic'],
        'data': sample['data']
    }
    try:
        pred_type = model(TYPE_PROMPT, img)
        pred_type = pred_type.replace("'", "")
    except Exception as e:
        print(f"Skipping sample {idx} type due to error: {e}")
    try:
        pred_topic = model(TOPIC_PROMPT, img)
    except Exception as e:
        print(f"Skipping sample {idx} type due to error: {e}")
    try:
        pred_data: str = model(DATA_PROMPT, img)
        pred_data = pred_data.replace("```data\n", "").replace("```", "")
    except Exception as e:
        print(f"Skipping sample {idx} type due to error: {e}")
    pred = {"type": pred_type, "topic": pred_topic, "data": pred_data}
    return {"index": idx, "ground_truth": gt, "predictions": pred}


ds = datasets.load_dataset("ahmedheakl/arocrbench_charts", split="train", num_proc=4)
output_path = f"{model_name}_diagrams.json"
data = []
tasks = [(idx, sample) for idx, sample in enumerate(ds)]
with mp.Pool(NUM_WORKERS) as pool:
    for result in tqdm(pool.imap(process_sample, tasks),
                        total=len(tasks),
                        desc=f"Evaluating charts ..."):
        data.append(result)
with open(output_path, "w") as f:
    json.dump(data, f, indent=4, ensure_ascii=False)