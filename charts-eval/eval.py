from argparse import ArgumentParser
import datasets
import os
from tqdm import tqdm
import json
from models import GPT4oOCR, Qwen2VLOCR, Qwen25VLOCR, GeminiOCR, AVAILABLE_MODELS
import ast
from prompts import TYPE_PROMPT, TOPIC_PROMPT, DATA_PROMPT


RESULTS_DIR = "results"
MAX_TOKENS = 2000
os.makedirs(RESULTS_DIR, exist_ok=True)
ds = datasets.load_dataset("ahmedheakl/arocrbench_charts", split="train[:4]", num_proc=4)

data = []

def process_sample(idx, sample, model):
    img = sample["image"]
    pred_type, pred_topic, pred_data = "", "", ""
    try:
        pred_type = model(TYPE_PROMPT, img)
        pred_type = pred_type.replace("'", "")
    except Exception as e:
        print(f"Skipping sample {idx} type due to error: {e}")
    try:
        pred_topic = model(TOPIC_PROMPT, img)
    except Exception as e:
        print(f"Skipping sample {idx} topic due to error: {e}")
    try:
        pred_data: str = model(DATA_PROMPT, img)
        pred_data = pred_data.split("```csv\n")[-1].split("```")[0]
    except Exception as e:
        print(f"Skipping sample {idx} data due to error: {e}")
    return {"type": pred_type, "topic": pred_topic, "data": pred_data}

def get_model(model_name: str, flash_attn, **kwargs):
  if model_name == "gpt-4o":
    return GPT4oOCR(max_tokens=MAX_TOKENS, model_name="gpt-4")
  if model_name == "gpt-4o-mini":
    return GPT4oOCR(max_tokens=MAX_TOKENS, model_name="gpt-4o-mini")
  if model_name == "qwen2vl":
    return Qwen2VLOCR(max_tokens=MAX_TOKENS, model_name="Qwen/Qwen2-VL-7B-Instruct", use_flash_attn=flash_attn)
  if model_name == "qwen25vl":
    return Qwen25VLOCR(max_tokens=MAX_TOKENS, model_name="Qwen/Qwen2.5-VL-7B-Instruct", use_flash_attn=flash_attn)
  if model_name == "gemini":
    return GeminiOCR()
  raise ValueError(f"Model {model_name} not found")


def main(args):
    model = get_model(args.model_name, args.flash_attn)
    output_path = f"{RESULTS_DIR}/{args.model_name}.json"
    for idx, sample in tqdm(enumerate(ds), total=len(ds), desc="Evaluating charts"):
        type_chart = ast.literal_eval(sample['metadata']).get('figure_type')
        gt = {'type': type_chart, 'topic': sample['topic'], 'data': sample['data']}
        pred = process_sample(idx, sample, model)
        data.append({"index": idx, "ground_truth": gt, "predictions": pred})
        
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument("--model_name", type=str, default="gpt-4o-mini", choices=AVAILABLE_MODELS)
  parser.add_argument("--flash_attn", default=False, action="store_true")
  args = parser.parse_args()
  main(args)