from argparse import ArgumentParser
import os
from tqdm import tqdm
import json

import datasets
from bs4 import BeautifulSoup

from models import DoclingOCR, GeminiOCR, GPT4oOCR, Qwen2VLOCR, Qwen25VLOCR, MarkerOCR, AVAILABLE_MODELS
from prompts import HTML_PROMPT, DF_PROMPT

RESULTS_DIR = "results"
MAX_TOKENS = 2000
DS_ID = "ahmedheakl/arocrbench_tables"
os.makedirs(RESULTS_DIR, exist_ok=True)

def get_table(html):
  soup = BeautifulSoup(html, "html.parser")
  return soup.find("table")

def get_type(meta):
    return eval(meta)['figure_type']


def get_model(model_name: str, flash_attn, **kwargs):
  if model_name == "doclingtesseract":
    return DoclingOCR(max_tokens=MAX_TOKENS, model_type="tesseract")
  if model_name == "doclingeasyocr":
    return DoclingOCR(max_tokens=MAX_TOKENS, model_type="easyocr")
  if model_name == "gemini":
    return GeminiOCR(max_tokens=MAX_TOKENS)
  if model_name == "img2tableeasyocr":
    return Img2tableOCR(max_tokens=MAX_TOKENS, model_type="easyocr")
  if model_name == "img2tabletesseract":
    return Img2tableOCR(max_tokens=MAX_TOKENS, model_type="easyocr")
  if model_name == "gpt-4o":
    return GPT4oOCR(max_tokens=MAX_TOKENS, model_name="gpt-4")
  if model_name == "gpt-4o-mini":
    return GPT4oOCR(max_tokens=MAX_TOKENS, model_name="gpt-4o-mini")
  if model_name == "qwen2vl":
    return Qwen2VLOCR(max_tokens=MAX_TOKENS, model_name="Qwen/Qwen2-VL-7B-Instruct", use_flash_attn=flash_attn)
  if model_name == "qwen25vl":
    return Qwen25VLOCR(max_tokens=MAX_TOKENS, model_name="Qwen/Qwen2.5-VL-7B-Instruct", use_flash_attn=flash_attn)
  if model_name == "marker":
    return MarkerOCR(max_tokens=MAX_TOKENS, is_html=kwargs.get("is_html"))
  raise ValueError(f"Model {model_name} not found")


def eval_ds(ds: datasets.Dataset, model, is_html: bool, output_path: str):
  data = []
  for idx, sample in tqdm(enumerate(ds), total=len(ds), desc=f"Evaluating tables HTML"):
    if is_html != ("HTML" in eval(sample['metadata'])["_pipeline"]): continue
    img = sample['image']
    try:
        prompt = HTML_PROMPT if is_html else DF_PROMPT
        pred = model(prompt, img)
        pred = str(get_table(pred))
        pred = pred.split("```html")[-1].split("```")[0]
        pred = pred.split("```csv")[-1].split("```")[0]
    except Exception as e:
        print(f"Skipping {idx} for {e}")
        pred = ""
    gt = str(get_table(sample["code"])) if is_html else sample['data']
    data.append({"idx": idx, "gt": gt, "pred": pred, "type": get_type(sample['metadata'])})
  with open(output_path, "w") as f:
    json.dump(data, f, indent=4, ensure_ascii=False)

def main(args):
  model = get_model(args.model_name, args.flash_attn, is_html=True)
  ds = datasets.load_dataset(DS_ID, split="train")
  output_path = f"{RESULTS_DIR}/{args.model_name}_html.json"
  eval_ds(ds, model, is_html=True, output_path=output_path)
  del model
  model = get_model(args.model_name, args.flash_attn, is_html=False)
  output_path = f"{RESULTS_DIR}/{args.model_name}_csv.json"
  eval_ds(ds, model, is_html=False, output_path=output_path)

if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument("--model_name", type=str, default="gemini", choices=AVAILABLE_MODELS)
  parser.add_argument("--flash_attn", default=False, action="store_true")
  parser.add_argument("--max_image_size", type=int, default=1024)
  args = parser.parse_args()
  main(args)