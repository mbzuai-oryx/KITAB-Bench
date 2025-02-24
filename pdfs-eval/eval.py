from argparse import ArgumentParser
from glob import glob 
import os
from tqdm import tqdm
import json

import datasets
from PIL import Image
import re

from models import GeminiOCR, DoclingPDF, MarkerPDF, GPT4oOCR, AVAILABLE_MODELS
from prompts import DEFAULT_PROMPT

RESULTS_DIR = "results"
MAX_TOKENS = 2048
os.makedirs(RESULTS_DIR, exist_ok=True)


def extract_html_tables_from_file(markdown_file_path: str):
    with open(markdown_file_path, 'r', encoding='utf-8') as file:
        markdown_text = file.read()
    table_regex = re.compile(r'<table[\s\S]*?</table>', re.IGNORECASE)
    tables = table_regex.findall(markdown_text)
    text_without_tables = table_regex.sub('', markdown_text)
    return text_without_tables, tables

def get_model(model_name: str, flash_attn: bool):
    if model_name == "gemini":
        return GeminiOCR()
    if model_name == "doclingeasyocr":
        return DoclingPDF(model_type="easyocr")
    if model_name == "doclingtesseract":
        return DoclingPDF(model_type="tesseract")
    if model_name == "marker":
        return MarkerPDF()
    if model_name == "gpt-4o":
        return GPT4oOCR(max_tokens=MAX_TOKENS, model_name="gpt-4")
    if model_name == "gpt-4o-mini":
        return GPT4oOCR(max_tokens=MAX_TOKENS, model_name="gpt-4o-mini")
    if model_name == "qwen2vl":
        return Qwen2VLOCR(max_tokens=MAX_TOKENS, model_name="Qwen/Qwen2-VL-7B-Instruct", use_flash_attn=flash_attn)
    if model_name == "qwen25vl":
        return Qwen25VLOCR(max_tokens=MAX_TOKENS, model_name="Qwen/Qwen2.5-VL-7B-Instruct", use_flash_attn=flash_attn)
    raise ValueError(f"Model {model_name} not found")

def main(args):
    output_path = f"{RESULTS_DIR}/{args.model_name}.json"
    files = glob(f"pdfs/*.pdf")
    data = []
    model = get_model(args.model_name, args.flash_attn)
    for idx, file_path in tqdm(enumerate(files), total=len(files), desc="Evaluating PDFs"):
        label_path = file_path.replace(".pdf", ".md").replace("pdfs/", "labels/")
        gt_text, gt_tables = extract_html_tables_from_file(label_path)
        pred_text, pred_tables = model(DEFAULT_PROMPT, file_path)
        gt = {"text": gt_text, "tables": gt_tables}
        pred = {"text": pred_text, "tables": pred_tables}
        data.append({"idx": idx, "gt": gt, "pred": pred})

    with open(output_path, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gemini", choices=AVAILABLE_MODELS)
    parser.add_argument("--flash_attn", default=False, action="store_true")
    args = parser.parse_args()
    main(args)