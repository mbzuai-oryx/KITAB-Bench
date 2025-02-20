from glob import glob 
import json
import re
from docling_model import DoclingPDF
from tqdm import tqdm
import os

def extract_html_tables_from_file(markdown_file_path):
    with open(markdown_file_path, 'r', encoding='utf-8') as file:
        markdown_text = file.read()
    
    table_regex = re.compile(r'<table[\s\S]*?</table>', re.IGNORECASE)
    tables = table_regex.findall(markdown_text)
    text_without_tables = table_regex.sub('', markdown_text)
    return text_without_tables, tables

files = glob(f"pdfs/*.pdf")

model = DoclingPDF()
data = []

model_name = "docling_tesseract"
out_dir = "results"
os.makedirs(out_dir, exist_ok=True)
output_path = f"{out_dir}/{model_name}.json"
for idx, file in tqdm(enumerate(files), total=len(files)):
    print(f"Working on file {file}")
    label = file.replace(".pdf", ".md").replace("pdfs", "labels")
    gt_text, gt_tables = extract_html_tables_from_file(label)
    try:
        pred_text, pred_tables = model(file)
    except Exception as e:
        print(f"Skipping sample {file} for {e}")
        pred_text, pred_tables = "", []
    data.append({
        "gt": {"text": gt_text, "tables": gt_tables},
        "pred": {"text": pred_text, "tables": pred_tables}
    })
with open(output_path, "w") as f:
    json.dump(data, f, indent=4, ensure_ascii=False)