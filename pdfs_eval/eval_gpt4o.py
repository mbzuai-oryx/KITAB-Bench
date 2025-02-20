import datasets
import os
from PIL import Image
from tqdm import tqdm
import json
from openai import OpenAI
import base64
import multiprocessing as mp
import io
from pdf2image import convert_from_path
from glob import glob 
import re

DEFAULT_PROMPT = """Extract the text from the document in Markdown format, and extract the tables in HTML format. 
Do not add style or anything, just the text. Do not ever generate tables in markdown format. Give me the output, nothing else."""
RESULTS_DIR = "results"
MAX_TOKENS = 2000
NUM_WORKERS = 32

def remove_html_comment(text):
    cleaned_text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
    return cleaned_text

def remove_tables(md_text: str) -> str:
    table_pattern = re.compile(r"(^\|.*\|$\n(^\|[-:]+\|$\n)?(\|.*\|$\n)*)", re.MULTILINE)
    cleaned_text = re.sub(table_pattern, '', md_text)
    return cleaned_text.strip()

def extract_html_tables(html):
    table_regex = re.compile(r'<table[\s\S]*?</table>', re.IGNORECASE)
    tables = table_regex.findall(html)
    return tables

os.makedirs(RESULTS_DIR, exist_ok=True)

def encode_image_bytes(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")

def _get_messages(image: Image, prompt=DEFAULT_PROMPT):
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    image_bytes = buf.read()
    image_base64 = encode_image_bytes(image_bytes)

    return [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                {"type": "text", "text": prompt}
            ]
        }
    ]

def get_answer(image):
    messages = _get_messages(image)
    client = OpenAI()
    result = client.chat.completions.create(messages=messages, model="gpt-4o", max_tokens=MAX_TOKENS)
    out_text = result.choices[0].message.content
    out_text = out_text.replace("```markdown", "").replace("```html", "").replace("```", "")
    print(out_text)
    return out_text

def extract_html_tables_from_file(markdown_file_path):
    with open(markdown_file_path, 'r', encoding='utf-8') as file:
        markdown_text = file.read()
    
    table_regex = re.compile(r'<table[\s\S]*?</table>', re.IGNORECASE)
    tables = table_regex.findall(markdown_text)
    text_without_tables = table_regex.sub('', markdown_text)
    return text_without_tables, tables

def process_sample(args):
    idx, file_path = args
    images = convert_from_path(file_path)
    label_path = file_path.replace(".pdf", ".md").replace("pdfs/", "labels/")
    gt_text, gt_tables = extract_html_tables_from_file(label_path)
    pred_str = ""
    for img in images:
        try:
            pred = get_answer(img)
            pred = remove_html_comment(pred)
        except Exception as e:
            print(f"Skipping sample {idx} due to error: {e}")
            pred = ""
        pred_str += pred
    pred_text, pred_tables = remove_tables(pred_str), extract_html_tables(pred_str)
    gt = {"text": gt_text, "tables": gt_tables}
    pred = {"text": pred_text, "tables": pred_tables}
    return {"idx": idx, "gt": gt, "pred": pred}
model_name = "gpt4o"
output_path = f"{RESULTS_DIR}/{model_name}.json"
files = glob(f"pdfs/*.pdf")
data = []

tasks = [(idx, sample) for idx, sample in enumerate(files)]
with mp.Pool(NUM_WORKERS) as pool:
    for result in tqdm(pool.imap(process_sample, tasks),
                        total=len(tasks),
                        desc=f"Evaluating PDFs ..."):
        data.append(result)
with open(output_path, "w") as f:
    json.dump(data, f, indent=4, ensure_ascii=False)