from argparse import ArgumentParser
import os
from tqdm import tqdm
import json

import re
from PIL import Image
import datasets

from models import GeminiOCR, GPT4oOCR, InterVL25OCR


VQA_PROMPT_TEMP = """You are given a question and some choices. Just answer with the choice letter. Nothing else.
Question: 
{question}
Choices: 
{choices}
Now give me only the choice letter."""
RESULTS_DIR = "results"
MAX_IMG_SIZE = 1024
MAX_TOKENS = 100
os.makedirs(RESULTS_DIR, exist_ok=True)

ds_ids = [
    "ahmedheakl/arocrbench_chartsvqa", 
    "ahmedheakl/arocrbench_diagramsvqa",
    "ahmedheakl/arocrbench_mtvqa", 
    "ahmedheakl/arocrbench_patdvqa",
]

def convert_arabic_choices(text):
    arabic_to_english = {
        "أ": "A",
        "ب": "B",
        "ج": "C",
        "د": "D"
    }
    pattern = r"(أ|ب|ج|د)\."
    converted_text = re.sub(pattern, lambda m: arabic_to_english[m.group(1)] + ".", text)
    return converted_text

def get_choices(choices):
    choices_str = ""
    for idx, c in enumerate(choices):
        choice_letter = chr(ord('A') + idx)
        choices_str += f"{choice_letter}. {c}\n"
    return choices_str

def get_prompt(sample, ds_id):
    prompts = []
    answers = []
    if "mtvqa" in ds_id:
        prompts = [sample['question'] + " Answer in just one word or a sentence. Try your best to find the answer in the image, if not, reply with 'غير مذكور'."]
        answers = [sample['answer']]
    elif "patdvqa" in ds_id:
        prompts = [convert_arabic_choices(sample['question']) + "Now give me only the choice letter."]
        answers = [sample['answer']]
    else:
        prompt = VQA_PROMPT_TEMP.format(question=sample['question'], choices=get_choices(sample['choices']))
        prompts = [prompt]
        answers = [sample['answer']]
    return prompts, answers

def resize(w, h):
    if w > h:
        aspect = h / w
        nw = min(MAX_IMG_SIZE, w)
        nh = min(h, int(aspect * nw))
    else:
        aspect = w / h
        nh = min(MAX_IMG_SIZE, h)
        nw = min(w, int(aspect * nh))
    return nw, nh

def get_model(model_name: str):
    if model_name == "internvl25":
        return InterVL25OCR(max_tokens=MAX_TOKENS)
    if model_name == "gemini":
        return GeminiOCR(max_tokens=MAX_TOKENS)
    if model_name == "gpt-4o":
        return GPT4oOCR(max_tokens=MAX_TOKENS, model_name="gpt-4")
    if model_name == "gpt-4o-mini":
        return GPT4oOCR(max_tokens=MAX_TOKENS, model_name="gpt-4o-mini")
    raise ValueError(f"Model {model_name} not found")

def main(args):
    model = get_model(args.model_name)
    for ds_id in tqdm(ds_ids):
        ds_name = ds_id.split("_")[-1]
        print(f"Evaluating {ds_name} ...")
        output_path = f"{RESULTS_DIR}/{args.model_name}_{ds_name}.json"
        ds = datasets.load_dataset(ds_id, split="train")
        data = []
        for idx, sample in tqdm(enumerate(ds), total=len(ds), desc=f"Evaluating {ds_name}"):
            img = sample['image']
            w, h = img.size
            nw, nh = resize(w, h)
            img = img.resize((nw, nh), resample=Image.LANCZOS)
            prompts, gts = get_prompt(sample, ds_id)
            preds = []
            for prompt_idx, prompt in enumerate(prompts):
                try:
                    pred = model(prompt, img)
                except Exception as e:
                    print(f"Skipping {idx}.{prompt_idx} for {e}")
                    pred = ""
                preds.append(pred)
            data.append({"idx": idx, "gt": gts, "pred": preds})
        with open(output_path, "w") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gemini")
    args = parser.parse_args()
    main(args)