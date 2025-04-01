from argparse import ArgumentParser
import os
from tqdm import tqdm
import json
import datasets
from models import EasyOCR, GeminiOCR, PaddleGPUOCR, SuryaOCR, TesseractOCR, GPT4oOCR, Qwen2VLOCR, Qwen25VLOCR, ArabicNougat, SmolDocling, QaariOCR, AzureOCR, AVAILABLE_MODELS

DEFAULT_PROMPT = "Extract the text in the image. Give me the final text, nothing else."
RESULTS_DIR = "results"
MAX_TOKENS = 500
os.makedirs(RESULTS_DIR, exist_ok=True)

ds_ids = [
    "ahmedheakl/arocrbench_patsocr", # answer
    "ahmedheakl/arocrbench_historicalbooks", # answer
    "ahmedheakl/arocrbench_khattparagraph", # answer
    "ahmedheakl/arocrbench_synthesizear",
    "ahmedheakl/arocrbench_historyar",
    "ahmedheakl/arocrbench_adab",
    "ahmedheakl/arocrbench_muharaf",
    "ahmedheakl/arocrbench_onlinekhatt",
    "ahmedheakl/arocrbench_khatt",
    "ahmedheakl/arocrbench_isippt",
    "ahmedheakl/arocrbench_arabicocr",
    "ahmedheakl/arocrbench_hindawi",
    "ahmedheakl/arocrbench_evarest",

]

def get_model(model_name: str, flash_attn: bool):
    if model_name == "easyocr":
        return EasyOCR(MAX_TOKENS)
    if model_name == "tesseract":
        return TesseractOCR()
    if model_name == "gemini":
        return GeminiOCR()
    if model_name == "paddle":
        return PaddleGPUOCR(MAX_TOKENS)
    if model_name == "surya":
        return SuryaOCR(MAX_TOKENS)
    if model_name == "gpt-4o":
        return GPT4oOCR(max_tokens=MAX_TOKENS, model_name="gpt-4")
    if model_name == "gpt-4o-mini":
        return GPT4oOCR(max_tokens=MAX_TOKENS, model_name="gpt-4o-mini")
    if model_name == "qwen2vl":
        return Qwen2VLOCR(max_tokens=MAX_TOKENS, model_name="Qwen/Qwen2-VL-7B-Instruct", use_flash_attn=flash_attn)
    if model_name == "qwen25vl":
        return Qwen25VLOCR(max_tokens=MAX_TOKENS, model_name="Qwen/Qwen2.5-VL-7B-Instruct", use_flash_attn=flash_attn)
    if model_name == "arabicnougat_small":
        return ArabicNougat(model_name="MohamedRashad/arabic-small-nougat")
    if model_name == "arabicnougat_base":
        return ArabicNougat(model_name="MohamedRashad/arabic-base-nougat")
    if model_name == "arabicnougat_large":
        return ArabicNougat(model_name="MohamedRashad/arabic-large-nougat")
    if model_name == "smoldocling":
        return SmolDocling()
    if model_name == "qaari":
        return QaariOCR(max_tokens=MAX_TOKENS, use_flash_attn=flash_attn)
    if model_name == "azure": 
        return AzureOCR()
    raise ValueError(f"Model {model_name} not found")


def main(args):
    model = get_model(args.model_name, args.flash_attn)
    for ds_id in tqdm(ds_ids):
        ds_name = ds_id.split("_")[-1]
        print(f"Evaluating {ds_name} ...")
        output_path = f"{RESULTS_DIR}/{args.model_name}_{ds_name}.json"
        ds = datasets.load_dataset(ds_id, split="train")
        data = []
        answer_name = "answer" if ds_id in ds_ids[:3] else "text"
        for idx, sample in tqdm(enumerate(ds), total=len(ds), desc=f"Evaluating {ds_name}"):
            img = sample["image"]
            try:
                pred = model(DEFAULT_PROMPT, img)
            except Exception as e:
                print(f"Skipping sample {idx} due to error: {e}")
                pred = ""
            gt = sample[answer_name]
            data.append({"idx": idx, "gt": gt, "pred": pred})
        with open(output_path, "w") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="easyocr", choices=AVAILABLE_MODELS)
    parser.add_argument("--flash_attn", default=False, action="store_true")
    args = parser.parse_args()
    main(args)