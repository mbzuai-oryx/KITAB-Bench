from argparse import ArgumentParser
import os
import json
from tqdm import tqdm
import datasets
from models import EasyOCR, SuryaOCR, TesseractOCR

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)
ds = datasets.load_dataset("ahmedheakl/arocrbench_ourslines", split="train")
data = []

def get_model(model_name):
    if model_name == "easyocr":
        return EasyOCR()
    elif model_name == "suryaocr":
        return SuryaOCR()
    elif model_name == "tesseract":
        return TesseractOCR()
    else:
        raise ValueError(f"Model {model_name} not found")

def main(args):
    ocr_model = get_model(args.model_name)
    output_path = f"{RESULTS_DIR}/{args.model_name}.json"
    for idx, sample in tqdm(enumerate(ds), total=len(ds), desc=f"Evaluating lines"):
        img = sample['image']
        try:
            pred = ocr_model(img)
        except Exception as e:
            print(f"Skipping {idx} for {e}")
            pred = ""
        gt = sample['data']
        data.append({"idx": idx, "gt": gt, "pred": pred})

    with open(output_path, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="easyocr", help="Model name")
    args = parser.parse_args()
    main(args)