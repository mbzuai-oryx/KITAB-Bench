from argparse import ArgumentParser
import json
from glob import glob
from tqdm import tqdm
import os

from Levenshtein import distance as lev
from sacrebleu import corpus_bleu, corpus_chrf
import nltk
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
from nltk.translate import meteor_score
from torchmetrics.text import CharErrorRate, WordErrorRate
import re

def preprocess_arabic_text(text: str) -> str:
    # Remove newlines
    text = text.replace("\n", " ")
    text = text.replace("\t", " ")
    # Remove diacritics (tashkeel)
    text = re.sub(r'[\u064B-\u065F\u0670]', '', text)
    # Normalize alef variants to bare alef
    text = re.sub('[إأٱآا]', 'ا', text)
    # Normalize teh marbuta to heh
    text = text.replace('ة', 'ه')
    # Normalize alef maksura to yeh
    text = text.replace('ى', 'ي')
    # This regex matches one or more tatweel characters (ـ) or any whitespace sequence.
    # The lambda replaces whitespace sequences with a single space,
    # and removes tatweel characters by replacing them with an empty string.
    text = re.sub(r'(ـ+)|\s+', lambda m: ' ' if m.group(0).isspace() else '', text).strip()
    text = ' '.join(text.split())
    return text

def avg(l):
    return 0 if len(l) == 0 else round(sum(l) / len(l), 2)
 
def evaluate_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    references = []
    hypotheses = []
 
    metrics = {
        'edit_distance': [],
        'cer': [],
        'wer': [],
        'meteor': [],
    }
    cr = CharErrorRate()
    wr = WordErrorRate()
    for item in data:
        pred = item['pred']
        if pred is None: pred = ""
        gt = preprocess_arabic_text(item['gt']).strip()
        pred = preprocess_arabic_text(pred).strip()
        edit_dist = lev(gt, pred)
        metrics['edit_distance'].append(edit_dist)
 
        metrics['meteor'].append(
            meteor_score.single_meteor_score(
                gt.split(),
                pred.split(),
                preprocess=lambda x: x.lower()
            )
        )
        
        references.append(gt)
        hypotheses.append(pred)
 
    return {
        'blue': round(corpus_bleu(hypotheses, [references]).score, 2),
        'chrf': round(corpus_chrf(hypotheses, [references]).score, 2),
        'ed': round(avg(metrics['edit_distance']), 2),
        'cer': round(cr(hypotheses, references).item(), 2),
        'wer': round(wr(hypotheses, references).item(), 2),
        'meteor': round(avg(metrics['meteor']), 2),
    }
 
def main(args):
    print(f"Working with model: {args.model_name}")
    files = glob(f"results/{args.model_name}_*.json")
    data = []
    cer, wer, chrf = 0, 0, 0
    for file in tqdm(files):
        file_name = file.split("/")[-1]
        ds_name = file_name.replace(".json", "").split("_")[-1]
        results = evaluate_file(file) 
        results["dataset"] = ds_name
        data.append(results)
        cer += results['cer']
        wer += results['wer']
        chrf += results['chrf']
    cer /= len(data)
    wer /= len(data)
    chrf /= len(data)
    print(f"CER: {cer:.2f} | WER: {wer:.2f} | CHrF: {chrf:.2f}")
    os.makedirs("metrics", exist_ok=True)
    with open(f"metrics/{args.model_name}_metrics.json", "w") as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="easyocr")
    args = parser.parse_args()
    main(args)

