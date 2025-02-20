from argparse import ArgumentParser
import json
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm
from torchmetrics.text import CharErrorRate, WordErrorRate
from Levenshtein import distance as lev
from sacrebleu import corpus_chrf
import nltk
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
from nltk.translate import meteor_score
import re


def preprocess_arabic_text(text: str) -> str:
    # Remove newlines
    text = text.replace("\n", " ")
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

def area(b):
    x1, y1, x2, y2 = b
    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    return w * h

def intersection(bbox1, bbox2):
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    return [x1, y1, x2, y2]

def merge_bboxes(bbox1, bbox2):
    x1 = min(bbox1[0], bbox2[0])
    y1 = min(bbox1[1], bbox2[1])
    x2 = max(bbox1[2], bbox2[2])
    y2 = max(bbox1[3], bbox2[3])
    return [x1, y1, x2, y2]

def post_process(pred_lines, gt_lines, th=0.95):
    to_merge = {}
    to_pop = []
    for gt_idx, gt_line in enumerate(gt_lines):
        gt_bbox = gt_line['bbox']
        gt_class = gt_line['class']
        for pred_idx, pred_line in enumerate(pred_lines):
            pred_bbox = pred_line['bbox']
            intersect_b = intersection(gt_bbox, pred_bbox)
            pred_area = area(pred_bbox)
            intersect_area = area(intersect_b)
            if intersect_area <= 1: continue
            if pred_area / intersect_area < th: continue
            # if gt_class == "table": to_pop.append(pred_idx); continue
            if gt_idx not in to_merge:
                to_merge[gt_idx] = []
            to_merge[gt_idx].append({'bbox': pred_bbox, 'idx': pred_idx})
    all_preds_to_merge = []
    for merges in to_merge.values():
        preds_to_merge = []
        merges = sorted(merges, key=lambda x: x['bbox'][-2], reverse=True)
        for merge in merges: 
            pred_idx = merge['idx']
            pred = pred_lines[pred_idx]
            preds_to_merge.append(pred)
            to_pop.append(pred_idx)
        all_preds_to_merge.append(preds_to_merge)
        merged_pred = preds_to_merge[0]
        for idx, pred in enumerate(preds_to_merge):
            if idx == 0: continue
            bbox1 = merged_pred['bbox']
            bbox2 = pred['bbox']
            area1 = area(bbox1)
            area2 = area(bbox2)
            tot_area = area1 + area2
            w1, w2 = area1 / tot_area, area2 / tot_area
            conf1, conf2 = merged_pred['conf'], pred['conf']
            conf = w1 * conf1 + w2 * conf2
            text = merged_pred['text'] + " " + pred['text']
            bbox = merge_bboxes(bbox1, bbox2)
            merged_pred = {'bbox': bbox, 'text': text, 'conf': conf}
        pred_lines.append(merged_pred)
    to_pop = sorted(to_pop)[::-1]
    for pred_idx in to_pop:
        try: pred_lines.pop(pred_idx)
        except: continue
    return pred_lines


def get_iou(b1, b2):
    intersect = intersection(b1, b2)
    union = merge_bboxes(b1, b2)
    area_intersect = area(intersect)
    area_union = area(union)
    if area_union < 1: return 0
    return area_intersect / area_union

def avg(l):
    return 0 if len(l) == 0 else round(sum(l) / len(l), 2)


def recognition_results(pred_lines, gt_lines):
    references = []
    hypotheses = []
    cr = CharErrorRate()
    wr = WordErrorRate()
    metrics = {"ed": [], "meteor": []}
    for pred_line in pred_lines:
        index = pred_line['index']
        if index < 0: continue
        pred = pred_line['text']
        gt = gt_lines[index]['text']
        gt = preprocess_arabic_text(gt).strip()
        pred = preprocess_arabic_text(pred).strip()
        metrics['ed'].append(lev(gt, pred))
        metrics['meteor'].append(meteor_score.single_meteor_score(
                gt.split(),
                pred.split(),
                preprocess=lambda x: x.lower()
        ))
        references.append(gt)
        hypotheses.append(pred)
        
    return {
        # 'blue': round(corpus_bleu(hypotheses, [references]).score, 2),
        'chrf': round(corpus_chrf(hypotheses, [references]).score, 2),
        'ed': round(avg(metrics['ed']), 2),
        'cer': round(cr(hypotheses, references).item(), 2),
        'wer': round(wr(hypotheses, references).item(), 2),
        'meteor': round(avg(metrics['meteor']), 2),
    }



def match_lines(pred_lines, gt_lines):
    for gt_idx, gt_line in enumerate(gt_lines):
        gt_bbox = gt_line['bbox']
        best_idx, best_iou = 0, -100
        for pred_idx, pred_line in enumerate(pred_lines):
            pred_bbox = pred_line['bbox']
            iou = get_iou(pred_bbox, gt_bbox)
            if iou > best_iou: 
                best_idx, best_iou = pred_idx, iou
        pred_lines[best_idx]['index'] = gt_idx
    for pred_idx, pred_line in enumerate(pred_lines):
        if 'index' not in pred_line: pred_lines[pred_idx]['index'] = -1
    return pred_lines

def map_eval(data):
    coco_preds = []
    coco_targets = []
    recog_metrics = []
    for idx, d in tqdm(enumerate(data), total=len(data), desc="Post processing"):
        if d['idx'] != 0: continue
        pred_lines = d['pred']['lines']
        gt = eval(d['gt'])
        gt_lines = [{'bbox': [b['bbox']['x1'], b['bbox']['y1'], b['bbox']['x2'], b['bbox']['y2']], 'text': b['text'], 'class': b['class']} for b in gt]
        pred_lines = post_process(pred_lines, gt_lines)
        pred_lines = match_lines(pred_lines, gt_lines)
        recog_metrics.append(recognition_results(pred_lines, gt_lines))
        data[idx]['pred']['lines'] = pred_lines
    for d in tqdm(data, desc="Evaluating"):
        pred, gt = d['pred'], eval(d['gt'])
        boxes = []
        scores = []
        labels = []
        
        for box in pred["lines"]:
            boxes.append(box["bbox"])
            scores.append(box["conf"])
            labels.append(0)
        
        if boxes:
            coco_preds.append({
                "boxes": torch.tensor(boxes, dtype=torch.float32),
                "scores": torch.tensor(scores, dtype=torch.float32),
                "labels": torch.tensor(labels, dtype=torch.int64)
            })
        else:
            coco_preds.append({
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "scores": torch.zeros(0, dtype=torch.float32),
                "labels": torch.zeros(0, dtype=torch.int64)
            })
        
        gt_boxes = []
        gt_labels = []
        
        for box in gt:
            b = box['bbox']
            gt_boxes.append([b['x1'], b['y1'], b['x2'], b['y2']])
            gt_labels.append(0)
        
        if gt_boxes:
            coco_targets.append({
                "boxes": torch.tensor(gt_boxes, dtype=torch.float32),
                "labels": torch.tensor(gt_labels, dtype=torch.int64)
            })
        else:
            coco_targets.append({
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros(0, dtype=torch.int64)
            })
    
    metric = MeanAveragePrecision(box_format="xyxy")
    metric.update(coco_preds, coco_targets)
    map_results = metric.compute()
    keys = list(recog_metrics[0].keys())
    averages_rec = {}
    for res in recog_metrics:
        for k, v in res.items():
            averages_rec[k] = averages_rec.get(k, 0) + v
    for k, v in averages_rec.items(): 
        res[k] = v / len(recog_metrics)
    return {
        "mAP_metrics": {
            "mAP_50": float(map_results["map_50"].item()),
            "mAP_75": float(map_results["map_75"].item()),
            "mAP@0.5:0.95": float(map_results["map"].item())
        },
        "recognition": averages_rec
    }

def main(args):
    
    path = f"results/{args.model_name}.json"
    with open(path, "r") as f:
        data = json.load(f)
    results = map_eval(data)
    for metric, value in results["mAP_metrics"].items():
        print(f"{metric}: {value*100:.2f}")
    print(results['recognition'])

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="easyocr", help="Model name")
    args = parser.parse_args()
    main(args)