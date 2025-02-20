import datasets
import json
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

RESULTS_DIR = "results"
NUM_WORKERS = 2

model_name = "surya"
output_path = f"{RESULTS_DIR}/{model_name}.json"
ds = datasets.load_dataset("ahmedheakl/arocrbench_ourslines", split="train")
with open(output_path, "r") as f:
    data = json.load(f)


def find_pred(idx):
    for d in data:
        if d['idx'] == idx:
            return d['pred']
    return None

idx = 0  
sample = ds[idx]
img: Image = sample['image']
pred = find_pred(idx)
gt = eval(sample['data'])
pred_lines = pred['lines']
gt_lines = [
    {'bbox': [b['bbox']['x1'], b['bbox']['y1'], b['bbox']['x2'], b['bbox']['y2']], 'text': b['text'], 'class': b['class']} 
    for b in gt]

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
    print("Post processing ...")
    to_merge = {}
    for gt_idx, gt_line in enumerate(gt_lines):
        gt_bbox = gt_line['bbox']
        for pred_idx, pred_line in enumerate(pred_lines):
            pred_bbox = pred_line['bbox']
            intersect_b = intersection(gt_bbox, pred_bbox)
            pred_area = area(pred_bbox)
            intersect_area = area(intersect_b)
            if intersect_area <= 1: continue
            if pred_area / intersect_area < th: continue
            if gt_idx not in to_merge:
                to_merge[gt_idx] = []
            to_merge[gt_idx].append({'bbox': pred_bbox, 'idx': pred_idx})
    to_pop = set()
    all_preds_to_merge = []
    for merges in to_merge.values():
        preds_to_merge = []
        for merge in merges: 
            pred_idx = merge['idx']
            pred = pred_lines[pred_idx]
            preds_to_merge.append(pred)
            to_pop.add(pred_idx)
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
            text = merged_pred['text'] + pred['text']
            bbox = merge_bboxes(bbox1, bbox2)
            merged_pred = {'bbox': bbox, 'text': text, 'conf': conf}
        pred_lines.append(merged_pred)
    to_pop = sorted(list(to_pop))[::-1]
    for pred_idx in to_pop:
        pred_lines.pop(pred_idx)
    return pred_lines



def get_iou(b1, b2):
    intersect = intersection(b1, b2)
    union = merge_bboxes(b1, b2)
    area_intersect = area(intersect)
    area_union = area(union)
    if area_union < 1: return 0
    return area_intersect / area_union

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
             
            
def render(img, lines, lines2=[], out="viz.png"):
    print(f"Rendering {out}")
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img)
    for idx, line in enumerate(lines):
        bbox = line["bbox"]
        rect = patches.Rectangle(
            (bbox[0], bbox[1]),  
            bbox[2] - bbox[0],   
            bbox[3] - bbox[1],   
            linewidth=3,
            edgecolor='blue',
            facecolor='none'
        )
        ax.add_patch(rect)
        ax.text(
            bbox[0], bbox[1] - 5, f"{line.get('index', str(idx))}",
            bbox=dict(facecolor='white', alpha=0.7),
            fontsize=8, color='black'
        )
    ax.axis("off")
    plt.savefig(out, bbox_inches='tight', dpi=300)
    plt.close()
pred_lines = post_process(pred_lines, gt_lines)
pred_lines = match_lines(pred_lines, gt_lines)
for line in pred_lines:
    if line['index'] < 0: continue
    pred_text = line['text']
    index = line['index']
    gt_text = gt_lines[index]['text']
    print(f"GT: {gt_text} | Pred: {pred_text}\n")
render(img, pred_lines, out="viz_pred.png")
render(img, gt_lines, out="viz_gt.png")