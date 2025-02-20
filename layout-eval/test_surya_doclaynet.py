import os
from datasets import load_dataset
from surya.layout import LayoutPredictor
from PIL import Image
import numpy as np
from tqdm import tqdm
import json
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from collections import defaultdict

# DocLayNet label mapping
DOCLAYNET_LABELS = {
    0: "Caption",
    1: "Footnote", 
    2: "Formula",
    3: "List-item",
    4: "Page-footer",
    5: "Page-header",
    6: "Picture",
    7: "Section-header", 
    8: "Table",
    9: "Text",
    10: "Title"
}

# Map Surya labels to DocLayNet labels
SURYA_TO_DOCLAYNET = {
    "Page-header": "Page-header",
    "Page-footer": "Page-footer", 
    "Section-header": "Section-header",
    "ListItem": "List-item",
    "Text": "Text",
    "Table": "Table",
    "Picture": "Picture",
    "Figure": "Picture",
    "Formula": "Formula",
    "Caption": "Caption",
    "Code": "Text",
    "TextInlineMath": "Formula",
    "Equation": "Formula",
    "Title": "Title",
    "Handwriting": "Text",
    "Form": "Text",
    "Footnote": "Footnote",
    "Table-of-contents": "Text",
    "Text-inline-math": "Formula",
    "PageHeader": "Page-header",
    "PageFooter": "Page-footer", 
    "SectionHeader": "Section-header",
    "TableOfContents": "Text"
}
def convert_gt_coco_to_xyxy(bbox):
    """Convert ground truth from COCO [x,y,w,h] to [x1,y1,x2,y2]"""
    x, y, w, h = bbox
    return [x, y, x + w, y + h]

def normalize_prediction(bbox, width, height):
    """Normalize Surya's predictions from pixel coordinates"""
    return [
        bbox[0] / width,
        bbox[1] / height,
        bbox[2] / width,
        bbox[3] / height
    ]

def evaluate_doclaynet(dataset, max_samples=None):
    """Evaluate Surya on DocLayNet dataset"""
    print(f"\nProcessing DocLayNet dataset")
    print(f"Dataset size: {len(dataset)}")
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    layout_predictor = LayoutPredictor()
    predictions = []
    ground_truth = []
    saved_data = []
    
    for idx, item in enumerate(tqdm(dataset)):
        try:
            image = item["image"]
            width = image.width
            height = image.height
            
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            # Get Surya predictions
            pred_result = layout_predictor([image])[0]
            
            # Process predictions
            pred_boxes = []
            for box in pred_result.bboxes:
                # Normalize coordinates
                normalized_bbox = normalize_prediction(box.bbox, width, height)
                
                # Map Surya label to DocLayNet format
                pred_label = SURYA_TO_DOCLAYNET.get(box.label, box.label)
                
                pred_boxes.append({
                    "bbox": normalized_bbox,
                    "label": pred_label,
                    "confidence": box.confidence
                })
            
            # Process ground truth
            gt_boxes_dict = {}
            for bbox, category in zip(item["bboxes_block"], item["categories"]):
                if len(bbox) >= 4:
                    
                    bbox_xyxy = bbox
                    bbox_key = tuple(bbox_xyxy)
                    
                    # Get label
                    label = DOCLAYNET_LABELS[category]
                    
                    # Store unique boxes
                    if bbox_key not in gt_boxes_dict:
                        gt_boxes_dict[bbox_key] = {
                            "bbox": list(bbox_xyxy),
                            "label": label
                        }
            
            gt_boxes = list(gt_boxes_dict.values())
            
            # Store for evaluation
            predictions.append({"boxes": pred_boxes})
            ground_truth.append({"boxes": gt_boxes})
            
            # Save data
            saved_data.append({
                "idx": idx,
                "predictions": pred_boxes,
                "ground_truth": gt_boxes,
                "image_size": {"width": width, "height": height}
            })
            
        except Exception as e:
            print(f"Error processing sample {idx}: {str(e)}")
            continue
    
    # Save predictions and ground truth
    os.makedirs("preds_test/surya_layout", exist_ok=True)
    with open("preds_test/surya_layout/surya_doclaynet_prediction_sample.json", "w") as f:
        json.dump(saved_data[:5], f, indent=2)
    
    # Calculate per-class metrics
    class_metrics = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    iou_threshold = 0.5
    
    for pred, gt in zip(predictions, ground_truth):
        for gt_box in gt["boxes"]:
            label = gt_box["label"]
            matched = False
            
            for pred_box in pred["boxes"]:
                if pred_box["label"] == label:
                    bbox1 = gt_box["bbox"]
                    bbox2 = pred_box["bbox"]
                    
                    # Calculate IoU
                    x1 = max(bbox1[0], bbox2[0])
                    y1 = max(bbox1[1], bbox2[1])
                    x2 = min(bbox1[2], bbox2[2])
                    y2 = min(bbox1[3], bbox2[3])
                    
                    intersection = max(0, x2 - x1) * max(0, y2 - y1)
                    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
                    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
                    union = area1 + area2 - intersection
                    iou = intersection / union if union > 0 else 0
                    
                    if iou >= iou_threshold:
                        class_metrics[label]["tp"] += 1
                        matched = True
                        break
            
            if not matched:
                class_metrics[label]["fn"] += 1
        
        # Count false positives
        for pred_box in pred["boxes"]:
            label = pred_box["label"]
            matched = False
            
            for gt_box in gt["boxes"]:
                if gt_box["label"] == label:
                    bbox1 = gt_box["bbox"]
                    bbox2 = pred_box["bbox"]
                    
                    x1 = max(bbox1[0], bbox2[0])
                    y1 = max(bbox1[1], bbox2[1])
                    x2 = min(bbox1[2], bbox2[2])
                    y2 = min(bbox1[3], bbox2[3])
                    
                    intersection = max(0, x2 - x1) * max(0, y2 - y1)
                    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
                    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
                    union = area1 + area2 - intersection
                    iou = intersection / union if union > 0 else 0
                    
                    if iou >= iou_threshold:
                        matched = True
                        break
            
            if not matched:
                class_metrics[label]["fp"] += 1
    
    # Calculate metrics for each class
    results = {}
    macro_avg = {"precision": 0, "recall": 0, "f1": 0}
    num_classes = 0
    
    for label, metrics in class_metrics.items():
        tp = metrics["tp"]
        fp = metrics["fp"]
        fn = metrics["fn"]
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        results[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": tp + fn,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn
        }
        
        macro_avg["precision"] += precision
        macro_avg["recall"] += recall
        macro_avg["f1"] += f1
        num_classes += 1
    
    # Calculate macro average
    if num_classes > 0:
        for key in macro_avg:
            macro_avg[key] /= num_classes
    
    # Calculate mAP using torchmetrics
    label_to_id = {label: idx for idx, label in enumerate(sorted(set(DOCLAYNET_LABELS.values())))}
    
    # Convert to COCO format
    coco_preds = []
    coco_targets = []
    
    for pred, gt in zip(predictions, ground_truth):
        # Format predictions
        boxes = []
        scores = []
        labels = []
        
        for box in pred["boxes"]:
            boxes.append(box["bbox"])
            scores.append(box["confidence"])
            labels.append(label_to_id[box["label"]])
        
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
        
        # Format ground truth
        gt_boxes = []
        gt_labels = []
        
        for box in gt["boxes"]:
            gt_boxes.append(box["bbox"])
            gt_labels.append(label_to_id[box["label"]])
        
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
    
    return {
        "per_class": results,
        "macro_avg": macro_avg,
        "mAP_metrics": {
            "mAP_50": float(map_results["map_50"].item()),
            "mAP_75": float(map_results["map_75"].item()),
            "mAP@0.5:0.95": float(map_results["map"].item())
        }
    }

def main():
    # Load DocLayNet dataset
    dataset = load_dataset("ahmedheakl/arocrbench_doclaynetv3", split="train")
    len_dataset = len(dataset)
    use_full_dataset = False
    max_samples=400
    if use_full_dataset:
        max_samples = len_dataset
    print(f"Dataset size: {len_dataset}")

    # Evaluate
    results = evaluate_doclaynet(dataset, max_samples=max_samples)
    
    # Save results
    with open("doclaynet_evaluation.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Print results
    print("\nDocLayNet Evaluation Results:")
    print("\nPer-class Metrics:")
    for label, metrics in results["per_class"].items():
        print(f"\n{label}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    print("\nMacro Average:")
    for metric, value in results["macro_avg"].items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nMean Average Precision:")
    for metric, value in results["mAP_metrics"].items():
        print(f"  {metric}: {value:.4f}")

if __name__ == "__main__":
    main()