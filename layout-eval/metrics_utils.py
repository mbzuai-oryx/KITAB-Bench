from collections import defaultdict
from collections import defaultdict
from typing import Dict, List, Tuple

def calculate_iou(box1, box2):
    """Calculate IoU between two boxes"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0

def calculate_detailed_metrics(predictions, ground_truth, iou_threshold=0.5):
    """Calculate detailed metrics including precision, recall, and F1 per class"""
    class_metrics = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    
    # Count true positives and false negatives
    for gt_box, gt_label in zip(ground_truth["boxes"], ground_truth["labels"]):
        matched = False
        for pred_box, pred_label in zip(predictions["boxes"], predictions["labels"]):
            if gt_label == pred_label and calculate_iou(gt_box, pred_box) >= iou_threshold:
                class_metrics[gt_label.item()]["tp"] += 1
                matched = True
                break
        if not matched:
            class_metrics[gt_label.item()]["fn"] += 1
    
    # Count false positives
    for pred_box, pred_label in zip(predictions["boxes"], predictions["labels"]):
        matched = False
        for gt_box, gt_label in zip(ground_truth["boxes"], ground_truth["labels"]):
            if pred_label == gt_label and calculate_iou(pred_box, gt_box) >= iou_threshold:
                matched = True
                break
        if not matched:
            class_metrics[pred_label.item()]["fp"] += 1
    
    # Calculate per-class metrics
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
            "support": tp + fn
        }
        
        macro_avg["precision"] += precision
        macro_avg["recall"] += recall
        macro_avg["f1"] += f1
        num_classes += 1
    
    if num_classes > 0:
        for key in macro_avg:
            macro_avg[key] /= num_classes
    
    return results, macro_avg


def calculate_class_metrics(predictions: List[Dict], ground_truth: List[Dict], iou_threshold: float = 0.5) -> Dict:
    """Calculate per-class precision, recall, and F1 scores"""
    class_metrics = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    
    for gt_boxes, pred_boxes in zip(ground_truth, predictions):
        # Track matched predictions to avoid double counting
        matched_preds = set()
        
        # Count true positives and false negatives
        for gt_box in gt_boxes['boxes']:
            label = gt_box['label']
            matched = False
            
            for i, pred_box in enumerate(pred_boxes['bboxes']):
                if i in matched_preds:
                    continue
                    
                if pred_box['label'] == label and calculate_iou(gt_box['bbox'], pred_box['bbox']) >= iou_threshold:
                    class_metrics[label]['tp'] += 1
                    matched_preds.add(i)
                    matched = True
                    break
                    
            if not matched:
                class_metrics[label]['fn'] += 1
        
        # Count false positives
        for i, pred_box in enumerate(pred_boxes['bboxes']):
            if i not in matched_preds:
                label = pred_box['label']
                class_metrics[label]['fp'] += 1
    
    # Calculate metrics for each class
    results = {}
    macro_avg = {"precision": 0, "recall": 0, "f1": 0}
    num_classes = 0
    
    for label, metrics in class_metrics.items():
        tp = metrics['tp']
        fp = metrics['fp']
        fn = metrics['fn']
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        results[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": tp + fn,
            "tp": tp,
            "fp": fp,
            "fn": fn
        }
        
        macro_avg["precision"] += precision
        macro_avg["recall"] += recall
        macro_avg["f1"] += f1
        num_classes += 1
    
    if num_classes > 0:
        for key in macro_avg:
            macro_avg[key] /= num_classes
            
    return results, macro_avg