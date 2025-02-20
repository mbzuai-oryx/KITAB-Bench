import os
from datasets import load_dataset, Dataset
from surya.layout import LayoutPredictor
from PIL import Image
import io
import numpy as np
from tqdm import tqdm
import json
from typing import List, Dict, Any, Tuple
import ast
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision

# Fixed label mapping for BCE
BCE_LABELS = {
    "TextRegion": "Text",
    "TableRegion": "Table",
    "ImageRegion": "Picture",
    "FormulaRegion": "Formula",
    "ListRegion": "ListItem",
    "HeaderRegion": "PageHeader",
    "FooterRegion": "PageFooter",
    "FigureRegion": "Figure",
    "TableCaptionRegion": "Caption",
    "MathRegion": "Formula",
    "ChartRegion": "Picture",
    "GraphicRegion": "Picture"
}

def normalize_bbox(bbox: List[float], width: int, height: int) -> List[float]:
    """Normalize bounding box coordinates by image dimensions."""
    return [
        bbox[0] / width,  # x1
        bbox[1] / height,  # y1
        bbox[2] / width,  # x2
        bbox[3] / height  # y2
    ]

def normalize_polygon(polygon: List[List[float]], width: int, height: int) -> List[List[float]]:
    """Normalize polygon coordinates by image dimensions."""
    return [[x / width, y / height] for x, y in polygon]

def convert_to_coco_format(predictions: List[Dict], ground_truth: List[Dict], label_to_id: Dict[str, int]) -> Tuple[List[Dict], List[Dict]]:
    """Convert predictions and ground truth to COCO format for torchmetrics."""
    coco_preds = []
    coco_targets = []
    
    for idx, (pred, gt) in enumerate(zip(predictions, ground_truth)):
        # Format predictions
        boxes = []
        scores = []
        labels = []
        
        for box in pred["bboxes"]:
            # Map predicted label to ground truth label set
            pred_label = box["label"]
            if pred_label in label_to_id:
                boxes.append(box["bbox"])
                scores.append(box.get("confidence", 1.0))
                labels.append(label_to_id[pred_label])
        
        if boxes:  # Only add if there are predictions
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
            if box["label"] in label_to_id:
                gt_boxes.append(box["bbox"])
                gt_labels.append(label_to_id[box["label"]])
        
        if gt_boxes:  # Only add if there are ground truth boxes
            coco_targets.append({
                "boxes": torch.tensor(gt_boxes, dtype=torch.float32),
                "labels": torch.tensor(gt_labels, dtype=torch.int64)
            })
        else:
            coco_targets.append({
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros(0, dtype=torch.int64)
            })
    
    return coco_preds, coco_targets

def parse_bounding_boxes(bbox_str: str) -> List[List[float]]:
    """Parse string representation of bounding boxes to list of coordinates."""
    try:
        return ast.literal_eval(bbox_str)
    except:
        return []

def process_dataset(dataset: Dataset, dataset_type: str = "doclaynet", max_samples: int = None) -> Dict[str, Any]:
    """Process a dataset and evaluate Surya's layout detection."""
    print(f"\nProcessing dataset of type: {dataset_type}")
    print(f"Dataset size: {len(dataset)}")
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    layout_predictor = LayoutPredictor()
    predictions = []
    ground_truth = []
    saved_data = []
    
    # Use fixed label set based on dataset type
    # if dataset_type == "doclaynet":
    #     label_to_id = {label: idx for idx, label in DOCLAYNET_LABELS.items()}
    # else:  # bcelayout
    #     # Create unique id for each unique BCE label
    unique_labels = set(BCE_LABELS.values())
    label_to_id = {label: idx for idx, label in enumerate(sorted(unique_labels))}
    
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
                normalized_bbox = normalize_bbox(box.bbox, width, height)
                normalized_polygon = normalize_polygon(box.polygon, width, height)
                
                pred_boxes.append({
                    "bbox": normalized_bbox,
                    "polygon": normalized_polygon,
                    "label": box.label,
                    "confidence": box.confidence,
                    "top_k": box.top_k
                })
            gt_boxes = []
            region_types = ast.literal_eval(item["region_types"])
            bboxes = parse_bounding_boxes(item["normalized_bounding_boxes"])
            text_types = ast.literal_eval(item["text_types"])

            for bbox, r_type in zip(bboxes, region_types):
                if len(bbox) >= 4:
                    bbox_coords = bbox if len(bbox) == 4 else [bbox[0], bbox[1], bbox[2], bbox[3]]
                    label = BCE_LABELS[r_type]
                    gt_boxes.append({
                        "bbox": bbox_coords,
                        "label": label
                    })
            
            # Store predictions and ground truth for evaluation
            predictions.append({"bboxes": pred_boxes})
            ground_truth.append({"boxes": gt_boxes})
            
            # Store both for saving
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
    save_file = f"surya_results_{dataset_type}.json"
    with open(save_file, 'w') as f:
        json.dump(saved_data, f, indent=2)
    print(f"Saved predictions and ground truth to {save_file}")
    
    from metrics_utils import calculate_class_metrics
    detailed_metrics, macro_avg = calculate_class_metrics(predictions, ground_truth)
    # Convert to COCO format
    coco_preds, coco_targets = convert_to_coco_format(predictions, ground_truth, label_to_id)
    
    # Calculate metrics
    metric = MeanAveragePrecision(box_format="xyxy")
    metric.update(coco_preds, coco_targets)
    results = metric.compute()
    
    return {
        "dataset_type": dataset_type,
        "label_mapping": label_to_id,
        "per_class_metrics": detailed_metrics,
        "macro_avg": macro_avg,
        "metrics": {
            "mAP_50": float(results["map_50"].item()),
            "mAP_75": float(results["map_75"].item()),
            "mAP@0.5:0.95": float(results["map"].item()),
            "mAP_small": float(results["map_small"].item()),
            "mAP_medium": float(results["map_medium"].item()),
            "mAP_large": float(results["map_large"].item()),
            "mar_1": float(results["mar_1"].item()),
            "mar_10": float(results["mar_10"].item()),
            "mar_100": float(results["mar_100"].item()),
            "mar_small": float(results["mar_small"].item()),
            "mar_medium": float(results["mar_medium"].item()),
            "mar_large": float(results["mar_large"].item()),
        }
    }

def main():
    results = {}
    
    try:
        max_samples = 200 
        use_full_sample = True
        bcelayout = load_dataset("ahmedheakl/arocrbench_bcelayout", split="train")
        len_bcelayout = len(bcelayout)
        if use_full_sample:
            max_samples = len_bcelayout
        result = process_dataset(bcelayout, dataset_type="bcelayout", max_samples=max_samples)
        results["bcelayout"] = result
    except Exception as e:
        print(f"Error processing BCE Layout: {str(e)}")
    
    # Save results
    output_path = "layout_evaluation_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved evaluation results to {output_path}")
    
    # Print summary
    print("\nEvaluation Results Summary:")
    for dataset_type, result in results.items():
        print(f"\n{dataset_type}:")
        print(f"Label mapping: {result['label_mapping']}")
        print("\nMetrics:")
        for metric_name, value in result["metrics"].items():
            print(f"  {metric_name}: {value:.4f}")

        print("\nMacro Average:")
        for metric_name, value in result["macro_avg"].items():
            print(f"  {metric_name}: {value:.4f}")


if __name__ == "__main__":
    main()