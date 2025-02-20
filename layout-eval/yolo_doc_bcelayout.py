import os
import numpy as np
from visualization_utils import visualize_boxes
import ast
import json
import torch
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
from collections import defaultdict
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from doclayout_yolo import YOLOv10
from metrics_utils import calculate_detailed_metrics

# BCE label mapping from original code
BCE_LABELS = {
    # High frequency text regions
    "paragraph__TextRegion": "Text",
    "heading__TextRegion": "Section-header",
    "caption__TextRegion": "figure_caption",
    "header__TextRegion": "Page-header",
    "page-number__TextRegion": "Page-footer",
    "floating__TextRegion": "Text",
    
    # Low frequency text regions (merged)
    "footer__TextRegion": "Footnote",
    "footnote__TextRegion": "Footnote",
    "drop-capital__TextRegion": "Text",
    "credit__TextRegion": "Text",
    
    # Charts and Images (merged)
    "line__ChartRegion": "Picture",
    "bar__ChartRegion": "Picture",
    "scatter__ChartRegion": "Picture",
    "pie__ChartRegion": "Picture",
    "surface__ChartRegion": "Picture",
    "line__ImageRegion": "Picture",
    
    # Other region types
    "TableRegion": "Table",
    "ImageRegion": "Picture",
    "FormulaRegion": "Formula",
    "ListRegion": "List-item",
    "HeaderRegion": "Page-header",
    "FooterRegion": "Page-footer",
    "FigureRegion": "Picture",
    "TableCaptionRegion": "table_caption",
    "MathRegion": "Formula",
    "GraphicRegion": "Picture",
    "TextRegion": "Text",
    "ChartRegion": "Picture"
}

# YOLOv10 label mapping
YOLO_LABELS = {
    'title': 0,
    'plain text': 1,
    'abandon': 2,
    'figure': 3,
    'figure_caption': 4,
    'table': 5,
    'table_caption': 6,
    'table_footnote': 7,
    'isolate_formula': 8,
    'formula_caption': 9
}

# Mapping from BCE labels to YOLO labels
BCE_TO_YOLO = {
    "Text": "plain text",
    "Section-header": "title",
    "figure_caption": "figure_caption",
    "table_caption": "table_caption",
    "Page-header": "plain text",
    "Page-footer": "abandon",
    "Footnote": "table_footnote",
    "Picture": "figure",
    "Table": "table",
    "Formula": "isolate_formula",
    "List-item": "plain text",
    "Title": "title"
}

# Create unified label mapping for evaluation
UNIFIED_LABELS = {
    "plain text": 0,
    "title": 1,
    "figure": 2,
    "figure_caption": 3,
    "table": 4,
    "table_caption": 5,
    "table_footnote": 6,
    "isolate_formula": 7,
    "formula_caption": 8
}

ID_TO_LABEL = {v: k for k, v in UNIFIED_LABELS.items()}

class YOLOPredictor:
    def __init__(self, model_path: str, device: str = "cuda:0"):
        self.model = YOLOv10(model_path)
        self.device = device
        self.image_size = 1024
        
    def predict(self, image):
        if isinstance(image, Image.Image):
            # Convert PIL image to numpy array
            image_np = np.array(image)
        else:
            image_np = image
            
        results = self.model.predict(
            image_np,
            imgsz=self.image_size,
            conf=0.2,
            device=self.device
        )[0]
        
        predictions = []
        for box in results.boxes:
            bbox = box.xyxy[0].cpu().numpy()  # Get bbox coordinates
            label_id = int(box.cls[0])
            confidence = float(box.conf[0])
            
            label_str = list(YOLO_LABELS.keys())[label_id]
            
            predictions.append({
                "bbox": bbox.tolist(),
                "label": label_str,
                "confidence": confidence
            })
            
        return predictions

def parse_bounding_boxes(bbox_str: str) -> list:
    """Parse string representation of bounding boxes to list of coordinates."""
    try:
        return ast.literal_eval(bbox_str)
    except:
        return []

def evaluate_model(model_path: str, use_all_data: bool=False, num_samples: int = 400):
    os.makedirs("preds_test", exist_ok=True)
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    predictor = YOLOPredictor(model_path, device)
    
    print("Loading BCE dataset...")
    dataset = load_dataset("ahmedheakl/arocrbench_bcelayout", split="train")
    if use_all_data:
        num_samples = len(dataset)
    dataset = dataset.select(range(num_samples))
    
    all_preds = []
    all_targets = []
    saved_predictions = []
    
    print("Processing images...")
    for idx, item in enumerate(tqdm(dataset)):
        image = item["image"].convert("RGB")
        width, height = image.size
        
        # Get predictions
        preds = predictor.predict(image)
        if not preds:
            continue
            
        # Format predictions
        boxes = []
        scores = []
        labels = []
        norm_predictions = []
        
        for pred in preds:
            # Normalize boxes to 0-1 range
            bbox = pred['bbox']
            norm_box = [
                bbox[0] / width,
                bbox[1] / height,
                bbox[2] / width,
                bbox[3] / height
            ]
            boxes.append(norm_box)
            scores.append(pred['confidence'])
            labels.append(UNIFIED_LABELS.get(pred['label'], 0))
            
            if len(norm_predictions) < 5:
                norm_predictions.append({
                    'bbox': bbox,
                    'label': pred['label'],
                    'confidence': pred['confidence']
                })
        
        pred_dict = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "scores": torch.tensor(scores, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64)
        }
        
        # Process ground truth
        gt_boxes = []
        gt_labels = []
        gt_vis = []
        
        region_types = ast.literal_eval(item["region_types"])
        text_types = ast.literal_eval(item["text_types"])
        bboxes = parse_bounding_boxes(item["normalized_bounding_boxes"])
        
        seen = set()
        for i, (bbox, r_type) in enumerate(zip(bboxes, region_types)):
            if len(bbox) >= 4:
                bbox_tuple = tuple(bbox[:4])
                if bbox_tuple not in seen:
                    seen.add(bbox_tuple)
                    
                    # Convert normalized coordinates to absolute
                    abs_bbox = [
                        bbox[0] * width,
                        bbox[1] * height,
                        bbox[2] * width,
                        bbox[3] * height
                    ]
                    
                    t_type = text_types[i] if i < len(text_types) else None
                    
                    # Map BCE label to YOLO label
                    if r_type == "TableRegion":
                        bce_label = "Table"
                    else:
                        combined_type = f"{t_type}__{r_type}" if t_type else r_type
                        bce_label = BCE_LABELS.get(combined_type, BCE_LABELS[r_type])
                    
                    yolo_label = BCE_TO_YOLO.get(bce_label, "plain text")
                    
                    gt_boxes.append(bbox[:4])
                    gt_labels.append(UNIFIED_LABELS.get(yolo_label, 0))
                    
                    if len(gt_vis) < 20:
                        gt_vis.append({
                            "bbox": abs_bbox,
                            "label": yolo_label
                        })
        
        target_dict = {
            "boxes": torch.tensor(gt_boxes, dtype=torch.float32),
            "labels": torch.tensor(gt_labels, dtype=torch.int64)
        }
        
        all_preds.append(pred_dict)
        all_targets.append(target_dict)
        
        if len(saved_predictions) < 20:
            saved_predictions.append({
                "image_idx": idx,
                "image": image,  # Store the PIL image for visualization
                "predictions": norm_predictions,
                "ground_truth": gt_vis
            })

    print("\nCalculating metrics...")
    metric = MeanAveragePrecision(box_format="xyxy")
    metric.update(all_preds, all_targets)
    results = metric.compute()

    # Calculate additional metrics
    per_class_metrics = defaultdict(lambda: defaultdict(list))
    all_macros = []
    
    for pred, target in zip(all_preds, all_targets):
        per_class, macro = calculate_detailed_metrics(pred, target)
        all_macros.append(macro)
        for label_id, metrics in per_class.items():
            class_name = ID_TO_LABEL[label_id]
            for k, v in metrics.items():
                per_class_metrics[class_name][k].append(v)
    
    # Average the metrics
    final_per_class = {}
    for class_name, metrics in per_class_metrics.items():
        final_per_class[class_name] = {
            k: sum(v) / len(v) for k, v in metrics.items()
        }
    
    # Calculate macro average
    macro_avg = {
        "precision": sum(m["precision"] for m in all_macros) / len(all_macros),
        "recall": sum(m["recall"] for m in all_macros) / len(all_macros),
        "f1": sum(m["f1"] for m in all_macros) / len(all_macros)
    }
    
    metrics = {
        "map_50": float(results["map_50"].item()),
        "map_75": float(results["map_75"].item()),
        "map": float(results["map"].item()),
        "processed_images": len(all_preds),
        "per_class_metrics": final_per_class,
        "macro_avg": macro_avg
    }
    print("Metrics ...")
    print(metrics)
    # Save prediction visualizations
    print("\nSaving predictions and visualizations...")
    for i, pred_data in enumerate(saved_predictions):
        save_dir = f"preds_test/bcelayout_yolo_doc"
        os.makedirs(save_dir, exist_ok=True)
        save_path = f"{save_dir}/yolo_pred_vis_{i}.png"
        image = dataset[pred_data["image_idx"]]["image"]
        visualize_boxes(
            image,
            pred_data["predictions"],
            pred_data["ground_truth"],
            save_path
        )
        
    # Save metrics
    with open("preds_test/bcelayout_detailed_results_yolo_doc.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    return metrics

if __name__ == "__main__":
    MODEL_PATH = "/l/users/mukul.ranjan/ocr/layout/doclayout_yolo_docstructbench_imgsz1024.pt"
    print("Model Path:", MODEL_PATH)
    metrics = evaluate_model(MODEL_PATH, use_all_data=True)
    print(f"\nResults:")
    print(f"Processed images: {metrics['processed_images']}")
    print(f"mAP@50: {metrics['map_50']:.4f}")
    print(f"mAP@75: {metrics['map_75']:.4f}")
    print(f"mAP@0.5:0.95: {metrics['map']:.4f}")
    print(f"Precision: {metrics['macro_avg']['precision']:.4f}")
    print(f"Recall: {metrics['macro_avg']['recall']:.4f}")
    print(f"F1: {metrics['macro_avg']['f1']:.4f}")