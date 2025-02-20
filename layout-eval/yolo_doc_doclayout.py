import os
import json
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from datasets import load_dataset
from collections import defaultdict

from doclayout_yolo import YOLOv10
from visualization_utils import visualize_boxes
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from metrics_utils import calculate_detailed_metrics

# Simplified categories
SIMPLIFIED_CATEGORIES = {
    "Text": 0,
    "Image": 1,
    "Table": 2,
    "Title": 3
}

# YOLO model outputs numbers 0-9 representing:
# 0: 'title'
# 1: 'plain text' 
# 2: 'abandon'
# 3: 'figure'
# 4: 'figure_caption'
# 5: 'table'
# 6: 'table_caption'
# 7: 'table_footnote'
# 8: 'isolate_formula'
# 9: 'formula_caption'
# Map YOLO labels to simplified categories

# DocLayNet categories (0-10):
# 0: "Caption"
# 1: "Footnote" 
# 2: "Formula"
# 3: "List-item"
# 4: "Page-footer"
# 5: "Page-header"
# 6: "Picture"
# 7: "Section-header"
# 8: "Table"
# 9: "Text"
# 10: "Title"
YOLO_TO_SIMPLE = {
    'title': "Title",
    'plain text': "Text",
    'abandon': "Text",
    'figure': "Image",
    'figure_caption': "Text",
    'table': "Table",
    'table_caption': "Text",
    'table_footnote': "Text",
    'isolate_formula': "Text",
    'formula_caption': "Text"
}

# Map DocLayNet categories to simplified categories
DOCLAYNET_TO_SIMPLE = {
    0: "Text",   # Caption
    1: "Text",   # Footnote
    2: "Text",   # Formula
    3: "Text",   # List-item
    4: "Text",   # Page-footer
    5: "Text",   # Page-header
    6: "Image",  # Picture
    7: "Text",   # Section-header
    8: "Table",  # Table
    9: "Text",   # Text
    10: "Title"  # Title
}

class YOLOPredictor:
    def __init__(self, model_path: str, device: str = "cuda:0"):
        self.model = YOLOv10(model_path)
        self.device = device
        self.image_size = 1024
        
    def predict(self, image):
        if isinstance(image, Image.Image):
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
            bbox = box.xyxy[0].cpu().numpy()
            label_id = int(box.cls[0])
            confidence = float(box.conf[0])
            
            # Get YOLO label and map to simplified category
            yolo_label = list(YOLO_TO_SIMPLE.keys())[label_id]
            simple_category = YOLO_TO_SIMPLE[yolo_label]
            category_id = SIMPLIFIED_CATEGORIES[simple_category]
            
            predictions.append({
                "bbox": bbox.tolist(),
                "label": simple_category,  # Use simplified category name
                "category": category_id,   # Use simplified category ID
                "confidence": confidence
            })
            
        return predictions

def evaluate_model(model_path: str, use_all_data: bool = False, num_samples: int = 400):
    os.makedirs("preds_test", exist_ok=True)
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    predictor = YOLOPredictor(model_path, device)
    
    print("Loading DocLayNet dataset...")
    dataset = load_dataset("ahmedheakl/arocrbench_doclaynetv3", split="train")
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
            bbox = pred['bbox']
            norm_box = [
                bbox[0] / width,
                bbox[1] / height,
                bbox[2] / width,
                bbox[3] / height
            ]
            boxes.append(norm_box)
            scores.append(pred['confidence'])
            labels.append(pred['category'])
            
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
        
        # Format ground truth
        gt_boxes = []
        gt_labels = []
        seen = set()
        gt_vis = []
        
        for bbox, category in zip(item["bboxes_block"], item["categories"]):
            bbox_tuple = tuple(bbox)
            if bbox_tuple not in seen:
                seen.add(bbox_tuple)
                
                # Map DocLayNet category to simplified category
                simple_category = DOCLAYNET_TO_SIMPLE[category]
                simple_category_id = SIMPLIFIED_CATEGORIES[simple_category]
                
                gt_boxes.append(bbox)
                gt_labels.append(simple_category_id)
                
                if len(gt_vis) < 15:
                    pixel_bbox = [
                        bbox[0] * width,
                        bbox[1] * height,
                        bbox[2] * width,
                        bbox[3] * height
                    ]
                    gt_vis.append({
                        "bbox": pixel_bbox,
                        "label": simple_category  # Use simplified category name
                    })
        
        target_dict = {
            "boxes": torch.tensor(gt_boxes, dtype=torch.float32),
            "labels": torch.tensor(gt_labels, dtype=torch.int64)
        }
        
        all_preds.append(pred_dict)
        all_targets.append(target_dict)
        
        if len(saved_predictions) < 15:
            saved_predictions.append({
                "image_idx": idx,
                "image": image,
                "predictions": norm_predictions,
                "ground_truth": gt_vis
            })

    print("\nSaving predictions and visualizations...")
    for i, pred_data in enumerate(saved_predictions):
        save_dir = f"preds_test/doclaynet_yolo_doc"
        os.makedirs(save_dir, exist_ok=True)
        save_path = f"{save_dir}/simple_vis_{i}.png"
        visualize_boxes(
            pred_data["image"],
            pred_data["predictions"],
            pred_data["ground_truth"],
            save_path
        )
        pred_data.pop("image")
    
    with open("preds_test/predictions_simple.json", "w") as f:
        json.dump(saved_predictions, f, indent=2)
    
    print("Calculating metrics...")
    metric = MeanAveragePrecision(box_format="xyxy")
    metric.update(all_preds, all_targets)
    results = metric.compute()

    # Calculate per-class metrics using simplified categories
    per_class_metrics = defaultdict(lambda: defaultdict(list))
    all_macros = []
    
    for pred, target in zip(all_preds, all_targets):
        per_class, macro = calculate_detailed_metrics(pred, target)
        all_macros.append(macro)
        for label_id, metrics in per_class.items():
            class_name = list(SIMPLIFIED_CATEGORIES.keys())[label_id]
            for k, v in metrics.items():
                per_class_metrics[class_name][k].append(v)
    
    final_per_class = {}
    for class_name, metrics in per_class_metrics.items():
        final_per_class[class_name] = {
            k: sum(v) / len(v) for k, v in metrics.items()
        }
    
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
    
    with open("preds_test/simple_results_yolo_doc.json", "w") as f:
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