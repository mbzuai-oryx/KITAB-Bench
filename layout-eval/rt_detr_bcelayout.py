import os
import ast
import json
import torch
from PIL import Image
from tqdm import tqdm

from datasets import load_dataset
from collections import defaultdict
from visualization_utils import visualize_boxes
from huggingface_hub import snapshot_download
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from metrics_utils import calculate_detailed_metrics

from dotenv import load_dotenv
load_dotenv()


BCE_LABELS = {
    # High frequency text regions
    "paragraph__TextRegion": "Text",
    "heading__TextRegion": "Section-header",
    "caption__TextRegion": "Caption",
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
    "TableCaptionRegion": "Caption",
    "MathRegion": "Formula",
    "GraphicRegion": "Picture",
    "TextRegion": "Text",
    "ChartRegion": "Picture"
}

# # BCE label mapping
# BCE_LABELS = {
#     "TextRegion": "Text",
#     "TableRegion": "Table",
#     "ImageRegion": "Figure",
#     "FormulaRegion": "Formula",
#     "ListRegion": "List-item",
#     "HeaderRegion": "Page-header",
#     "FooterRegion": "Page-footer",
#     "FigureRegion": "Figure",
#     "TableCaptionRegion": "Caption",
#     "MathRegion": "Formula",
#     "ChartRegion": "Figure",
#     "GraphicRegion": "Figure"
# }
# {
#     "0": "background",
#     "1": "Caption",
#     "10": "Text",
#     "11": "Title",
#     "12": "Document Index",
#     "13": "Code",
#     "14": "Checkbox-Selected",
#     "15": "Checkbox-Unselected",
#     "16": "Form",
#     "17": "Key-Value Region",
#     "2": "Footnote",
#     "3": "Formula",
#     "4": "List-item",
#     "5": "Page-footer",
#     "6": "Page-header",
#     "7": "Picture",
#     "8": "Section-header",
#     "9": "Table"
#   },

# Create unified label mapping for model
UNIFIED_LABELS = {
    "Caption": 0, "Footnote": 1, "Formula": 2, "List-item": 3,
    "Page-footer": 4, "Page-header": 5, "Picture": 6,
    "Section-header": 7, "Table": 8, "Text": 9, "Title": 10
}

ID_TO_LABEL = {v: k for k, v in UNIFIED_LABELS.items()}

black_classes = ["Document Index", "Code", "Form", "Checkbox-Selected", "Checkbox-Unselected", "Key-Value Region"]

def get_model_path(force_download=False):
    """Get path to model, downloading only if necessary"""
    model_path = snapshot_download(
        repo_id="ds4sd/docling-models",
        revision="v2.1.0",
        force_download=force_download
    )
    model_path = f"{model_path}/model_artifacts/layout"
    print(f"Using model from: {model_path}")
    return model_path

class LayoutPredictor:
    def __init__(self, model_path: str, device: str = "cpu", threshold: float = 0.3):
        self.black_classes = black_classes
        self.threshold = threshold
        self.device = torch.device(device)
        self.image_size = 640
        
        # Load model and processor
        self.image_processor = RTDetrImageProcessor.from_pretrained(model_path)
        self.model = RTDetrForObjectDetection.from_pretrained(
            model_path,
            ignore_mismatched_sizes=True
        ).to(self.device)
        self.model.eval()
        
        # Load class mapping from config
        with open(f"{model_path}/config.json") as f:
            config = json.load(f)
        self.classes_map = {int(k): v for k, v in config['id2label'].items()}

    def predict(self, image):
        if isinstance(image, Image.Image):
            image = image.convert("RGB")
        else:
            image = Image.fromarray(image).convert("RGB")
            
        inputs = self.image_processor(
            images=image,
            return_tensors="pt",
            size={"height": self.image_size, "width": self.image_size},
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        results = self.image_processor.post_process_object_detection(
            outputs,
            target_sizes=torch.tensor([image.size[::-1]]),
            threshold=self.threshold,
        )[0]
        
        predictions = []
        w, h = image.size
        for score, label_id, box in zip(results["scores"], results["labels"], results["boxes"]):
            score = float(score.item())
            label_id = int(label_id.item()) + 1
            label_str = self.classes_map[label_id]

            if label_str in self.black_classes:
                continue

            bbox = [float(b.item()) for b in box]
            
            # Clip boxes to image boundaries
            l = min(w, max(0, bbox[0]))
            t = min(h, max(0, bbox[1]))
            r = min(w, max(0, bbox[2]))
            b = min(h, max(0, bbox[3]))
            
            predictions.append({
                "bbox": [l, t, r, b],
                "label": label_str,
                "confidence": score
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
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = LayoutPredictor(model_path, device)
    
    print("Loading BCE dataset...")
    dataset = load_dataset("ahmedheakl/arocrbench_bcelayout", split="train")
    if use_all_data:
        num_samples=len(dataset)
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
            labels.append(UNIFIED_LABELS.get(pred['label'], 0))  # Default to 0 if label not found
            
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
        
        # Process ground truth from BCE format
        gt_boxes = []
        gt_labels = []
        gt_vis = []
        
        region_types = ast.literal_eval(item["region_types"])
        text_types = ast.literal_eval(item["text_types"])
        bboxes = parse_bounding_boxes(item["normalized_bounding_boxes"])
        
        # # Debug print lengths
        # if idx < 2:  # Only print for first 2 images
        #     print(f"\nDebug info for image {idx}")
        #     print(f"Number of region_types: {len(region_types)}")
        #     print(f"Number of text_types: {len(text_types)}")
        #     print(f"Number of bboxes: {len(bboxes)}")
        #     print("Region types:", region_types)
        #     print("Text types:", text_types)
        
        seen = set()
        # Use the length of region_types and handle text_types separately
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
                    
                    # Get text type if available, else None
                    t_type = text_types[i] if i < len(text_types) else None
                    
                    # Handle label mapping
                    if r_type == "TableRegion":
                        bce_label = "Table"
                    else:
                        combined_type = f"{t_type}__{r_type}" if t_type else r_type
                        bce_label = BCE_LABELS.get(combined_type, BCE_LABELS[r_type])
                    
                    # if idx < 2 and r_type == "TableRegion":  # Debug print for tables
                    #     print(f"\nFound table in image {idx}:")
                    #     print(f"bbox: {bbox}")
                    #     print(f"bce_label: {bce_label}")
                    #     print(f"unified_label: {UNIFIED_LABELS.get(bce_label, 0)}")
                    
                    gt_boxes.append(bbox[:4])
                    gt_labels.append(UNIFIED_LABELS.get(bce_label, 0))
                    
                    if len(gt_vis) < 20:
                        gt_vis.append({
                            "bbox": abs_bbox,
                            "label": bce_label
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
                "image": image,
                "predictions": norm_predictions,
                "ground_truth": gt_vis
            })

    print("\nSaving predictions and visualizations...")
    for i, pred_data in enumerate(saved_predictions):
        save_dir = "preds_test/rt_detr_bce"
        os.makedirs(save_dir, exist_ok=True)
        save_path = f"{save_dir}/pred_vis_{i}.png"
        visualize_boxes(
            pred_data["image"],
            pred_data["predictions"],
            pred_data["ground_truth"],
            save_path
        )
        pred_data.pop("image")
    
    with open("preds_test/rt_detr_bce_predictions.json", "w") as f:
        json.dump(saved_predictions, f, indent=2)
    
    print("Calculating mAP...")
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
    
    # Average the metrics across all images
    final_per_class = {}
    for class_name, metrics in per_class_metrics.items():
        final_per_class[class_name] = {
            k: sum(v) / len(v) for k, v in metrics.items()
        }
    
    # Calculate final macro average
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
    
    with open("preds_test/bcelayout_detailed_results_rt_detr.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    return metrics

if __name__ == "__main__":
    MODEL_PATH = get_model_path()
    print("Model Path:", MODEL_PATH)
    metrics = evaluate_model(MODEL_PATH, use_all_data=False, num_samples=10)
    print(f"\nResults:")
    print(f"Processed images: {metrics['processed_images']}")
    print(f"mAP@50: {metrics['map_50']:.4f}")
    print(f"mAP@75: {metrics['map_75']:.4f}")
    print(f"mAP@0.5:0.95: {metrics['map']:.4f}")
    print(f"Precision: {metrics['macro_avg']['precision']:.4f}")
    print(f"Recall: {metrics['macro_avg']['recall']:.4f}")
    print(f"F1: {metrics['macro_avg']['f1']:.4f}")