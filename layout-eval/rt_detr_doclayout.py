import os
import json
import torch
import random
from tqdm import tqdm
from PIL import Image
from datasets import load_dataset
from collections import defaultdict

from visualization_utils import visualize_boxes
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from huggingface_hub import snapshot_download
from metrics_utils import calculate_detailed_metrics

# DocLayNet label mapping
DOCLAYNET_LABELS = {
    0: "Caption", 1: "Footnote", 2: "Formula", 3: "List-item",
    4: "Page-footer", 5: "Page-header", 6: "Picture", 
    7: "Section-header", 8: "Table", 9: "Text", 10: "Title"
}

# Create label mapping
label_mapping = {
    "Caption": 0, "Footnote": 1, "Formula": 2, "List-item": 3,
    "Page-footer": 4, "Page-header": 5, "Picture": 6,
    "Section-header": 7, "Table": 8, "Text": 9, "Title": 10
}

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
            # if label_id == 0:  # Skip background class
            #     continue
            score = float(score.item())
            label_id = int(label_id.item()) + 1  # Advance the label_i
            label_str = self.classes_map[label_id]

            # Filter out blacklisted classes
            if label_str in self.black_classes:
                continue

            # label = self.classes_map[int(label_id.item())]
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

def evaluate_model(model_path: str, use_all_data: bool = False, num_samples: int = 400):
    # Create output directory
    os.makedirs("preds_test", exist_ok=True)
    
    # Initialize model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = LayoutPredictor(model_path, device)
    
    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset("ahmedheakl/arocrbench_doclaynetv3", split="train")
    if use_all_data:
        num_samples = len(dataset)
    dataset = dataset.select(range(num_samples))
    
    # Process images and calculate mAP
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
            labels.append(label_mapping[pred['label']])
            
            # Save original format predictions for visualization
            if len(norm_predictions) < 5:
                norm_predictions.append({
                    'bbox': bbox,  # Keep original coordinates for visualization
                    'label': pred['label'],
                    'confidence': pred['confidence']
                })
        
        # Add predictions
        pred_dict = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "scores": torch.tensor(scores, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64)
        }
        
        # Format ground truth
        gt_boxes = []
        gt_labels = []
        seen = set()
        gt_vis = []  # Ground truth for visualization
        
        
        unique_blocks = set(tuple(bbox) for bbox in item['bboxes_block'])
        num_unique_blocks = len(unique_blocks)
        # print(f"processing total of {num_unique_blocks} unique blocks..")
        # print(unique_blocks)
        # print("\n\n")
        for bbox, category in zip(item["bboxes_block"], item["categories"]):
            
            bbox_tuple = tuple(bbox)
            # print(bbox_tuple)
            if bbox_tuple not in seen:
                seen.add(bbox_tuple)
                gt_boxes.append(bbox)
                gt_labels.append(category)
                # Save original format ground truth for visualization
                if len(gt_vis) < 15:
                    # Convert normalized bbox to pixel coordinates
                    pixel_bbox = [
                        bbox[0] * width,
                        bbox[1] * height,
                        bbox[2] * width,
                        bbox[3] * height
                    ]
                    gt_vis.append({
                        "bbox": pixel_bbox,
                        "label": DOCLAYNET_LABELS[category]
                    })
        
        target_dict = {
            "boxes": torch.tensor(gt_boxes, dtype=torch.float32),
            "labels": torch.tensor(gt_labels, dtype=torch.int64)
        }
        
        all_preds.append(pred_dict)
        all_targets.append(target_dict)
        
        # Save first 5 predictions for visualization
        if len(saved_predictions) < 15:
            saved_predictions.append({
                "image_idx": idx,
                "image": image,
                "predictions": norm_predictions,
                "ground_truth": gt_vis
            })
    # Save predictions and visualize
    print("\nSaving predictions and visualizations...")
    for i, pred_data in enumerate(saved_predictions):
        # Save visualization
        save_dir = "preds_test/rt_detr_doclaynet"
        os.makedirs(save_dir, exist_ok=True)
        save_path = f"{save_dir}/pred_vis_{i}.png"
        visualize_boxes(
            pred_data["image"],
            pred_data["predictions"],
            pred_data["ground_truth"],
            save_path
        )
        
        # Remove image from JSON data
        pred_data.pop("image")
    
    
    # Save prediction details to JSON
    with open("preds_test/predictions_detr_doclayout.json", "w") as f:
        json.dump(saved_predictions, f, indent=2)
    
    # Calculate mAP
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
            class_name = DOCLAYNET_LABELS[label_id]
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
    
    with open("preds_test/doclayout_detailed_results_rt_detr.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    return metrics

if __name__ == "__main__":
    MODEL_PATH = get_model_path()
    print("Printing Model Paths")
    print(MODEL_PATH)
    metrics = evaluate_model(MODEL_PATH, use_all_data=True)
    print(f"\nResults:")
    print(f"Processed images: {metrics['processed_images']}")
    print(f"mAP@50: {metrics['map_50']:.4f}")
    print(f"mAP@0.5:0.95: {metrics['map']:.4f}")
    print(f"mAP@0.5:0.95: {metrics['map']:.4f}")
    print(f"Precision: {metrics['macro_avg']['precision']:.4f}")
    print(f"Recall: {metrics['macro_avg']['recall']:.4f}")
    print(f"F1: {metrics['macro_avg']['f1']:.4f}")
