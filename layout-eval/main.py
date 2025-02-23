import os
import argparse
from typing import Dict, Any
import json

# Import all evaluation functions
from rt_detr_bcelayout import evaluate_model as evaluate_rtdetr_bce
from rt_detr_doclayout import evaluate_model as evaluate_rtdetr_doclaynet
from test_surya_bce_layout import process_dataset as evaluate_surya_bce
from test_surya_doclaynet import evaluate_doclaynet as evaluate_surya_doclaynet
from yolo_doc_bcelayout import evaluate_model as evaluate_yolo_bce
from yolo_doc_doclayout import evaluate_model as evaluate_yolo_doclaynet

def get_model_paths() -> Dict[str, str]:
    """Get all required model paths from environment or config."""
    return {
        "rtdetr": os.getenv("RTDETR_MODEL_PATH", "models/rtdetr"),
        "yolo": os.getenv("YOLO_MODEL_PATH", "models/yolo"),
        # Surya doesn't need a path as it downloads automatically
    }

def save_results(results: Dict[str, Any], output_dir: str, model_name: str):
    """Save evaluation results to JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{model_name}_results.json")
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_file}")

def print_metrics(metrics: Dict[str, Any], model_name: str):
    """Print evaluation metrics in a consistent format."""
    print(f"\n{model_name} Results:")
    print(f"Processed images: {metrics['processed_images']}")
    print(f"mAP@50: {metrics['map_50']:.4f}")
    print(f"mAP@75: {metrics['map_75']:.4f}")
    print(f"mAP@0.5:0.95: {metrics['map']:.4f}")
    print(f"Precision: {metrics['macro_avg']['precision']:.4f}")
    print(f"Recall: {metrics['macro_avg']['recall']:.4f}")
    print(f"F1: {metrics['macro_avg']['f1']:.4f}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate document layout analysis models')
    parser.add_argument('--models', nargs='+', choices=['rtdetr', 'surya', 'yolo', 'all'],
                      default=['all'], help='Models to evaluate')
    parser.add_argument('--datasets', nargs='+', choices=['bce', 'doclaynet', 'all'],
                      default=['all'], help='Datasets to evaluate on')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                      help='Directory to save results')
    parser.add_argument('--num_samples', type=int, default=None,
                      help='Number of samples to evaluate (None for all)')
    args = parser.parse_args()

    # Expand 'all' options
    if 'all' in args.models:
        args.models = ['rtdetr', 'surya', 'yolo']
    if 'all' in args.datasets:
        args.datasets = ['bce', 'doclaynet']

    # Get model paths
    model_paths = get_model_paths()
    use_all_data = args.num_samples is None

    # Run evaluations
    for model in args.models:
        for dataset in args.datasets:
            try:
                print(f"\nEvaluating {model} on {dataset} dataset...")
                
                if model == 'rtdetr':
                    if dataset == 'bce':
                        metrics = evaluate_rtdetr_bce(model_paths['rtdetr'], 
                                                    use_all_data=use_all_data,
                                                    num_samples=args.num_samples)
                    else:
                        metrics = evaluate_rtdetr_doclaynet(model_paths['rtdetr'], 
                                                          use_all_data=use_all_data,
                                                          num_samples=args.num_samples)
                
                elif model == 'surya':
                    if dataset == 'bce':
                        metrics = evaluate_surya_bce(dataset_type='bcelayout',
                                                   max_samples=args.num_samples)
                    else:
                        metrics = evaluate_surya_doclaynet(dataset=None,  # It loads internally
                                                         max_samples=args.num_samples)
                
                elif model == 'yolo':
                    if dataset == 'bce':
                        metrics = evaluate_yolo_bce(model_paths['yolo'],
                                                  use_all_data=use_all_data,
                                                  num_samples=args.num_samples)
                    else:
                        metrics = evaluate_yolo_doclaynet(model_paths['yolo'],
                                                        use_all_data=use_all_data,
                                                        num_samples=args.num_samples)

                # Save and print results
                save_results(metrics, args.output_dir, f"{model}_{dataset}")
                print_metrics(metrics, f"{model.upper()} on {dataset.upper()}")

            except Exception as e:
                print(f"Error evaluating {model} on {dataset}: {str(e)}")
                continue

if __name__ == "__main__":
    main()