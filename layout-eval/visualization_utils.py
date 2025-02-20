import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import os

def plot_boxes(image, predictions, ground_truth, save_path):
    """
    Plot and save an image with predicted and ground truth bounding boxes.
    
    Args:
        image: PIL Image
        predictions: List of dict with 'bbox', 'label', and 'confidence'
        ground_truth: List of dict with 'bbox' and 'label'
        save_path: Path to save the visualization
    """
    # Create figure and axis
    fig, ax = plt.subplots(1, figsize=(16, 16))
    
    # Display image
    ax.imshow(image)
    
    # Plot ground truth boxes in green
    for box in ground_truth:
        bbox = box['bbox']
        label = box['label']
        
        # Create rectangle patch
        rect = patches.Rectangle(
            (bbox[0], bbox[1]),
            bbox[2] - bbox[0],
            bbox[3] - bbox[1],
            linewidth=2,
            edgecolor='g',
            facecolor='none'
        )
        ax.add_patch(rect)
        
        # Add label
        ax.text(
            bbox[0], bbox[1] - 5,
            f'GT: {label}',
            color='g',
            fontsize=8,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none')
        )
    
    # Plot predicted boxes in red
    for box in predictions:
        bbox = box['bbox']
        label = box['label']
        conf = box.get('confidence', 1.0)
        
        # Create rectangle patch
        rect = patches.Rectangle(
            (bbox[0], bbox[1]),
            bbox[2] - bbox[0],
            bbox[3] - bbox[1],
            linewidth=2,
            edgecolor='r',
            facecolor='none'
        )
        ax.add_patch(rect)
        
        # Add label with confidence
        ax.text(
            bbox[0], bbox[3] + 5,
            f'Pred: {label} ({conf:.2f})',
            color='r',
            fontsize=8,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none')
        )
    
    # Remove axes
    ax.axis('off')
    
    # Save the plot
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()

def save_visualizations(saved_predictions, output_dir="visualizations"):
    """
    Save visualizations for a batch of predictions.
    
    Args:
        saved_predictions: List of dict containing image and box data
        output_dir: Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    for i, pred_data in enumerate(saved_predictions):
        if 'image' not in pred_data:
            continue
            
        save_path = os.path.join(output_dir, f'boxes_visualization_{i}.png')
        plot_boxes(
            pred_data['image'],
            pred_data['predictions'],
            pred_data['ground_truth'],
            save_path
        )
        
        # Remove image from data to avoid serialization issues
        pred_data.pop('image')

def visualize_boxes(image, predictions, ground_truth, save_path):
    """Visualize and save prediction vs ground truth comparison"""
    # Create figure and axes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Convert PIL image to RGB for matplotlib
    img_array = image.convert('RGB')
    
    # Show image in both subplots
    ax1.imshow(img_array)
    ax2.imshow(img_array)
    
    # Plot predictions
    for pred in predictions:
        bbox = pred["bbox"]
        rect = patches.Rectangle(
            (bbox[0], bbox[1]), 
            bbox[2] - bbox[0], 
            bbox[3] - bbox[1],
            linewidth=2,
            edgecolor='r',
            facecolor='none'
        )
        ax1.add_patch(rect)
        ax1.text(
            bbox[0], bbox[1]-5, 
            f"{pred['label']}: {pred['confidence']:.2f}",
            color='red', 
            fontsize=8
        )
    
    # Plot ground truth
    for gt in ground_truth:
        bbox = gt["bbox"]
        rect = patches.Rectangle(
            (bbox[0], bbox[1]), 
            bbox[2] - bbox[0], 
            bbox[3] - bbox[1],
            linewidth=2,
            edgecolor='g',
            facecolor='none'
        )
        ax2.add_patch(rect)
        ax2.text(
            bbox[0], bbox[1]-5, 
            gt["label"],
            color='green', 
            fontsize=8
        )
    
    ax1.set_title("Predictions")
    ax2.set_title("Ground Truth")
    
    # Remove axes
    ax1.axis('off')
    ax2.axis('off')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
        