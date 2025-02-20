import json

def evaluate_table_extraction(ground_truth, prediction, row_key=None):
    """
    Compares two table JSONs (lists of dictionaries) on a cell-level basis.

    Args:
        ground_truth (list): List of dictionaries for ground truth.
        prediction (list): List of dictionaries for model output.
        row_key (str, optional): Key that uniquely identifies each row.
            If provided, rows are aligned based on this key.
            Otherwise, rows are compared in order.

    Returns:
        precision (float): Correctly predicted cells / total predicted cells.
        recall (float): Correctly predicted cells / total ground truth cells.
        f1 (float): Harmonic mean of precision and recall.
    """
    if isinstance(ground_truth, dict) and len(ground_truth) == 1:
        key, value = next(iter(ground_truth.items()))
        if isinstance(value, list):
            ground_truth = value

    correct_cells = 0
    total_predicted_cells = 0
    total_ground_truth_cells = 0

    # If a row_key is provided, build an index by that key
    if row_key:
        gt_index = {row[row_key]: row for row in ground_truth}
        pred_index = {row[row_key]: row for row in prediction}

        # Process each ground truth row
        for key_value, gt_row in gt_index.items():
            total_ground_truth_cells += len(gt_row)
            if key_value in pred_index:
                pred_row = pred_index[key_value]
                total_predicted_cells += len(pred_row)
                # Compare each cell in the ground truth row
                for col, gt_val in gt_row.items():
                    # Fetch the predicted value (if any)
                    pred_val = pred_row.get(col)
                    # Compare values after stripping whitespace; adjust as needed
                    if pred_val is not None and str(pred_val).strip() == str(gt_val).strip():
                        correct_cells += 1
            else:
                # If a row is missing in the prediction, its cells are counted as false negatives.
                # (total_ground_truth_cells has already been increased for that row.)
                pass

        # Count cells from predicted rows that do not exist in ground truth (false positives)
        for key_value, pred_row in pred_index.items():
            if key_value not in gt_index:
                total_predicted_cells += len(pred_row)
    else:
        # If no row_key is provided, assume rows are aligned by index.
        for i, gt_row in enumerate(ground_truth):
            total_ground_truth_cells += len(gt_row)
            if i < len(prediction):
                pred_row = prediction[i]
                total_predicted_cells += len(pred_row)
                for col, gt_val in gt_row.items():
                    pred_val = pred_row.get(col)
                    if pred_val is not None and str(pred_val).strip() == str(gt_val).strip():
                        correct_cells += 1
            # Else, missing prediction for this row; the ground truth cells count as false negatives.

        # For extra rows in the prediction (if any), count their cells as false positives.
        if len(prediction) > len(ground_truth):
            for i in range(len(ground_truth), len(prediction)):
                total_predicted_cells += len(prediction[i])

    # Calculate precision, recall, and F1-score
    precision = correct_cells / total_predicted_cells if total_predicted_cells > 0 else 0
    recall = correct_cells / total_ground_truth_cells if total_ground_truth_cells > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1

if __name__ == "__main__":
    with open("results/qwen2vl7b.json", "r") as f:
        data = json.load(f)
    p, r, f1, tot = 0, 0, 0, 0
    for d in data:
      if len(d['pred'].strip()) == 0: continue
      try:
        gt = eval(d['gt'])
        pred = eval(d['pred'])
      except:
        continue        
      from pprint import pprint
      pprint(gt)
      pprint(pred)
      print("="*50)
      precision, recall, f1_score = evaluate_table_extraction(gt, pred)
      p += precision
      r += recall 
      f1 += f1_score
      tot += 1

    print(f"Precision: {p*100/tot:.2f} | Recall: {r*100/tot:.2f} | F1 Score: {f1*100/tot:.2f}")

