import numpy as np
import editdistance

def calculate_edit_distance(pred_str, gt_str):
    return editdistance.eval(pred_str, gt_str)

def calculate_relative_error(pred_value, gt_value):
    if gt_value == 0:
        return 0.0  # Avoid division by zero
    return abs(pred_value - gt_value) / abs(gt_value)

def arabic_to_english_numerals(input_str):
    arabic_numerals = '٠١٢٣٤٥٦٧٨٩'  
    english_numerals = '0123456789' 
    translation_table = str.maketrans(arabic_numerals, english_numerals)
    return input_str.translate(translation_table)

def extract_numeric_values(data):
    """
    Extracts numeric values from a comma-separated data string (e.g., '150, 300, 400').
    """
    numbers = []
    for item in data.split("\n"):
        try:
            value = float(item.split(",")[-1].strip())
            numbers.append(value)
        except ValueError:
            continue
    return numbers

def calculate_scrm(prediction: dict, ground_truth: dict, type_w: float = 0.7, topic_w: float = 0.3):
    pred_type = prediction.get("type", "").strip().lower()
    gt_type = ground_truth.get("type", "").strip().lower()
    J1 = calculate_edit_distance(pred_type, gt_type)

    pred_topic = prediction.get("topic", "").strip().lower()
    gt_topic = ground_truth.get("topic", "").strip().lower()
    J2 = calculate_edit_distance(pred_topic, gt_topic)

    J = type_w * J1 + topic_w * J2
    
    pred_data = prediction.get("data", "")
    gt_data = ground_truth.get("data", "")
    pred_data = arabic_to_english_numerals(pred_data)
    gt_data = arabic_to_english_numerals(gt_data)
    pred_values = extract_numeric_values(pred_data)
    gt_values = extract_numeric_values(gt_data)
    
    
    if len(pred_values) > 0 and len(gt_values) > 0:
        ethr_value = np.mean([calculate_relative_error(p, g) for p, g in zip(pred_values, gt_values)])
    else:
        ethr_value = 0.0  # No numeric data present

    score = 0
    if J <= 10 and ethr_value <= 0.5:
        score = 5  # Excellent prediction
    elif J <= 15 and ethr_value <= 1:
        score = 4  # Good prediction
    elif J <= 20 and ethr_value <= 1.5:
        score = 3  # Good understanding with minor errors
    elif J <= 30 and ethr_value <= 5:
        score = 2  # Good understanding with minor errors        
    elif J > 30 or ethr_value > 5:
        score = 1  # Largely incorrect prediction
    print(f"{ethr_value:.2f} | {J:.2f} | {score}")
    return score