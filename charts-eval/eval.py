import numpy as np
import editdistance, re
from io import StringIO
import pandas as pd
from sacrebleu import corpus_chrf

def find_word_before_chart(text):
    match = re.search(r'(\b\w+)\s+chart', text)
    return f"{match.group(1)} chart" if match else "LOL"

# Function to calculate edit distance between two strings
def calculate_edit_distance(pred_str, gt_str):
    return editdistance.eval(pred_str, gt_str)

# Function to calculate relative error between two values
def calculate_relative_error(pred_value, gt_value):
    if gt_value == 0:
        return 0.0  # Avoid division by zero
    return abs(pred_value - gt_value) / abs(gt_value)

def arabic_to_english_numerals(input_str):
    arabic_numerals = '٠١٢٣٤٥٦٧٨٩'  # Arabic-Indic digits
    english_numerals = '0123456789'  # Western Arabic digits
    translation_table = str.maketrans(arabic_numerals, english_numerals)
    return input_str.translate(translation_table)

def calculate_scrm(prediction: dict, ground_truth: dict):

    pred_type = prediction.get("type", "").strip().lower()
    gt_type = ground_truth.get("type", "").strip().lower()
    # Calculate the edit distance for the type (Jthr)
    J1 = calculate_edit_distance(pred_type, gt_type)

    pred_topic = prediction.get("topic", "").strip().lower()
    gt_topic = ground_truth.get("topic", "").strip().lower()
    J2 = calculate_edit_distance(pred_topic, gt_topic)

    J = 0.7*J1 + 0.3*J2
    
    # Calculate the relative error for the data (ethr)
    pred_data = prediction.get("data", "")
    gt_data = ground_truth.get("data", "")
    # pred_data = arabic_to_english_numerals(pred_data)
    # gt_data = arabic_to_english_numerals(gt_data)
    
    # If 'data' contains numeric values, calculate relative error
    pred_values = extract_numeric_values(pred_data)
    gt_values = extract_numeric_values(gt_data)
    
    
    if len(pred_values) > 0 and len(gt_values) > 0:
        ethr_value = np.mean([calculate_relative_error(p, g) for p, g in zip(pred_values, gt_values)])
    else:
        ethr_value = 0.0  # No numeric data present

    # Evaluate if prediction passes the thresholds
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
    return score, ethr_value, J

# def compare_dataframes(df1: pd.DataFrame, df2: pd.DataFrame) -> float:
#     # Create the union of indices and columns
#     all_index = df1.index.union(df2.index)
#     all_columns = df1.columns.union(df2.columns)
#     # Align both DataFrames on the same index and columns
#     df1_aligned = df1.reindex(index=all_index, columns=all_columns)
#     df2_aligned = df2.reindex(index=all_index, columns=all_columns)
#     # Compare cell by cell (treating NaNs as equal)
#     comparison = (df1_aligned == df2_aligned) | (df1_aligned.isna() & df2_aligned.isna())
#     score = comparison.sum().sum() / comparison.size
#     return score

import pandas as pd
from rapidfuzz import process, fuzz
from scipy.optimize import linear_sum_assignment

def get_fuzzy_common_columns(df1: pd.DataFrame, df2: pd.DataFrame, threshold: float = 80) -> dict:
    mapping = {}
    for col in df1.columns:
        match = process.extractOne(col, df2.columns, scorer=fuzz.token_sort_ratio)
        if match and match[1] >= threshold:
            mapping[col] = match[0]
    return mapping

def row_similarity(row1: pd.Series, row2: pd.Series, col_mapping: dict, numeric_tol: float = 1e-6) -> float:
    """
    Compute a similarity score between two rows using the fuzzy-matched columns.
    
    For numeric values, a near-equality (within a tolerance) counts as a full match.
    For text values, a fuzzy token sort ratio is used.
    Missing values in both rows are treated as a match.
    
    Parameters:
        row1 (pd.Series): Row from the first DataFrame.
        row2 (pd.Series): Row from the second DataFrame.
        col_mapping (dict): Mapping from df1 column names to df2 column names.
        numeric_tol (float): Tolerance for numeric comparisons.
        
    Returns:
        A float score between 0 and 1.
    """
    scores = []
    for col1, col2 in col_mapping.items():
        v1 = row1[col1]
        v2 = row2[col2]
        
        # Both missing? Perfect match.
        if pd.isna(v1) and pd.isna(v2):
            scores.append(1)
        # Only one missing? Mismatch.
        elif pd.isna(v1) or pd.isna(v2):
            scores.append(0)
        else:
            try:
                num1 = float(v1)
                num2 = float(v2)
                scores.append(1 if abs(num1 - num2) < numeric_tol else 0)
            except (ValueError, TypeError):
                # Use fuzzy matching for text comparison.
                score = fuzz.token_sort_ratio(str(v1), str(v2)) / 100.0
                scores.append(score)
    return np.mean(scores) if scores else 0

def compare_dataframes(df1: pd.DataFrame, df2: pd.DataFrame, numeric_tol: float = 1e-6, col_threshold: float = 80) -> float:
    """
    Compare two DataFrames without assuming a constant key.
    
    This version uses fuzzy matching to determine common columns, then computes
    a row-to-row similarity matrix over the fuzzy-matched columns. The Hungarian
    algorithm is used to optimally assign rows between the DataFrames.
    
    Parameters:
        df1 (pd.DataFrame): Ground truth DataFrame.
        df2 (pd.DataFrame): Predicted DataFrame.
        numeric_tol (float): Tolerance for numeric comparisons.
        col_threshold (float): Minimum similarity score (0-100) for column name matching.
        
    Returns:
        A float between 0 and 1 representing the average similarity.
    """
    # Get fuzzy common columns based on column names.
    col_mapping = get_fuzzy_common_columns(df1, df2, threshold=col_threshold)
    if not col_mapping:
        return 0.0  # No basis for comparison if no columns are similar.
    
    n1, n2 = len(df1), len(df2)
    sim_matrix = np.zeros((n1, n2))
    
    # Compute similarity for every pair of rows.
    for i, (_, row1) in enumerate(df1.iterrows()):
        for j, (_, row2) in enumerate(df2.iterrows()):
            sim_matrix[i, j] = row_similarity(row1, row2, col_mapping, numeric_tol)
    
    # Convert similarity to cost (lower cost => higher similarity).
    cost_matrix = 1 - sim_matrix
    
    # Pad cost matrix to square shape to allow unmatched rows.
    n = max(n1, n2)
    if n1 < n:
        pad_rows = np.ones((n - n1, n2))
        cost_matrix = np.vstack([cost_matrix, pad_rows])
    if n2 < n:
        pad_cols = np.ones((cost_matrix.shape[0], n - n2))
        cost_matrix = np.hstack([cost_matrix, pad_cols])
    
    # Solve the assignment problem.
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    total_similarity = 0.0
    matched_rows = 0
    for r, c in zip(row_ind, col_ind):
        if r < n1 and c < n2:
            total_similarity += sim_matrix[r, c]
            matched_rows += 1
    
    # Penalize unmatched rows from df1.
    overall_similarity = total_similarity / n1
    return overall_similarity


def safe_to_df(s: str) -> pd.DataFrame:
    s = arabic_to_english_numerals(s)
    s = s.split("\n\n")[-1]
    csv_data = StringIO(s)
    try:
        df = pd.read_csv(csv_data)
    except pd.errors.ParserError as e:
        csv_data.seek(0)
        print(f"ParserError encountered {e}. Falling back to Python engine with on_bad_lines='skip'.")
        try:
            df = pd.read_csv(csv_data, engine='python', on_bad_lines='skip')
        except:
            df = pd.DataFrame()
    except: 
        df = pd.DataFrame()
    
    # --- Ensure unique column labels ---
    # Convert all column names to strings. This converts NaN to 'nan'
    df.columns = df.columns.astype(str)
    # If there are duplicate column names, drop all but the first occurrence.
    if df.columns.duplicated().any():
        print("Warning: Duplicate column labels found. Dropping duplicates.")
        df = df.loc[:, ~df.columns.duplicated(keep='first')]
    
    # --- (Optional) Ensure unique index labels ---
    if df.index.duplicated().any():
        print("Warning: Duplicate index labels found. Resetting index.")
        df = df.reset_index(drop=True)
    
    return df

def calculate_chartex(pred: dict, gt: dict):
    pred_type = pred['type']
    pred_topic = pred['topic']
    pred_data = pred['data']
    pred_data = pred_data.replace("```", "")
    gt_type = gt['type']
    gt_topic = gt['topic']
    gt_data = gt['data']
    gt_df, pred_df = safe_to_df(gt_data), safe_to_df(pred_data)
    data_score = compare_dataframes(gt_df, pred_df) * 100
    type_variants = {
        "histogram": ["histogram", "bar", "bar chart"], 
        "dual axis": ["dual axis", "combined chart", "line chart", "bar chart"],
        "density plot": ['density plot', "line chart"],
        "sunburst": ["sunburst", "pie chart"],
        "dot plot": ["dot plot", "scatter chart"]
    }
    varients = type_variants.get(gt_type, [gt_type])
    type_score = 0
    for var in varients:
        type_score = max(type_score, corpus_chrf([pred_type], [[var]]).score)    
    topic_score = corpus_chrf([pred_topic], [[gt_topic]]).score
    # print(f"Data: {data_score:.2f} | Topic: {topic_score:.2f} | Type: {type_score:.2f}")
    final_score = 0.85 * data_score + 0.1 * topic_score + 0.05 * type_score
    return final_score

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


TYPE_PROMPT = """You are an expert in detecting chart types. Below are examples of the expected output format:

Example 1:  
bar chart

Example 2:  
scatter chart

Example 3:
histogram

Your task is to determine the type of chart shown in the given image.  

**Instructions:**  
- **Respond with only the chart type** (e.g., 'bar chart', 'scatter chart').  
- **Do not include any additional text, explanations, or descriptions.**  
- **Ensure the output matches the format in the examples exactly.**  

Provide only the chart type in **single quotes** as shown in the examples above.  

What type of chart is shown in the image? Don't output any extra text"""





TOPIC_PROMPT = """أنت خبير في تحليل وتقييم المخططات البيانية. فيما يلي أمثلة توضح تنسيق الإجابة المتوقع:  

**مثال 1:**  
توزيع الكتب الأكثر مبيعاً حسب النوع الأدبي 

**مثال 2:**  
آراء العملاء حول الموضوعات المثيرة للجدل في الكتب

**التعليمات:**  
- **حدد موضوع أو محتوى المخطط البياني فقط.**  
- **اكتب الإجابة باللغة العربية فقط.**  
- **اتبع التنسيق المحدد دون إضافة أي شرح أو تعليق إضافي.**  

ما هو موضوع أو محتوى المخطط البياني؟"""

DATA_PROMPT = """You are an expert in chart data extraction. You are given a chart image and you should provide the chart data in CSV format.
Here are some examples. 
Example 1:
```csv
النوع الأدبي,المبيعات (بالآلاف)  
روايات,٣٥٠  
خيال علمي,١٢٠  
فانتازيا,١٨٠  
حياتي,٩٠  
تاريخ,٧٠  
علم نفس,١١٠  
مذكرات,٨٥  
تكنولوجيا,٦٥  
فنون,٤٥  
أطفال,٢٠٠
```

Example 2:
```csv
موضوع,نسبة العملاء الإيجابية,نسبة العملاء السلبية  
السياسة في الأدب,٤٠,٦٠  
الدين والفكر,٣٥,٦٥  
العلاقات غير التقليدية,٥٥,٤٥  
العنف في القصص,٣٠,٧٠  
الحريات الفردية,٥٠,٥٠  
النقد الاجتماعي,٦٠,٤٠  
التكنولوجيا والمستقبل,٦٥,٣٥
```
Not give me the results as in the previous CSV format."""


prompt = "You're an expert evaluating a model's description of a chart, based on its alignment with the \
ground truth and raw data. Score the model from 0 to 5 based on these criteria: \
0 points: Description irrelevant or shows no understanding of the chart or data. \
1 point: Refers to the chart but with largely incorrect details; minimal understanding. \
2 points: Some correct details, but key elements are wrong or missing; basic understanding with \
significant errors. \
3 points: Most details are correct; good understanding but with minor errors/omissions. \
4 points: All details are correct; very good understanding, minor improvements possible. \
5 points: Comprehensive, accurate description; excellent understanding with no errors; clear \
and detailed, perfect as a standalone explanation. \
Score the model's description on this scale, providing a single value without providing any \
reasons."


