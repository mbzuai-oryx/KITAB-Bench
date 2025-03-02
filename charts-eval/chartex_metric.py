import pandas as pd
from rapidfuzz import process, fuzz
from scipy.optimize import linear_sum_assignment
from sacrebleu import corpus_chrf
import pandas as pd
from io import StringIO
import numpy as np

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

def arabic_to_english_numerals(input_str):
    arabic_numerals = '٠١٢٣٤٥٦٧٨٩'  
    english_numerals = '0123456789' 
    translation_table = str.maketrans(arabic_numerals, english_numerals)
    return input_str.translate(translation_table)

def calculate_chartex(pred: dict, gt: dict, data_w: float = 0.85, topic_w: float = 0.1, type_w: float = 0.05):
    pred_type = pred['type']
    pred_topic = pred['topic']
    pred_data = pred['data'].replace("```", "")
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
    final_score = data_w * data_score + topic_w * topic_score + type_w * type_score
    return final_score