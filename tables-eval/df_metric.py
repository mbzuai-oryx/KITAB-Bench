import pandas as pd
import json
from io import StringIO
from tqdm import tqdm
import traceback

def compare_dataframes(df1: pd.DataFrame, df2: pd.DataFrame) -> float:
    # Create the union of indices and columns
    all_index = df1.index.union(df2.index)
    all_columns = df1.columns.union(df2.columns)
    # Align both DataFrames on the same index and columns
    df1_aligned = df1.reindex(index=all_index, columns=all_columns)
    df2_aligned = df2.reindex(index=all_index, columns=all_columns)
    # Compare cell by cell (treating NaNs as equal)
    comparison = (df1_aligned == df2_aligned) | (df1_aligned.isna() & df2_aligned.isna())
    score = comparison.sum().sum() / comparison.size
    return score

def arabic_to_english_numerals(input_str):
    arabic_numerals = '٠١٢٣٤٥٦٧٨٩'  # Arabic-Indic digits
    english_numerals = '0123456789'  # Western Arabic digits
    translation_table = str.maketrans(arabic_numerals, english_numerals)
    return input_str.translate(translation_table)

def safe_to_df(s: str) -> pd.DataFrame:
    s = arabic_to_english_numerals(s)
    csv_data = StringIO(s)
    try:
        df = pd.read_csv(csv_data)
    except pd.errors.ParserError as e:
        csv_data.seek(0)
        print("ParserError encountered. Falling back to Python engine with on_bad_lines='skip'.")
        df = pd.read_csv(csv_data, engine='python', on_bad_lines='skip')
    
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

with open("results/doclingeasyocr_csv.json", "r") as f:
    data = json.load(f)


tot = 0
num_failed = 0
bad_sample = []
for f in tqdm(data):
    try:
        gt, pred = safe_to_df(f['gt']), safe_to_df(f['pred'])
        score = compare_dataframes(gt, pred)
        tot += score
        if score < 0.1:
            bad_sample.append(f['idx'])
    except Exception as e:
        idx = f['idx']
        if "No columns to parse" in str(e): continue
        print(f"Skipping sample {idx}: {traceback.format_exc()}\n===>GT:\n{f['gt']}\n===>Pred:\n{f['pred']}")
        num_failed += 1
tot /= len(data)
print(f"Score: {tot*100:.2f} | Failed: {num_failed}/{len(data)}")
print(bad_sample)