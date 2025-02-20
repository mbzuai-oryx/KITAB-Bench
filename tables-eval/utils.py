import pandas as pd
from io import StringIO
from bs4 import BeautifulSoup

def safe_to_df(s: str) -> pd.DataFrame:
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

def add_tbody(html):
    soup = BeautifulSoup(html, 'html.parser')
    table = soup.find('table')
    rows = table.find_all('tr')
    tbody = soup.new_tag('tbody')
    for row in rows:
        tbody.append(row.extract())
    table.append(tbody)
    return str(soup.prettify())