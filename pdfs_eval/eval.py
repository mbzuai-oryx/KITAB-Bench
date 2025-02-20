import re
from metric import TEDS
from bs4 import BeautifulSoup
import json
from tqdm import tqdm
from sacrebleu import corpus_chrf


def flatten_table(html):
    soup = BeautifulSoup(html, 'html.parser')
    new_table = soup.new_tag('table')
    new_tbody = soup.new_tag('tbody')
    new_table.append(new_tbody)
    thead = soup.find('thead')
    if thead:
        for row in thead.find_all('tr'):
            new_tbody.append(row)
        thead.extract()
    tbody = soup.find('tbody')
    if tbody:
        for row in tbody.find_all('tr'):
            new_tbody.append(row)
        tbody.extract()
    tfoot = soup.find('tfoot')
    if tfoot:
        for row in tfoot.find_all('tr'):
            new_tbody.append(row)
        tfoot.extract()
    table = soup.find('table')
    if table:
        for row in table.find_all('tr'):
            new_tbody.append(row)
        table.extract()
    return str(new_table.prettify())

def arabic_to_english_numerals(input_str):
    arabic_numerals = '٠١٢٣٤٥٦٧٨٩'  # Arabic-Indic digits
    english_numerals = '0123456789'  # Western Arabic digits
    translation_table = str.maketrans(arabic_numerals, english_numerals)
    return input_str.translate(translation_table)

def clean_html_tags(table_html):
    allowed_tags = {"table", "tr", "th", "tbody", "td"}
    soup = BeautifulSoup(table_html, "html.parser")
    for tag in soup.find_all():
        if tag.name not in allowed_tags:
            tag.unwrap()  # Replace the tag with its contents
    return str(soup)


def aug_html(s: str):
    s = s.replace('dir=\"<built-in function dir>\"', "")
    s = s.replace("border=\"1\"", "")
    s = re.sub(r"\s+", ' ', s)
    s = arabic_to_english_numerals(s)
    s = flatten_table(s)
    s = clean_html_tags(s)
    return f"<html>\n<body>\n{s}\n</body></html>"

def avg(l):
    if len(l) == 0: return None
    return sum(l) / len(l)

def preprocess_arabic_text(text: str) -> str:
    # Remove newlines
    text = text.replace("<image>", "")
    text = text.replace("\n", " ")
    text = text.replace("\t", " ")
    # Remove diacritics (tashkeel)
    text = re.sub(r'[\u064B-\u065F\u0670]', '', text)
    # Normalize alef variants to bare alef
    text = re.sub('[إأٱآا]', 'ا', text)
    # Normalize teh marbuta to heh
    text = text.replace('ة', 'ه')
    # Normalize alef maksura to yeh
    text = text.replace('ى', 'ي')
    # This regex matches one or more tatweel characters (ـ) or any whitespace sequence.
    # The lambda replaces whitespace sequences with a single space,
    # and removes tatweel characters by replacing them with an empty string.
    text = re.sub(r'(ـ+)|\s+', lambda m: ' ' if m.group(0).isspace() else '', text).strip()
    text = ' '.join(text.split())
    return text

if __name__ == '__main__':
    model_name = "qwen2vl"
    with open(f'results/{model_name}.json') as fp:
        data = json.load(fp)
    teds = TEDS(n_jobs=32)
    all_scores = []
    pred_texts = []
    gt_texts = []
    for sample in tqdm(data):
        pred_texts.append(preprocess_arabic_text(sample['pred']['text']))
        gt_texts.append(preprocess_arabic_text(sample['gt']['text']))
        gt_tables = sample['gt']['tables']
        pred_tables = sample['pred']['tables']
        true_json = []
        pred_json = []
        samples_idx = []
        max_scores = {str(gt_idx): 0 for gt_idx in range(len(gt_tables))}
        for gt_idx, gt_table in enumerate(gt_tables):
            for pred_idx, pred_table in enumerate(pred_tables):
                idx = f"{gt_idx}.{pred_idx}"
                samples_idx.append(idx)
                true_json.append(aug_html(preprocess_arabic_text(gt_table)))
                pred_json.append(aug_html(preprocess_arabic_text(pred_table)))
        scores = teds.batch_evaluate(pred_json, true_json, samples_idx)
        for k, score in scores.items():
            gt_idx = k.split(".")[0]
            max_scores[gt_idx] = max(max_scores[gt_idx], score)
        avg_score = avg(list(max_scores.values()))
        if avg_score is not None:
            all_scores.append(avg_score)
        
    chrf = corpus_chrf(pred_texts, [gt_texts]).score
    tables_average = avg(all_scores) * 100
    all_average = (chrf + tables_average) / 2
    print(f"Text Average: {chrf:.2f}")
    print(f"Tables Average: {tables_average:.2f}")
    print(f"All Average: {all_average:.2f}")