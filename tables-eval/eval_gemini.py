import datasets
import os
from PIL import Image
from tqdm import tqdm
import json
from openai import OpenAI
import base64
import multiprocessing as mp
import io
from bs4 import BeautifulSoup
from gemini_model import GeminiOCR

HTML_PROMPT = """Extract the data from the table below and provide the output in HTML format. Output only the data as HTML and nothing else. Here is one example:
```html
<table> <thead> <tr> <th>الفئة</th> <th>النسبة المئوية</th> <th>التفاصيل</th> </tr> </thead> <tbody> <tr> <td>الأسهم المحلية</td> <td>٣٥٪</td> <td>شركة سابك, شركة الاتصالات السعودية, شركة أرامكو</td> </tr> <tr> <td>الأوراق المالية الحكومية</td> <td>٢٠٪</td> <td>حكومة السعودية, حكومة الإمارات</td> </tr> <tr> <td>السندات الدولية</td> <td>١٥٪</td> <td>بنك سويسري, بنك جي بي مورغان</td> </tr> <tr> <td>العقارات التجارية</td> <td>١٥٪</td> <td>دبي, الرياض, المنامة</td> </tr> <tr> <td>الاستثمارات البديلة</td> <td>١٠٪</td> <td>صناديق الاستثمار الخاصة, صناديق التحوط</td> </tr> <tr> <td>النقد وما يعادله</td> <td>٥٪</td> <td>بنك الإمارات دبي الوطني, بنك أبوظبي الأول</td> </tr> </tbody> </table>
```
Now generate the data for the provided table."""

DF_PROMPT = """Extract the data from the table below and provide the output in CSV format. Output only the data as CSV and nothing else. Here is one example:
```csv
اسم الشركة,الصفقة,مبلغ الصفقة (مليون دولار),تاريخ الاتفاقية,نوع التقنية 
أوراكل,الاستحواذ على شركة سيرنر,28,2023-06-15,الحوسبة السحابية والنمذجة الحيوية 
أمازون ويب سيرفيسز,شراكة مع شركة موديلينغ بيو,15,2023-04-20,النمذجة الحيوية 
مايكروسوفت,شراكة مع شركة بيومادكس,12,2023-03-10,الحوسبة السحابية 
جوجل كلاود,شراء شركة بيوكيم سوليوشنز,35,2023-09-01,النمذجة الحيوية 
آي بي إم,توسع في شراكتها مع شركة جينوميك سوفتوير,18,2023-05-05,حوسبة بيولوجية

```
Now generate the data for the provided table."""

RESULTS_DIR = "results"
MAX_TOKENS = 2000
NUM_WORKERS = 16

os.makedirs(RESULTS_DIR, exist_ok=True)

def get_table(html):
  soup = BeautifulSoup(html, "html.parser")
  return soup.find("table")

def get_type(meta):
    return eval(meta)['figure_type']

def process_sample(args):
    idx, sample = args
    img = sample["image"]
    is_html = "HTML" in eval(sample['metadata'])['_pipeline']
    prompt = HTML_PROMPT if is_html else DF_PROMPT
    model = GeminiOCR()
    try:
        pred = model(prompt, img)
        # pred = pred.split("```html")[-1].split("```")[0]
        pred = pred.split("```csv")[-1].split("```")[0]
    except Exception as e:
        print(f"Skipping sample {idx} due to error: {e}")
        pred = ""
    gt = str(get_table(sample["code"])) if is_html else sample['data']
    return {"idx": idx, "gt": gt, "pred": pred, "type": get_type(sample['metadata'])}

model_name = "gemini2flash"
ds_id = "ahmedheakl/arocrbench_tablesf"
output_path = f"{RESULTS_DIR}/{model_name}_csv.json"
ds = datasets.load_dataset(ds_id, split="train", num_proc=NUM_WORKERS)
data = []
tasks = [(idx, sample) for idx, sample in enumerate(ds) if "HTML" not in eval(sample['metadata'])["_pipeline"]]
with mp.Pool(NUM_WORKERS) as pool:
    for result in tqdm(pool.imap(process_sample, tasks),
                        total=len(tasks),
                        desc=f"Evaluating tables ..."):
        data.append(result)
with open(output_path, "w") as f:
    json.dump(data, f, indent=4, ensure_ascii=False)