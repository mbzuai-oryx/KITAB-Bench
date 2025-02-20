import datasets
import os
from tqdm import tqdm
import json
from bs4 import BeautifulSoup
from marker_model import MarkerOCR as OCRModel
from utils import safe_to_df

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

def get_table(html):
  soup = BeautifulSoup(html, "html.parser")
  return soup.find("table")


  

def to_html(csv):
    df = safe_to_df(csv)
    html_string = df.to_html(index=False, escape=False)
    return html_string


RESULTS_DIR = "results"
MAX_TOKENS = 2000
NUM_WORKERS = 2
os.makedirs(RESULTS_DIR, exist_ok=True)
def get_type(meta):
    return eval(meta)['figure_type']

model_name = "marker"
output_path = f"{RESULTS_DIR}/{model_name}_csv.json"
ds = datasets.load_dataset("ahmedheakl/arocrbench_tablesf", split="train", num_proc=NUM_WORKERS)
data = []
model = OCRModel(max_tokens=MAX_TOKENS)

for idx, sample in tqdm(enumerate(ds), total=len(ds), desc=f"Evaluating tables"):
  is_html = "HTML" in eval(sample['metadata'])["_pipeline"]
  if is_html: continue
  img = sample['image']
  try:
      prompt = HTML_PROMPT if is_html else DF_PROMPT
      pred = model(prompt, img)
      pred = str(get_table(pred))
      # pred = pred.split("```html")[-1].split("```")[0]
      # pred = pred.split("```csv")[-1].split("```")[0]
  except Exception as e:
      print(f"Skipping {idx} for {e}")
      pred = ""
  gt = str(get_table(sample["code"])) if is_html else to_html(sample['data'])
  data.append({"idx": idx, "gt": gt, "pred": pred, "type": get_type(sample['metadata'])})


with open(output_path, "w") as f:
    json.dump(data, f, indent=4, ensure_ascii=False)