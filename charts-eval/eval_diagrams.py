import numpy as np
import editdistance
import json, re
import networkx as nx

import networkx as nx
import signal


def convert_to_json(text):
    match = re.search(r'\{\s*"nodes":.*?"edges":\s*\[.*?\]\s*\}', text, re.DOTALL)
    if match:
        json_str = match.group(0)
        try:
            return json.loads(json_str)  # Convert to a Python dictionary
        except json.JSONDecodeError:
            print("Extracted text is not valid JSON")
    else:
        print("No valid JSON found")
    return None


import networkx as nx

def json_to_graph(json_data):
    G = nx.Graph()
    
    def add_nodes_and_edges(data, parent=None):
        if isinstance(data, dict):
            for key, value in data.items():
                # Add key as node
                G.add_node(key, details=value)
                
                # If the current node has a parent, create an edge
                if parent is not None:
                    G.add_edge(parent, key)
                
                # Recursively handle nested dictionaries or lists
                add_nodes_and_edges(value, parent=key)
        elif isinstance(data, list):
            for item in data:
                # If the list item is a dictionary, call recursively
                add_nodes_and_edges(item, parent)
        else:
            # If it's a simple value, treat as a node
            if parent is not None:
                G.add_edge(parent, data)
    
    # Start the recursive function
    add_nodes_and_edges(json_data)
    
    return G


def compute_ged(json_pred, json_gt):
    json_gt = eval(json_gt) 
    json_pred = eval(json_pred)

    G1 = json_to_graph(json_gt)
    G2 = json_to_graph(json_pred)

    # Compute Graph Edit Distance (GED) using NetworkX
    try:
        ged = nx.graph_edit_distance(G1, G2)
    except Exception as e:
        return 1000


    # Normalized GED (optional)
    max_nodes = max(len(G1.nodes), len(G2.nodes))
    normalized_ged = ged / max_nodes if max_nodes > 0 else 0

    return normalized_ged



# Function to calculate edit distance between two strings
def calculate_edit_distance(pred_str, gt_str):
    return editdistance.eval(pred_str, gt_str) / max(len(pred_str), len(gt_str))

# Function to calculate relative error between two values
def calculate_relative_error(pred_value, gt_value):
    if gt_value == 0:
        return 0.0  # Avoid division by zero
    return abs(pred_value - gt_value) / abs(gt_value)

def calculate_scrm(prediction: dict, ground_truth: dict):

    pred_type = prediction.get("type", "").strip().lower()
    gt_type = ground_truth.get("type", "").strip().lower()
    # Calculate the edit distance for the type (Jthr)
    J1 = calculate_edit_distance(pred_type, gt_type)

    pred_topic = prediction.get("topic", "").strip().lower()
    gt_topic = ground_truth.get("topic", "").strip().lower()
    J2 = calculate_edit_distance(pred_topic, gt_topic)
    
    # Calculate the relative error for the data (ethr)
    pred_data = prediction.get("data", "")
    gt_data = ground_truth.get("data", "")
    J3 = compute_json_similarity(pred_data, gt_data)

    J = 0.5*J1 + 0.2*J2 + 0.3*J3

    # if len(pred_values) > 0 and len(gt_values) > 0:
    #     ethr_value = np.mean([calculate_relative_error(p, g) for p, g in zip(pred_values, gt_values)])
    # else:
    ethr_value = 0.0  # No numeric data present

    # Evaluate if prediction passes the thresholds
    score = 0
    Jthr = 1.0/5.0
    if J <= Jthr:
        score = 5  # Excellent prediction
    elif J <= 2.0/5:
        score = 4  # Good prediction
    elif J <= 3.0/5:
        score = 3  # Good understanding with minor errors
    elif J <= 4.0/5:
        score = 2  # Good understanding with minor errors        
    elif J > Jthr:
        score = 1  # Largely incorrect prediction
    print(f"{J1:.2f} | {J2:.2f} | {J3:.2f} | {J:.2f} | {score}")
    return score, ethr_value, J

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

from difflib import SequenceMatcher

def compute_json_similarity(json1, json2):
    """
    Compute a similarity ratio (0 to 1) between two JSON objects.
    The JSON objects are dumped as strings with sorted keys to ensure
    a consistent ordering.
    """
    json1_str = json.dumps(json1, sort_keys=True, ensure_ascii=False)
    json2_str = json.dumps(json2, sort_keys=True, ensure_ascii=False)
    ratio = SequenceMatcher(None, json1_str, json2_str).ratio()
    return ratio


TYPE_PROMPT = """You are an expert in detecting chart types. Below are examples of the expected output format:

Example 1:  
treemap 

Example 2:  
flowchart 

Example 3:
diagram

Your task is to determine the type of chart shown in the given image.  

**Instructions:**  
- **Respond with only the chart type** (e.g., 'flowchart', 'sequence').  
- **Do not provide any explanations, descriptions, or additional text.**  
- **Ensure the output strictly follows the format shown in the examples.**  

What type of chart is shown in the image?"""




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

DATA_PROMPT = """You are an expert in diagram data extraction. Your task is to analyze the given diagram and generate structured data in JSON format that captures nodes (entities) and edges (relationships).  

## Output Format Example:
flowchart: 
```json
{
  "nodes": [
    {
      "id": "1",
      "text": "جمع النفايات",
      "description": "جمع النفايات الصلبة من المناطق الحضرية"
    },
    {
      "id": "2",
      "text": "فرز النفايات",
      "description": "فرز النفايات إلى مواد قابلة لإعادة التدوير وغير قابلة"
    },
    {
      "id": "3",
      "text": "نقل النفايات",
      "description": "نقل النفايات غير القابلة للتدوير إلى مرافق التحويل"
    }
  ],
  "edges": [
    {
      "from": "1",
      "to": "2",
      "text": "فرز"
    },
    {
      "from": "2",
      "to": "3",
      "text": "نقل"
    },
    {
      "from": "3",
      "to": "4",
      "text": "معالجة"
    }
  ]
}
treemap:
```json
{
  "تخصيص التمويل الحكومي والمساعدات": {
    "التنمية الاقتصادية": {
      "دعم المشاريع الصغيرة": "تمويل الأسر المتأثرة",
      "تحفيز الاستثمارات المحلية": "إعفاءات ضريبية"
    },
    "البنية التحتية": {
      "تحسين النقل العام": "تحديث الحافلات والقطارات",
      "الصرف الصحي والمياه": "معالجة مياه الصرف"
    },
    "التعليم والتأهيل": {
      "توفير المنح الدراسية": "تعليم للفئات المحرومة",
      "برامج التدريب المهني": "تأهيل للخريجين العاطلين"
    },
    "الرعاية الصحية": {
      "تعزيز المراكز الصحية": "بناء مستشفيات جديدة",
      "التوعية الصحية": "حملات ضد الأمراض المزمنة"
    },
    "البيئة": {
      "حماية المناطق الطبيعية": "إنشاء محميات طبيعية",
      "التخفيف من التلوث": "مراقبة الانبعاثات الصناعية"
    }
  }
}
class diagram:
```json
{
  "ورشة_عمل": {
    "خصائص": [
      "اسم_الورشة",
      "تاريخ_الورشة",
      "مدة_الورشة"
    ],
    "علاقات": {
      "تحتوي_على": "جلسة_تدريب",
      "ينظم_من": "منظم",
      "يشارك_في": "مشارك"
    }
  },
  "جلسة_تدريب": {
    "خصائص": [
      "اسم_الجلسة",
      "مدة_الجلسة",
      "محتوى"
    ],
    "علاقات": {
      "يقدم_من": "مدرب",
      "يتبع_إلى": "ورشة_عمل"
    }
  },
  "مدرب": {
    "خصائص": [
      "اسم_المدرب",
      "تخصص",
      "سنوات_الخبرة"
    ],
    "علاقات": {
      "يقدم": "جلسة_تدريب"
    }
  },
  "مشارك": {
    "خصائص": [
      "اسم_المشارك",
      "المؤهلات",
      "الخبرة"
    ],
    "علاقات": {
      "يشترك_في": "ورشة_عمل"
    }
  },
  "منظم": {
    "خصائص": [
      "اسم_المنظم",
      "مهام_تنظيمية"
    ],
    "علاقات": {
      "ينظم": "ورشة_عمل"
    }
  }
}
"""




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


