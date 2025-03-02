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