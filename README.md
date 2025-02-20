<h1 align="center">
  <img src="https://github.com/mbzuai-oryx/KITAB-Bench/blob/website/static/images/kitab.png" alt="KITAB-Bench" width="150px">
  <br>
  KITAB-Bench: A Comprehensive Multi-Domain Benchmark for Arabic OCR and Document Understanding
</h1>
## **Overview**
With the increasing adoption of **Retrieval-Augmented Generation (RAG)** in document processing, robust Arabic **Optical Character Recognition (OCR)** is essential for knowledge extraction. Arabic OCR presents unique challenges due to:
- **Cursive script** and **right-to-left text flow**.
- **Complex typographic** and **calligraphic** variations.
- **Tables, charts, and diagram-heavy documents**.

We introduce **KITAB-Bench**, a **comprehensive Arabic OCR benchmark** that evaluates the performance of **traditional OCR, vision-language models (VLMs), and specialized AI systems**.

### **Key Highlights**
✅ **9 major domains & 36 sub-domains** across **8,809** samples.  
✅ **Diverse document types**: PDFs, handwritten text, structured tables, financial & legal reports.  
✅ **Strong baselines**: Benchmarked against **Tesseract, GPT-4o, Gemini, Qwen**, and more.  
✅ **Evaluation across OCR, layout detection, table recognition, chart extraction, & PDF conversion.**  
✅ **Novel evaluation metrics**: Markdown Recognition (MARS), Table Edit Distance (TEDS), Chart Data Extraction (SCRM).  

---

## **Dataset Overview**
KITAB-Bench covers a **wide range of document types**:

| **Domain**            | **Total Samples** |
|----------------------|-----------------|
| PDF-to-Markdown      | 33              |
| Layout Detection     | 2,100           |
| Line Recognition     | 378             |
| Table Recognition    | 456             |
| Charts-to-DataFrame  | 576             |
| Diagram-to-JSON      | 226             |
| Visual QA (VQA)      | 902             |
| **Total**            | **8,809**        |

📌 **High-quality human-labeled annotations** for fair evaluation.

---

## **Benchmark Tasks**
KITAB-Bench evaluates **9 key OCR and document processing tasks**:

1️⃣ **Text Recognition (OCR)** - Printed & handwritten Arabic OCR.  
2️⃣ **Layout Detection** - Extracting text blocks, tables, figures, etc.  
3️⃣ **Line Recognition** - Identifying & recognizing individual Arabic text lines.  
4️⃣ **Table Recognition** - Parsing structured tables into machine-readable formats.  
5️⃣ **PDF-to-Markdown** - Converting Arabic PDFs into structured Markdown format.  
6️⃣ **Charts-to-DataFrame** - Extracting **21 types of charts** into structured datasets.  
7️⃣ **Diagram-to-JSON** - Extracting **flowcharts, Venn diagrams, networks into JSON.**  
8️⃣ **Visual Question Answering (VQA)** - Understanding questions about Arabic documents.  
9️⃣ **Semantic Reasoning** - Analyzing **complex text layouts, diagrams, and mixed formats.**  

---

## **Evaluation Metrics**
To accurately assess OCR models, KITAB-Bench introduces **new Arabic OCR evaluation metrics**:

| **Metric** | **Purpose** |
|------------|------------|
| **Character Error Rate (CER)** | Measures accuracy of recognized characters. |
| **Word Error Rate (WER)** | Evaluates word-level OCR accuracy. |
| **MARS (Markdown Recognition Score)** | Assesses **PDF-to-Markdown conversion** accuracy. |
| **TEDS (Tree Edit Distance Score)** | Measures **table extraction correctness**. |
| **SCRM (Chart Representation Metric)** | Evaluates **chart-to-data conversion**. |
| **CODM (Code-Oriented Diagram Metric)** | Assesses **diagram-to-JSON extraction accuracy**. |

📌 **KITAB-Bench ensures a rigorous evaluation across multiple dimensions of Arabic document processing.**

---

## **Performance Results**
Our benchmark results demonstrate **significant performance gaps** between different OCR systems:

| **Model** | **OCR Accuracy (CER%)** | **Table Recognition (TEDS%)** | **Charts-to-Data (SCRM%)** |
|----------|--------------------|-----------------|------------------|
| GPT-4o    | **31.0%** | 85.7% | 68.6% |
| Gemini-2.0 | **13.0%** | 83.0% | 71.4% |
| Qwen-2.5 | **49.2%** | 59.3% | 36.2% |
| EasyOCR  | **58.0%** | 49.1% | N/A |
| Tesseract | **54.0%** | 28.2% | N/A |

📌 **Key Insights**:  
✅ **GPT-4o and Gemini models significantly outperform traditional OCR**.  
✅ **Surya and Tesseract perform well for standard text but fail in table and chart recognition**.  
✅ **Open-source models like Qwen-2.5 still lag behind proprietary solutions**.

---

## **Installation & Usage**
To use KITAB-Bench, follow these steps:

### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/yourusername/kitab-bench.git
cd kitab-bench
```

### **2️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```

### **3️⃣ Run Benchmark Evaluation**
```bash
python evaluate.py --model GPT-4 --task ocr
```

### **4️⃣ Custom OCR Model Evaluation**
```bash
python evaluate.py --model YOUR_MODEL --task table_recognition
```



---

## 🚀 Contributing 

We welcome **contributions** from the community! 🚀  

✅ If you find a **bug**, please [open an issue](https://github.com/your-repo/issues).  

✅ Want to **add a new OCR model? Submit a pull request**.  

✅ Found an **improvement**? We are **open to suggestions**!  

---

## 📖 Citations  

If you use **KITAB-Bench** in your research, please cite:

```bibtex
@article{heakl2024kitab,
    title={KITAB-Bench: A Comprehensive Multi-Domain Benchmark for Arabic OCR and Document Understanding},
    author={Ahmed Heakl et al.},
    year={2024}
}
```

