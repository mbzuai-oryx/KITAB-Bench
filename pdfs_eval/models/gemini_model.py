from PIL import Image
import os
from google import genai
import re
from pdf2image import convert_from_path

def remove_html_comment(text):
    cleaned_text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
    return cleaned_text

def remove_tables(md_text: str) -> str:
    table_pattern = re.compile(r"(^\|.*\|$\n(^\|[-:]+\|$\n)?(\|.*\|$\n)*)", re.MULTILINE)
    cleaned_text = re.sub(table_pattern, '', md_text)
    return cleaned_text.strip()

def extract_html_tables(html):
    table_regex = re.compile(r'<table[\s\S]*?</table>', re.IGNORECASE)
    tables = table_regex.findall(html)
    return tables


class GeminiOCR:
    def __init__(self, max_tokens=2000, model_name="gemini-2.0-flash"):
        self.max_tokens = max_tokens
        self.tmp = f"{os.getcwd()}/geminitmp"
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self.model_name = model_name
        os.makedirs(self.tmp, exist_ok=True)

    def _answer(self, prompt: str, image: Image) -> str:
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[prompt, image])
        out_text = response.text
        print(out_text)
        return out_text

    def __call__(self, prompt: str, pdf_path: str):
        images = convert_from_path(pdf_path)
        label_path = pdf_path.replace(".pdf", ".md").replace("pdfs/", "labels/")
        pred_str = ""
        for idx, img in enumerate(images):
            try:
                pred = self._answer(prompt, img)
                pred = pred.replace("```markdown", "").replace("```html", "").replace("```", "")
                pred = remove_html_comment(pred)
            except Exception as e:
                print(f"Skipping sample {pdf_path}:{idx} due to error: {e}")
                pred = ""
            pred_str += pred
        pred_text, pred_tables = remove_tables(pred_str), extract_html_tables(pred_str)
        return pred_text, pred_tables