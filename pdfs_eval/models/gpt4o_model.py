import base64
import io
from PIL import Image
from openai import OpenAI
import re
from pdf2image import convert_from_path


def remove_tables(md_text: str) -> str:
    table_pattern = re.compile(r"(^\|.*\|$\n(^\|[-:]+\|$\n)?(\|.*\|$\n)*)", re.MULTILINE)
    cleaned_text = re.sub(table_pattern, '', md_text)
    return cleaned_text.strip()

def remove_html_comment(text):
    cleaned_text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
    return cleaned_text

def extract_html_tables(html):
    table_regex = re.compile(r'<table[\s\S]*?</table>', re.IGNORECASE)
    tables = table_regex.findall(html)
    return tables


class GPT4oOCR:
    def __init__(self, **kwargs):
        self.max_tokens = kwargs["max_tokens"]
        self.model_name = kwargs["model_name"]

    def _enc(self, image_bytes: bytes) -> str:
        return base64.b64encode(image_bytes).decode("utf-8")

    def _get_messages(self, image: Image, prompt):
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        buf.seek(0)
        image_bytes = buf.read()
        image_base64 = self._enc(image_bytes)

        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

    def _answer(self, prompt, image):
        messages = self._get_messages(image, prompt)
        client = OpenAI()
        result = client.chat.completions.create(
            messages=messages,
            model=self.model_name,
            max_tokens=self.max_tokens,
        )
        out_text = result.choices[0].message.content
        print(out_text)
        return out_text

    

    def __call__(self, prompt: str, pdf_path: str):
        images = convert_from_path(pdf_path)
        pred_str = ""
        for idx, img in enumerate(images):
            try:
                pred = self._answer(prompt, img)
                pred = pred.split("```html")[-1].split("```")[0]
                pred = remove_html_comment(pred)
            except Exception as e:
                print(f"Skipping sample {pdf_path}:{idx} due to error: {e}")
                pred = ""
            pred_str += pred
        pred_text, pred_tables = remove_tables(pred_str), extract_html_tables(pred_str)
        return pred_text, pred_tables
