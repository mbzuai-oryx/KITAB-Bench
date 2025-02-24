import os
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
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

class Qwen2VLOCR:
    def __init__(self, model_name: str="Qwen/Qwen2-VL-7B-Instruct", max_tokens: int=2000, use_flash_attn: bool=False):
        if use_flash_attn:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="auto",
            )
        else:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="auto"
            )
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.max_tokens = max_tokens

    def _answer(self, prompt: str, image: Image) -> str:
        src = "qwen2vl_image1.png"
        image.save(src)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{src}"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        os.remove(src)
        print(output_text)
        return output_text

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
        