from PIL import Image

class SmolDocling:
    def __init__(self, model_name: str="ds4sd/SmolDocling-256M-preview"):
        import torch
        from transformers import AutoProcessor, AutoModelForVision2Seq
        from transformers.image_utils import load_image
        from pathlib import Path

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            _attn_implementation="flash_attention_2" if self.device == "cuda" else "eager",
        )

    def __call__(self, _: str, image: Image) -> str:
        from docling_core.types.doc import DoclingDocument
        from docling_core.types.doc.document import DocTagsDocument
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Convert this page to docling."}
                ]
            },
        ]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=prompt, images=[image], return_tensors="pt")
        inputs = inputs.to(self.device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=512)
        prompt_length = inputs.input_ids.shape[1]
        trimmed_generated_ids = generated_ids[:, prompt_length:]
        doctags = self.processor.batch_decode(
            trimmed_generated_ids,
            skip_special_tokens=False,
        )[0].lstrip()

        doctags_doc = DocTagsDocument.from_doctags_and_image_pairs([doctags], [image])
        print(doctags)
        doc = DoclingDocument(name="Document")
        doc.load_from_doctags(doctags_doc)
        md = doc.export_to_markdown()
        print(md)
        return md
