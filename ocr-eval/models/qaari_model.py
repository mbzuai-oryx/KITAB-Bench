import os
from PIL import Image



class QaariOCR:
    def __init__(self, model_name: str="NAMAA-Space/Qari-OCR-0.1-VL-2B-Instruct", max_tokens: int=2000, use_flash_attn: bool=False):
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        import torch
        if use_flash_attn:
            print("Inferencing with flash attention ...")
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

    def __call__(self, _: str, image: Image) -> str:
        from qwen_vl_utils import process_vision_info
        src = "qaari_image1.png"
        image.save(src)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{src}"},
                    {"type": "text", "text": "Below is the image of one page of a document, as well as some raw textual content that was previously extracted for it. Just return the plain text representation of this document as if you were reading it naturally. Do not hallucinate."},
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
        generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_tokens, use_cache=True)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        os.remove(src)
        print(output_text)
        return output_text
        
