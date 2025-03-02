import os
from PIL import Image

class Qwen25VLOCR:
    def __init__(self, model_name: str="Qwen/Qwen2.5-VL-7B-Instruct", max_tokens: int=2000, use_flash_attn: bool=False):
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        import torch
        from qwen_vl_utils import process_vision_info
        if use_flash_attn:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="auto",
            )
        else:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="auto"
            )
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.max_tokens = max_tokens

    def __call__(self, prompt: str, image: Image) -> str:
        src = "qwen25vl_image1.png"
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