import os
from PIL import Image
from unsloth import FastVisionModel
import torch

class AtlasOCR:
    def __init__(self, model_name: str="atlasia/AtlasOCR-v0", max_tokens: int=2000):
        self.model, self.processor = FastVisionModel.from_pretrained(
            model_name,
            device_map="auto",
            load_in_4bit=True,
            use_gradient_checkpointing="unsloth"
        )
        self.max_tokens = max_tokens
        self.prompt = ""
    
    def prepare_inputs(self,image:Image):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                    },
                    {"type": "text", "text": self.prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(
            image,
            text,
            add_special_tokens=False,
            return_tensors="pt",
        )
        return inputs
    
    def predict(self,image:Image) -> str:
        inputs = self.prepare_inputs(image, self.processor)
        inputs = inputs.to("cuda")

        inputs['attention_mask'] = inputs['attention_mask'].to(torch.float32)
        print("attention_mask dtype:", inputs['attention_mask'].dtype)

        generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_tokens, use_cache=True)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0]

    def __call__(self, _: str, image: Image) -> str:
        return self.predict(image)
        
