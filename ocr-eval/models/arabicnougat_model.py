from PIL import Image

class ArabicNougat:
    def __init__(self, model_name: str="MohamedRashad/arabic-small-nougat"):
        import torch
        from transformers import NougatProcessor, VisionEncoderDecoderModel
        self.processor = NougatProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(
            model_name, 
            torch_dtype=torch.bfloat16, 
            attn_implementation={"decoder": "flash_attention_2", "encoder": "eager"},
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.context_length = self.model.decoder.config.max_position_embeddings
        self.model = self.model.to(self.device)
        self.torch_dtype = self.model.dtype

    def __call__(self, _: str, image: Image) -> str:
        image = image.convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.to(self.torch_dtype).to(self.device)
        outputs = self.model.generate(
            pixel_values,
            repetition_penalty=1.5,
            min_length=1,
            max_new_tokens=self.context_length,
            bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
        )

        page_sequence = self.processor.batch_decode(outputs, skip_special_tokens=True)[0].replace("#", "")
        print(page_sequence)
        return page_sequence