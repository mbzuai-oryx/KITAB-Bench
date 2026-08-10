from PIL import Image

# dots.ocr is served through its own prompt set (dots_ocr/utils/prompts.py upstream).
# "prompt_ocr" is the mode used for plain text extraction.
DOTS_OCR_PROMPT = "Extract the text content from this image."

# Matches the official vLLM recipe, which serves the model with
# --chat-template-content-format string, i.e. the image placeholder is inlined
# into the user turn before the chat template is applied.
PROMPT_TEMPLATE = "<|user|><|img|><|imgpad|><|endofimg|>{prompt}<|endofuser|><|assistant|>"


class DotsOCR:
    """dots.ocr through the vLLM engine (natively supported since vLLM 0.11).

    Requests are handed to vLLM in bulk so its continuous batching keeps the GPU
    saturated; there is no fixed batch size to tune.
    """

    def __init__(
        self,
        model_name: str = "rednote-hilab/dots.ocr",
        max_tokens: int = 2000,
        max_model_len: int = 16384,
        max_num_seqs: int = 512,
        max_num_batched_tokens: int = 16384,
        gpu_memory_utilization: float = 0.90,
        min_pixels: int = 3136,
        max_pixels: int = 4014080,
    ):
        try:
            from vllm import LLM, SamplingParams
        except ImportError as e:
            raise ImportError(
                "dots.ocr runs on vLLM, which is not in requirements.txt because it "
                "conflicts with the pinned torch/transformers versions. Install a vLLM "
                "build matching your CUDA setup (>=0.11.0, when dots.ocr was integrated) "
                "in a separate environment: pip install 'vllm>=0.11.0'"
            ) from e

        self.llm = LLM(
            model=model_name,
            trust_remote_code=True,
            dtype="bfloat16",
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            # Also caps the vision encoder cache. Raising it past what vLLM profiles
            # at startup lets the ViT activations overrun the KV cache and OOM.
            max_num_batched_tokens=max_num_batched_tokens,
            limit_mm_per_prompt={"image": 1},
            mm_processor_kwargs={"min_pixels": min_pixels, "max_pixels": max_pixels},
            mm_processor_cache_gb=0,  # every benchmark image is unique, the cache only costs memory
        )
        self.sampling_params = SamplingParams(temperature=0.1, top_p=0.9, max_tokens=max_tokens)
        self.prompt = PROMPT_TEMPLATE.format(prompt=DOTS_OCR_PROMPT)

    def batch(self, _: str, images: list) -> list:
        requests = [
            {"prompt": self.prompt, "multi_modal_data": {"image": image.convert("RGB")}}
            for image in images
        ]
        outputs = self.llm.generate(requests, self.sampling_params)
        return [output.outputs[0].text.strip() for output in outputs]

    def __call__(self, prompt: str, image: Image) -> str:
        return self.batch(prompt, [image])[0]
