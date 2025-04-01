import base64
import io
from PIL import Image


class GPT4oOCR:
    def __init__(self, **kwargs):
        self.max_tokens = kwargs.get("max_tokens", 2000)
        self.model_name = kwargs.get("model_name", "gpt-4o-mini")

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

    def __call__(self, prompt, image):
        from openai import OpenAI
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
