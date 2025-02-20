from PIL import Image
from openai import OpenAI
import base64
import io

class GPTModel:
    def encode_image_bytes(self, image_bytes: bytes) -> str:
        return base64.b64encode(image_bytes).decode("utf-8")

    def _get_messages(self, image: Image, prompt):
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        buf.seek(0)
        image_bytes = buf.read()
        image_base64 = self.encode_image_bytes(image_bytes)

        return [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

    def __call__(self, prompt, image):
        messages = self._get_messages(image, prompt)
        client = OpenAI()
        result = client.chat.completions.create(messages=messages, model="gpt-4o")
        out_text = result.choices[0].message.content
        print(out_text)
        return out_text
