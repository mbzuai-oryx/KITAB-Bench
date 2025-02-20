from PIL import Image
import os
import base64
from openai import OpenAI


class Qwen2VLOCR:
    def __init__(self, max_tokens=2000):
        self.max_tokens = max_tokens
        self.tmp = f"{os.getcwd()}/qwentmp"
        self.port = os.environ.get('API_PORT', 8077)
        os.makedirs(self.tmp, exist_ok=True)

    def _enc(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def _get_messages(self, image: Image, prompt: str):
        image_path = f"{self.tmp}/image0.png"
        image.save(image_path)
        image_base64 = self._enc(image_path)

        return [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

    def __call__(self, prompt: str, image: Image):
        messages = self._get_messages(image, prompt)
        client = OpenAI(
            api_key="{}".format(os.environ.get("API_KEY", "0")),
            base_url=f"http://localhost:{self.port}/v1",
        )
        result = client.chat.completions.create(messages=messages, model="test", max_tokens=self.max_tokens)
        out_text = result.choices[0].message.content
        print(out_text)
        return out_text