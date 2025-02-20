from PIL import Image
import os
from google import genai
import os


class GeminiOCR:
    def __init__(self, max_tokens=2000, model_name="gemini-2.0-flash"):
        self.max_tokens = max_tokens
        self.tmp = f"{os.getcwd()}/geminitmp"
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self.model_name = model_name
        os.makedirs(self.tmp, exist_ok=True)

    def __call__(self, prompt: str, image: Image):
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[prompt, image])
        out_text = response.text
        print(out_text)
        return out_text