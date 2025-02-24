from PIL import Image
import os
import easyocr
import multiprocessing as mp


class EasyOCR:
    def __init__(self, max_tokens=2000):
        self.max_tokens = max_tokens
        self.tmp = f"{os.getcwd()}/easyocrtmp"
        self.reader = easyocr.Reader(['ar', 'en'])
        os.makedirs(self.tmp, exist_ok=True)

    def __call__(self, _: str, image: Image):
        thread_idx = mp.current_process().pid
        image_path = f"{self.tmp}/{thread_idx}.png"
        image.save(image_path)
        res = self.reader.readtext(image_path, detail=0)
        text = "\n".join(res)
        print(text)
        return text
