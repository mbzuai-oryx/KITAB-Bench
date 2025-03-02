from PIL import Image
import os
import multiprocessing as mp


class PaddleGPUOCR:
    def __init__(self, max_tokens=2000):
        from paddleocr import PaddleOCR
        self.max_tokens = max_tokens
        self.tmp = f"{os.getcwd()}/paddletmp"
        self.ocr = PaddleOCR(use_angle_cls=False, lang='ar', use_gpu=True, show_log=False)
        os.makedirs(self.tmp, exist_ok=True)

    def __call__(self, _: str, image: Image):
        thread_idx = mp.current_process().pid
        image_path = f"{self.tmp}/{thread_idx}.png"
        image.save(image_path)
        reg = self.ocr.ocr(image_path)[0]
        text = "\n".join([l[-1][0] for l in reg])
        print(text)
        return text
