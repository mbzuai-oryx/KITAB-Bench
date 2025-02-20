from PIL import Image
from surya.table_rec import TableRecPredictor


class SuryaOCR:
    def __init__(self, max_tokens=2000):
        self.max_tokens = max_tokens
        self.table_predictor = TableRecPredictor()

    def __call__(self, _: str, image: Image) -> str:
        preds = self.table_predictor([image])[0]
        print(preds)
        return ""