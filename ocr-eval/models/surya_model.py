from PIL import Image
from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor


class SuryaOCR:
    def __init__(self, max_tokens=2000):
        self.max_tokens = max_tokens
        self.recognition_predictor = RecognitionPredictor()
        self.detection_predictor = DetectionPredictor()
        self.langs = ["ar"]

    def __call__(self, _: str, image: Image) -> str:
        preds = self.recognition_predictor([image], [self.langs], self.detection_predictor)[0]
        text = "\n".join([t.text for t in preds.text_lines])
        print(text)
        return text