

class SuryaOCR:
    def __init__(self):
        from surya.recognition import RecognitionPredictor
        from surya.detection import DetectionPredictor
        self.langs = ['ar']
        self.recognition_predictor = RecognitionPredictor()
        self.detection_predictor = DetectionPredictor()
    def __call__(self, image):
        pred = self.recognition_predictor([image], [self.langs], self.detection_predictor)[0]
        lines = pred.text_lines
        lines = [{'bbox': l.bbox, 'text': l.text, 'conf': l.confidence} for l in lines]
        w, h = pred.image_bbox[-2:]
        return {"width": w, "height": h, 'lines': lines}
