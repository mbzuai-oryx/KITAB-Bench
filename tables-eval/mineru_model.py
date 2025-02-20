from PIL import Image
from magic_pdf.data.data_reader_writer import FileBasedDataWriter
from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
from magic_pdf.data.read_api import read_local_images

class MinerUOCR:
    def __init__(self, max_tokens=2000):
        self.max_tokens = max_tokens
        self.table_predictor = TableRecPredictor()

    def __call__(self, _: str, image: Image) -> str:
        preds = self.table_predictor([image])[0]
        print(preds)
        return ""

table_pred = TableRecPredictor()
img = Image.open("1.png")
print(table_pred([img])[0])