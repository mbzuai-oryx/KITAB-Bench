from img2table.ocr import TesseractOCR
from img2table.document import Image
from img2table.ocr import EasyOCR
from utils import add_tbody

class Img2tableOCR:
    def __init__(self, **kwargs):
        self.ocr = EasyOCR(lang=["ar"])
        # self.ocr = TesseractOCR(n_threads=8, lang="ara", psm=11)

    def _csv_table(self, df):
        table = df.to_csv(index=False, lineterminator='\n')
        first_row = table.split("\n")[0]
        return table.replace(first_row, "")
    
    def __call__(self, _, img, out_format="csv"):
        src = "image0.png"
        img.save(src)
        doc = Image(src, detect_rotation=False)

        tables = doc.extract_tables(ocr=self.ocr,
                                        implicit_rows=False,
                                        implicit_columns=False,
                                        borderless_tables=False,
                                        min_confidence=5)
        if out_format == "html":       
            table = "" if len(tables) == 0 else add_tbody(str(tables[0].html))
        else:
            table = "" if len(tables) == 0 else self._csv_table(tables[0].df)
        print(table)
        return table

