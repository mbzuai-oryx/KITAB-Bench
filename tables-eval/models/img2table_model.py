import os

from img2table.ocr import TesseractOCR
from img2table.document import Image
from img2table.ocr import EasyOCR

def add_tbody(html):
    soup = BeautifulSoup(html, 'html.parser')
    table = soup.find('table')
    rows = table.find_all('tr')
    tbody = soup.new_tag('tbody')
    for row in rows:
        tbody.append(row.extract())
    table.append(tbody)
    return str(soup.prettify())

class Img2tableOCR:
    def __init__(self, **kwargs):
        if kwargs.get("model_type", "easyocr") == "easyocr":
            self.ocr = EasyOCR(lang=["ar"])
        else:
            self.ocr = TesseractOCR(n_threads=8, lang="ara", psm=11)

    def _csv_table(self, df):
        table = df.to_csv(index=False, lineterminator='\n')
        first_row = table.split("\n")[0]
        return table.replace(first_row, "")
    
    def __call__(self, prompt: str, img):
        src = "image0.png"
        img.save(src)
        doc = Image(src, detect_rotation=False)

        tables = doc.extract_tables(ocr=self.ocr,
                                        implicit_rows=False,
                                        implicit_columns=False,
                                        borderless_tables=False,
                                        min_confidence=5)
        if "csv" not in prompt:       
            table = "" if len(tables) == 0 else self._csv_table(tables[0].df)
        else:
            table = "" if len(tables) == 0 else add_tbody(str(tables[0].html))
        print(table)
        os.remove(src)
        return table

