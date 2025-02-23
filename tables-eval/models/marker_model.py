import os
from marker.converters.table import TableConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from bs4 import BeautifulSoup


def get_table(html):
  soup = BeautifulSoup(html, "html.parser")
  return soup.find("table")

class MarkerOCR:
    def __init__(self, **kwargs):
        is_html = kwargs.get("is_html", True)
        if is_html:
            self.converter = TableConverter(
                artifact_dict=create_model_dict(), 
                renderer="marker.renderers.html.HTMLRenderer",
            )
        else:
            self.converter = TableConverter(
                artifact_dict=create_model_dict(), 
            )
        self.index = 0
    def __call__(self, _, image) -> str:
        source = f"{self.index}.png"
        image.save(source)
        out = self.converter(source).html
        self.index += 1
        out = str(get_table(out))
        print(out)
        os.remove(source)
        return out
        