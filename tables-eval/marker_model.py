from marker.converters.table import TableConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered


class MarkerOCR:
    def __init__(self, **kwargs):
        self.converter = TableConverter(
            artifact_dict=create_model_dict(), 
            renderer="marker.renderers.html.HTMLRenderer",
        )
        self.index = 0
    def __call__(self, _, image) -> str:
        source = f"/home/omkar/tables-eval/qwentmp/{self.index}.png"
        image.save(source)
        out = self.converter(source).html
        self.index += 1
        print(out)
        return out
        