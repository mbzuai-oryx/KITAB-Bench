from marker.converters.pdf import PdfConverter
from marker.config.parser import ConfigParser
from marker.models import create_model_dict
import re

def remove_tables(md_text: str) -> str:
    table_pattern = re.compile(r"(^\|.*\|$\n(^\|[-:]+\|$\n)?(\|.*\|$\n)*)", re.MULTILINE)
    cleaned_text = re.sub(table_pattern, '', md_text)
    return cleaned_text.strip()


def extract_html_tables(html):
    table_regex = re.compile(r'<table[\s\S]*?</table>', re.IGNORECASE)
    tables = table_regex.findall(html)
    return tables

class MarkerPDF:
    def __init__(self):
        html_config = ConfigParser({"output_format": "html"})
        self.html_converter = PdfConverter(
            config=html_config.generate_config_dict(),
            artifact_dict=create_model_dict(),
            processor_list=html_config.get_processors(),
            renderer=html_config.get_renderer(),
        )
        
        md_config = ConfigParser({"output_format": "markdown"})
        self.md_converter = PdfConverter(
            config=md_config.generate_config_dict(),
            artifact_dict=create_model_dict(),
            processor_list=md_config.get_processors(),
            renderer=md_config.get_renderer(),
        )
    
    def __call__(self, _, source: str):
        md_code = self.md_converter(source).markdown
        text = remove_tables(str(md_code))
        html_code = self.html_converter(source).html
        tables = extract_html_tables(str(html_code))
        return text, tables