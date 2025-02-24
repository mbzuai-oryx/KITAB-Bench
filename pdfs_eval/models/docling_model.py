from docling.document_converter import DocumentConverter, FormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions, TableStructureOptions, TesseractCliOcrOptions
from docling.backend.docling_parse_v2_backend import DoclingParseV2DocumentBackend
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from docling_core.types.doc import TableItem
import re


def remove_tables(md_text: str) -> str:
    table_pattern = re.compile(r"(^\|.*\|$\n(^\|[-:]+\|$\n)?(\|.*\|$\n)*)", re.MULTILINE)
    cleaned_text = re.sub(table_pattern, '', md_text)
    return cleaned_text.strip()

class DoclingPDF:
    def __init__(self, **kwargs):
        if kwargs.get("model_type", "easyocr") == "easyocr":
            ocr_options = EasyOcrOptions(
                use_gpu=True,
                lang=['ar'],
            )
        else:
            ocr_options = TesseractCliOcrOptions(
                lang=['ara'],
            )
        table_structure_options = TableStructureOptions(do_cell_matching=True, model="accurate")
        pipeline_options = PdfPipelineOptions(
            do_ocr=True,               
            ocr_options=ocr_options,    
            force_full_page_ocr=True,
            table_structure_options=table_structure_options,
        )
        format_option = FormatOption(
            pipeline_cls=StandardPdfPipeline,
            pipeline_options=pipeline_options,
            backend=DoclingParseV2DocumentBackend,
        )
        self.converter = DocumentConverter(
            format_options={"pdf": format_option}
        )

    def __call__(self, _, source: str):
        result = self.converter.convert(source)
        text = remove_tables(str(result.document.export_to_markdown(image_placeholder="")))
        tables = [item.export_to_html() for item, level in result.document.iterate_items() if isinstance(item, TableItem)]
        return text, tables