from docling.document_converter import DocumentConverter, FormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions, TableStructureOptions, TesseractOcrOptions
from docling.backend.docling_parse_v2_backend import DoclingParseV2DocumentBackend
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from docling_core.types.doc import TableItem

class DoclingOCR:
    def __init__(self, **kwargs):
        # ocr_options = EasyOcrOptions(
        #     use_gpu=True,
        #     lang=['ar'],
        # )
        ocr_options = TesseractOcrOptions(
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
            format_options={"image": format_option}
        )
    def __call__(self, _, img, out_format="html") -> str:
        source = "1.png"
        img.save(source)
        result = self.converter.convert(source)
        if out_format == "html":
            table = str(result.document.export_to_html())
        else:
            df = [item.export_to_dataframe() for item, level in result.document.iterate_items() if isinstance(item, TableItem)][0]
            table = df.to_csv(index=False, lineterminator='\n')
        print(table)
        return table
