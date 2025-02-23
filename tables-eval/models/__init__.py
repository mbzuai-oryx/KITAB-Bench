from .docling_model import DoclingOCR
from .gemini_model import GeminiOCR
from .img2table_model import Img2tableOCR
from .marker_model import MarkerOCR
from .qwen2vl_model import Qwen2VLOCR
from .qwen25vl_model import Qwen25VLOCR
from .gpt4o_model import GPT4oOCR


AVAILABLE_MODELS = ["doclingtesseract", "doclingeasyocr", "gemini", "img2tableeasyocr", "img2tabletesseract", "marker", "gpt-4o", "gpt-4o-mini", "qwen2vl", "qwen25vl"]