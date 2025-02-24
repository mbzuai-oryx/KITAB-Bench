from .docling_model import DoclingPDF
from .gemini_model import GeminiOCR
from .marker_model import MarkerPDF
from .gpt4o_model import GPT4oOCR

AVAILABLE_MODELS = ['gemini', "doclingeasyocr", "doclingtesseract", "marker", "gpt-4o", "gpt-4o-mini", "qwen2vl", "qwen25vl"]