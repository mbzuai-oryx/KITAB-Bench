from .easyocr_model import EasyOCR
from .gemini_model import GeminiOCR
from .paddle_model import PaddleGPUOCR
from .surya_model import SuryaOCR
from .tesseract_model import TesseractOCR
from .gpt4o_model import GPT4oOCR
from .qwen2vl_model import Qwen2VLOCR
from .qwen25vl_model import Qwen25VLOCR
from .arabicnougat_model import ArabicNougat
from .smoldocling_model import SmolDocling
from .qaari_model import QaariOCR
from .azure_model import AzureOCR

AVAILABLE_MODELS = ["easyocr", "gemini", "paddle", "surya", "tesseract", "gpt-4o", "gpt-4o-mini", "qwen2vl", "qwen25vl", "arabicnougat_small", "arabicnougat_base", "arabicnougat_large", "smoldocling", "qaari", "azure"]