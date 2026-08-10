import importlib

# Imported on first access so that running one model does not require every
# other model's dependencies to be installed (PEP 562).
_MODULE_OF = {
    "EasyOCR": ".easyocr_model",
    "GeminiOCR": ".gemini_model",
    "PaddleGPUOCR": ".paddle_model",
    "SuryaOCR": ".surya_model",
    "TesseractOCR": ".tesseract_model",
    "GPT4oOCR": ".gpt4o_model",
    "Qwen2VLOCR": ".qwen2vl_model",
    "Qwen25VLOCR": ".qwen25vl_model",
    "ArabicNougat": ".arabicnougat_model",
    "SmolDocling": ".smoldocling_model",
    "QaariOCR": ".qaari_model",
    "AzureOCR": ".azure_model",
    "AtlasOCR": ".atlasocr_model",
    "DotsOCR": ".dotsocr_model",
}


def __getattr__(name):
    if name not in _MODULE_OF:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(_MODULE_OF[name], __name__), name)


AVAILABLE_MODELS = ["easyocr", "gemini", "paddle", "surya", "tesseract", "gpt-4o", "gpt-4o-mini", "qwen2vl", "qwen25vl", "arabicnougat_small", "arabicnougat_base", "arabicnougat_large", "smoldocling", "qaari", "azure", "atlasocr", "dotsocr"]
