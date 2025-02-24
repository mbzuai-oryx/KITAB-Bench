import pytesseract

class TesseractOCR:
    def __call__(self, _, image):
        text = pytesseract.image_to_string(image, lang='ara')
        print(text)
        return text
