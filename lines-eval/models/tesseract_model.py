import pytesseract

class TesseractOCR:
    def __call__(self, img):
        w, h = img.size
        pred = pytesseract.image_to_data(img, lang='ara', output_type="dict")
        data =  []
        for idx, word in enumerate(pred['text']):
            conf = pred['conf'][idx]
            if conf <= 0: continue
            top = pred['top'][idx]
            left = pred['left'][idx]
            width = pred['width'][idx]
            height = pred['height'][idx]
            x1, y1, x2, y2 = left, top, left + width, top + height
            data.append({"bbox": [x1, y1, x2, y2], 'text': word, 'conf': conf / 100})
        return {"width": w, "height": h, "lines": data}