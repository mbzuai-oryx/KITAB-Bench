
class EasyOCR:
    def __init__(self):
        import easyocr
        self.reader = easyocr.Reader(['ar','en'])

    def _data(self, d):
        bbox = d[0]
        bbox = bbox[0] + bbox[-2]
        bbox = list(map(int, bbox))
        text = d[1]
        conf = float(d[2])
        return bbox, text, conf
    
    def __call__(self, image):
        src = "1.png"
        w, h = image.size
        image.save(src)
        result = self.reader.readtext(src)
        outputs = {"width": w, "height": h, "lines": []}
        for res in result:
            bbox, text, conf = self._data(res)    
            outputs['lines'].append({"bbox": bbox, "text": text, "conf": conf})
        return outputs
    