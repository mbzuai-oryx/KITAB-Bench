from PIL import Image
import time
import os

class AzureOCR:
    min_pixels = 50
    def __init__(self):
        from azure.cognitiveservices.vision.computervision import ComputerVisionClient
        
        from msrest.authentication import CognitiveServicesCredentials
        subscription_key = os.environ["VISION_KEY"]
        endpoint = os.environ["VISION_ENDPOINT"]
        self.client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))
    
    def _resize(self, w, h):
        if w >= self.min_pixels and h >= self.min_pixels:
            return w, h
        if h < w:
            aspect = w / h
            nh = max(self.min_pixels, h)
            nw = max(int(aspect * nh), w)
        else:
            aspect = h / w
            nw = max(self.min_pixels, w)
            nh = max(int(aspect * nw), h)
        return nw, nh

    def __call__(self, _: str, image: Image) -> str:
        from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
        src = "azure_image1.png"
        w, h = image.width, image.height
        nw, nh = self._resize(w, h)
        image = image.resize((nw, nh), resample=Image.LANCZOS)
        image.save(src)
        with open(src, "rb") as image_stream:
            read_response = self.client.read_in_stream(image_stream, raw=True)

        read_operation_location = read_response.headers["Operation-Location"]
        operation_id = read_operation_location.split("/")[-1]

        while True:
            read_result = self.client.get_read_result(operation_id)
            if read_result.status not in ['notStarted', 'running']:
                break
            time.sleep(1)

        output_text = ""
        if read_result.status == OperationStatusCodes.succeeded:
            for text_result in read_result.analyze_result.read_results:
                for line in text_result.lines:
                    output_text += line.text
                    # print(line.bounding_box)

        os.remove(src)
        print(output_text)
        return output_text
        