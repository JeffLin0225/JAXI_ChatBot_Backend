from transformers import BlipProcessor, BlipForConditionalGeneration
from googletrans import Translator

class BLIPProcessor:
    
    def __init__(self):
        self.translator = Translator()   
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base",
            use_safetensors=True
        )
        self.translator = Translator()

    def blip_analyze(self , image):
        inputs = self.processor(images=image, return_tensors="pt")
        generated_ids = self.model.generate(**inputs)
        description = self.processor.decode(generated_ids[0], skip_special_tokens=True)
        description_tw = self.translator.translate(description, dest="zh-tw").text  # 同步翻譯
        return description 
