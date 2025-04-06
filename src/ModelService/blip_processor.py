from transformers import BlipProcessor, BlipForConditionalGeneration
from deep_translator import GoogleTranslator

class BLIPProcessor:
    
    def __init__(self):
        self.google_translator = GoogleTranslator(source='auto', target='zh-TW')
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base",
            use_safetensors=True
        )

    def blip_analyze(self , image):
        inputs = self.processor(images=image, return_tensors="pt")
        generated_ids = self.model.generate(**inputs)
        description = self.processor.decode(generated_ids[0], skip_special_tokens=True)
        # description_tw = self.google_translator.translate(description) # 同步翻譯
        return description
