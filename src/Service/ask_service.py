import time
import numpy as np
from PIL import Image

class AskService:
    def __init__(self, blip_processor , ocr_processor , ollama_handler):
        self.blip_processor = blip_processor
        self.ocr_processor = ocr_processor
        self.ollama_handler = ollama_handler

    def analyze_image(self, image_file):
        try:
            image = Image.open(image_file.stream).convert("RGB")
            image_np = np.array(image)
            success, ocr_description = self.ocr_processor.process_ocr(image_np)
            if not success:
                ocr_description = None
            blip_description = self.blip_processor.blip_analyze(image)
            return ocr_description, blip_description , False # 返回結果
        except Exception as e:
            print(f"AskService圖片處理失敗：{str(e)}")
            return None, None, True  # 錯誤時返回 True 讓 ask_service_result 觸發error

    def answer_question(self, question_request , ocr_description , blip_description , is_deepsearch):
        full_response = ""
        try:
            if question_request:    # 有輸入問題
                if blip_description:
                    for chunk in self.ollama_handler.ask_llama(question_request , is_deepsearch , ocr_description , blip_description):
                        full_response +=chunk
                        yield f"data: {chunk}\n\n"
                        time.sleep(0.05)
                else:
                    for chunk in self.ollama_handler.ask_llama(question_request , is_deepsearch):
                        full_response += chunk
                        yield f"data: {chunk}\n\n"
                        time.sleep(0.05)
            else:
                full_response = blip_description or ""
            yield "data: [DONE]\n\n"
        finally:
            print(f"Bot Response: {full_response}")
