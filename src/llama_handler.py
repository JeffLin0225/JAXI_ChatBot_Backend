import requests  

class LlamaHandler:
    OLLAMA_API_URL = "http://localhost:11434/api/generate"

    def ask_llama(self , question ,OCRDescription= None , blipDescription = None ):
        if OCRDescription:
            prompt = f"圖片描述：{blipDescription}\n圖片分析文字資料：{OCRDescription}\n問題：{question}\n只能用繁體中文自然回答問題。"
        elif blipDescription:
            prompt = f"圖片描述：{blipDescription}\n問題：{question}\n只能用繁體中文自然回答問題。"
        else:
            prompt = f"問題：{question}\n只能用繁體中文自然回答問題。"

        try:
            response = requests.post(
                self.OLLAMA_API_URL,
                json={
                    "model": "gemma3:4b",
                    "prompt": prompt,
                    "stream": False
                }
            )
            response.raise_for_status()
            return response.json()["response"].strip()
        except requests.RequestException as e:
            return f"錯誤：無法連接到 JAXI模型 " 
    