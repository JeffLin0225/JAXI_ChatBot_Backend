import requests  

class LlamaHandler:
    OLLAMA_API_URL = "http://localhost:11434/api/generate"

    def ask_llama(self , question, description = None ):
        if description:
            prompt = f"圖片描述：{description}\n問題：{question}\n只能用繁體中文自然回答問題。"
        else:
            prompt = f"問題：{question}\n只能用繁體中文自然回答問題。"
        try:
            response = requests.post(
                self.OLLAMA_API_URL,
                json={
                    "model": "llama3.2:3B",
                    "prompt": prompt,
                    "stream": False
                }
            )
            response.raise_for_status()
            return response.json()["response"].strip()
        except requests.RequestException as e:
            return f"錯誤：無法連接到 Llama 3.2 - {str(e)}"
    