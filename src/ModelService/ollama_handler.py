import requests  
from search_handler import SearchHandler

class LlamaHandler:
    OLLAMA_API_URL = "http://localhost:11434/api/generate"

    def __init__(self):
        self.search_handler = SearchHandler()

    def ask_llama(self , question ,isDeepSearch = None, OCRDescription= None , blipDescription = None ):
        prompt = (
            "你是 JAXI，一個結合圖像分析和文字理解的智能助手。\n"
            "請優先使用你的內建知識回答問題。\n"
            "如果提供網路搜尋結果，僅在你的知識不足或需要最新資訊時作為參考；若搜尋結果明顯不相關，請忽略。\n"
        )
        
        if isDeepSearch:
            search_results = self.search_handler.resultAnalysis(question)
            if isinstance(search_results, list):
                search_str = "\n".join([f"標題：{r['標題']}\n連結：{r['連結']}\n摘要：{r['摘要']}" for r in search_results])
                prompt += f"參考資料網路搜尋結果(可以不理會)：\n{search_str}\n"
            else:
                prompt += f"網路搜尋失敗：{search_results}\n"

        if OCRDescription and blipDescription:
            prompt += f"圖片描述：{blipDescription}\n圖片分析文字資料：{OCRDescription}\n"
        elif blipDescription:
            prompt += f"圖片描述：{blipDescription}\n"  
        elif OCRDescription:
            prompt += f"圖片分析文字資料：{OCRDescription}\n"

        prompt += f"問題：{question}\n只能用繁體中文自然回答問題。"
        print(prompt )
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
            return f"錯誤：無法連接到 JAXI模型 {str(e)}" 
    