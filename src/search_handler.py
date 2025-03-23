# search_handler.py
import requests
import jieba
import jieba.posseg as pseg
from dotenv import load_dotenv
import os

load_dotenv()  # 載入 .env 文件中的變數

class SearchHandler:
    API_KEY = os.getenv("API_KEY")
    SEARCH_ENGINE_ID = os.getenv("SEARCH_ENGINE_ID")
    BASE_URL = "https://www.googleapis.com/customsearch/v1"

    def extract_keywords(self, question):
        # 分詞並標記詞性
        words = pseg.cut(question)
        # n：普通名詞（例如「總統」「立場」）
        # nr：人名（例如「唐納德·川普」）
        # ns：地名（例如「美國」「台北」）
        # nt：機構名（例如「共和黨」）
        # t：時間詞（例如「現在」「今天」「2025年」）
        # v：動詞（例如「是」「有」），可能用於語意補充。
        # m：數詞（例如「幾歲」），可能與年齡相關。
        allowed_pos = {'n', 'nr', 'ns', 'nt', 't'}
        keywords = [word for word, flag in words if flag in allowed_pos]
        # 如果沒有名詞，fallback 到完整問題的前幾個字
        if not keywords:
            keywords = [question[:5]]  # 取前 5 個字作為備案
        # 拼接關鍵字，用空格分隔
        return " ".join(keywords)

    def resultAnalysis(self, query):
        print("有call resultAnalysis")
        # Call Search
        results = self.search(query)
        if isinstance(results, list):
            # 格式化結果為列表，每個元素是一個字典
            formatted_results = []
            for i, result in enumerate(results, 1):
                formatted_result = {
                    "結果": f"結果 {i}",
                    "標題": result["title"],
                    "連結": result["link"],
                    "摘要": result["snippet"]
                }
                formatted_results.append(formatted_result)
                # 保留 print 以便除錯
                print(f"結果 {i}:")
                print(f"標題: {result['title']}")
                print(f"連結: {result['link']}")
                print(f"摘要: {result['snippet']}")
                print("---")
            return formatted_results
        else:
            print(results)  # 印出錯誤訊息
            return results

    # Do Search
    def search(self, query, num_results=3): # 三個結果
        keywords = self.extract_keywords(query)
        print("keywords:"+keywords)
        params = {
            "key": self.API_KEY,
            "cx": self.SEARCH_ENGINE_ID,
            "q": keywords,
            "num": num_results
        }
        try:
            response = requests.get(self.BASE_URL, params=params)
            response.raise_for_status()
            results = response.json().get("items", [])
            return [{"title": item["title"], "link": item["link"], "snippet": item["snippet"]} for item in results]
        except requests.RequestException as e:
            return f"搜尋失敗：{str(e)}"

    
