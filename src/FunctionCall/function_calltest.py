import json

# 定義一個處理類
class FunctionHandler:
    def get_weather(self, city):
        return f"{city} 的天氣是晴天"
    def search_web(self, query):
        return f"搜索結果: {query}"

# 實例化
handler = FunctionHandler()

# 模擬 LLM 輸出的 JSON
json_output = '{"function_name": "search_web", "arguments": {"query": "Beijing"}}'
parsed = json.loads(json_output)
func_name = parsed["function_name"]
args = parsed["arguments"]

# 用 getattr() 動態獲取方法並執行
func = getattr(handler, func_name)
result = func(**args)
print(result)  # 輸出: Beijing 的天氣是晴天