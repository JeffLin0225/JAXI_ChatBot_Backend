from selenium import webdriver
from bs4 import BeautifulSoup
from urllib.parse import quote

# 配置 Chrome 選項
options = webdriver.ChromeOptions()
options.add_argument('--headless')  # 無頭模式（理論上不彈窗）
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')

# 初始化瀏覽器
driver = webdriver.Chrome(options=options)

# 生成搜尋 URL
query = "美國總統是誰"
encoded_query = quote(query)
url = f"https://www.google.com/search?q={encoded_query}"

# 獲取頁面
driver.get(url)
html = driver.page_source
soup = BeautifulSoup(html, 'html.parser')

# 解析特色片段
snippet = soup.find('div', class_='BNeawe')
answer = snippet.text if snippet else "未找到直接答案"
print(f"答案: {answer}")

# 關閉瀏覽器
driver.quit()