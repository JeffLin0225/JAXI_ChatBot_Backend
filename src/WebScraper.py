import undetected_chromedriver as uc
import time
import random
import json

def google_search_undetected(query, num_pages=2):
    results = []
    
    options = uc.ChromeOptions()
    options.add_argument("--disable-blink-features=AutomationControlled")
    
    driver = uc.Chrome(options=options)
    
    try:
        # 訪問 Google
        driver.get("https://www.google.com")
        time.sleep(random.uniform(2, 4))
        
        # 輸入搜索關鍵詞
        search_box = driver.find_element("name", "q")
        search_box.send_keys(query)
        search_box.submit()
        
        time.sleep(random.uniform(3, 5))
        
        # 處理多頁結果
        for page in range(num_pages):
            # 獲取搜索結果
            search_results = driver.find_elements("css selector", ".g")
            
            for result in search_results:
                try:
                    title_element = result.find_element("css selector", "h3")
                    title = title_element.text
                    
                    link_element = result.find_element("css selector", "a")
                    link = link_element.get_attribute("href")
                    
                    # 嘗試獲取描述
                    try:
                        desc_element = result.find_element("css selector", ".VwiC3b")
                        description = desc_element.text
                    except:
                        description = "無描述"
                    
                    if title and link:
                        results.append({
                            "title": title,
                            "url": link,
                            "description": description
                        })
                except Exception as e:
                    print(f"提取結果時出錯: {e}")
            
            # 如果不是最後一頁，點擊下一頁
            if page < num_pages - 1:
                try:
                    next_button = driver.find_element("id", "pnnext")
                    next_button.click()
                    time.sleep(random.uniform(3, 5))
                except:
                    print("沒有更多頁面了")
                    break
    
    except Exception as e:
        print(f"搜索過程中發生錯誤: {e}")
    
    finally:
        driver.quit()
    
    return results

# 使用示例
# results = google_search_undetected("Python 網頁爬蟲教學")