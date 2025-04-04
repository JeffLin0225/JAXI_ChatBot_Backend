from paddleocr import PaddleOCR as PaddleOCRLib

class PaddleOCR:

    # 處理圖片的 OCR 並返回結果
    def process_ocr(self, image_path, lang='ch'):
        try:
            # 不帶參數的初始化
            ocr = PaddleOCRLib(show_log = False ,lang=lang)
            
            result = ocr.ocr(image_path)
            # 檢查 OCR 結果是否有效
            if result is None or result == [None]:
                print("圖片中沒有可辨識的文字, return False, []")
                return False, []
            
            if not result or not result[0]:
                print("OCR 結果為空或無效, return False, []")
                return False, []

            # 定義區域和對應閾值的函數
            def get_threshold(x, y):
                if y < 200: return 0.8
                elif y > 800: return 0.7
                elif x < 200: return 0.85
                elif x > 800: return 0.9
                else: return 0.75

            filtered_texts = []
            # 處理每個文字塊
            for line in result[0]:
                box = line[0]          # 文字塊的座標
                text = line[1][0]      # 識別出的文字內容
                confidence = line[1][1] # 置信度分數
                
                # 計算文字塊的中心座標
                x_center = sum([point[0] for point in box]) / 4
                y_center = sum([point[1] for point in box]) / 4
                
                # 根據中心座標獲取該區域的閾值
                threshold = get_threshold(x_center, y_center)
                
                # 篩選：只保留置信度高於閾值的文字
                if confidence > threshold:
                    filtered_texts.append(text)
            
            # print("篩選後的文字:", filtered_texts)  # 檢查最終結果
            if filtered_texts:
                return True, filtered_texts
            else:
                print("無符合閾值的文字, return False, []")
                return False, []

        except FileNotFoundError:
            print("檔案不存在, return False, []")
            return False, []
        except Exception as e:
            print(f"OCR 失敗，發生錯誤：{str(e)}, return False, []")
            return False, []