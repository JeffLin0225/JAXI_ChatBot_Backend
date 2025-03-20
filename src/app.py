from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS  # 引入 CORS
from PIL import Image
import numpy as np
import time

from blip_processor import BLIPProcessor
from llama_handler import LlamaHandler
from ocr_processor import PaddleOCR 

# 初始化
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB 限制
app.config['JSON_AS_ASCII'] = False  # 禁用 ASCII 編碼，讓中文直接輸出
CORS(app)  # 啟用 CORS，允許所有來源訪問（預設設定）

try:
    blip_processor = BLIPProcessor()
    llama_handler = LlamaHandler()
    ocr_processor = PaddleOCR()
except Exception as e:
    print(f"初始化失敗：{str(e)}")
    exit(1)

# Web api pip show flask
@app.route('/ask', methods=['POST'])
def generate_caption():

    # 取得圖片跟問題
    image_file = request.files.get('image')
    question = request.form.get('question')
    print("有打到後端")
    if not image_file and (not question or question.strip== "" ):
        return jsonify({"error" : "請輸入文字或是圖片"}),400

    if image_file and image_file.content_type not in ['image/jpeg', 'image/png']:
        return jsonify({"error": "僅支援 JPEG 或 PNG 圖片"}), 400
    
    blipDescription = None
    ocrDescription = None
    answer = None 

    if image_file: 
        try:
            image = Image.open(image_file.stream).convert("RGB")
            # imageResize = image.resize((image.width // 2, image.height // 2))  # 壓縮圖片大小
            image_np = np.array(image)
            success, ocrDescription = ocr_processor.process_ocr(image_np)
            if success is False:
                ocrDescription = None
            blipDescription = blip_processor.blip_analyze(image)
        except Exception as e:
            return jsonify({"error": f"圖片處理失敗：{str(e)}"}), 400
        
    def generate_stream():
        if question:
            if blipDescription:
                # 假設 ask_llama 返回一個生成器，逐步生成回應
                for chunk in llama_handler.ask_llama(question, ocrDescription, blipDescription):
                    yield f"data: {chunk}\n\n"
            else:
                for chunk in llama_handler.ask_llama(question):
                    yield f"data: {chunk}\n"
                    time.sleep(0.05)  # 在每個 chunk 後延遲 0.1 秒
        else:
            yield f"data: {blipDescription}\n\n"
        yield "data: [DONE]\n\n"

    # 使用 SSE 格式返回流式回應
    return Response(stream_with_context(generate_stream()), mimetype='text/event-stream')


# 錯誤處理 (413圖片超出規範大小)
@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({"error": "檔案大小超過 10MB 限制"}), 413

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)