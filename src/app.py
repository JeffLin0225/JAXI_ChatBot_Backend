from flask import Flask, request, jsonify
from PIL import Image

from blip_processor import BLIPProcessor
from llama_handler import LlamaHandler

# 初始化
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB 限制

try:
    blip_processor = BLIPProcessor()
    llama_handler = LlamaHandler()
except Exception as e:
    print(f"初始化失敗：{str(e)}")
    exit(1)

# 錯誤處理 (413圖片超出規範大小)
@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({"error": "檔案大小超過 10MB 限制"}), 413

# Web api 
@app.route('/ask', methods=['POST'])
def generate_caption():
    image_file = request.files.get('image')
    question = request.form.get('question')

    if not image_file and (not question or question.strip== "" ):
        return jsonify({"error" : "請輸入文字或是圖片"}),400

    blipDescription = None
    answer = None 

    if image_file and image_file.content_type not in ['image/jpeg', 'image/png']:
        return jsonify({"error": "僅支援 JPEG 或 PNG 圖片"}), 400
    
    if image_file: 
        try:
            image = Image.open(image_file.stream).convert("RGB")
            blipDescription = blip_processor.blip_analyze(image)
        except Exception as e:
            return jsonify({"error": f"圖片處理失敗：{str(e)}"}), 400
        
    if question:
        if blipDescription:
            answer = llama_handler.ask_llama( question , blipDescription )
        else:
            answer = llama_handler.ask_llama( question )
    
    return jsonify(
        {
            # "Description" : blipDescription if blipDescription else "無圖片描述" , 
            "answer" : answer if answer else blipDescription
        }
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)