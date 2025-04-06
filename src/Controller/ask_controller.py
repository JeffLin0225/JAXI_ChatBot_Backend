from flask import Blueprint , request , Response , jsonify , stream_with_context
from src.Service import AskService
from src.ModelService import BLIPProcessor, OCRProcessor, OllamaHandler

jaxi_bp = Blueprint('jaxiapi' , __name__)

try:
    blip_processor = BLIPProcessor()
    ocr_processor = OCRProcessor()
    ollama_handler = OllamaHandler()
    ask_service = AskService(blip_processor , ocr_processor , ollama_handler)
except Exception as e:
    print(f"ask_controller 初始化失敗：{str(e)}")
    exit(1)

@jaxi_bp.route('/ask', methods=['POST'])
def ask_controller():
    # 取得圖片和問題
    image_request = request.files.get('image')
    question_request = request.form.get('question' , '').strip()
    is_deepsearch_request = request.form.get('isDeepSearch' , '').lower() == 'true'
    print(f"User Question: {question_request} , isDeepSearch:{is_deepsearch_request}")  # 監控：顯示問題

    # 如果沒有圖片跟問題
    if not image_request and not question_request:
        return jsonify({"error": "請輸入文字或是圖片"}), 400

    # 如果圖片不是JPEG 或 PNG
    if image_request and image_request.content_type not in ['image/jpeg', 'image/png']:
        return jsonify({"error": "僅支援 JPEG 或 PNG 圖片"}), 400

    blip_description = None
    ocr_description = None

    # 如果有 輸入圖片
    if image_request:
        result = ask_service.analyze_image(image_request) #丟進去 ask_service
        ocr_description , blip_description , is_analyze_error = result
        if is_analyze_error:
            return jsonify({"error" : "AskService圖片處理失敗"}) ,400

    # 使用 SSE 格式返回流式回應
    return Response(
        stream_with_context(
            ask_service.answer_question(question_request , ocr_description , blip_description , is_deepsearch_request)
        ),
        mimetype='text/event-stream'
    )

'''
錯誤處理 (413 圖片超出規範大小)
'''
@jaxi_bp.errorhandler(413)
def request_img_file_too_large(error):
    print("檔案大小超過 10MB 限制")
    return jsonify({"error": "檔案大小超過 10MB 限制"}), 413
