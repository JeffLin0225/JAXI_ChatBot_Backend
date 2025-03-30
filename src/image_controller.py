from flask import Flask, Blueprint, jsonify ,request
from StableDiffusion import StableDiffusion
from PIL import Image
from blip_processor import BLIPProcessor

'''測試檔案'''

# 初始化 Flask
app = Flask(__name__)

# 建立 StableDiffusion 服務實例
stable_diffusion_service = StableDiffusion()
blip_processor = BLIPProcessor()

# 建立 Flask Blueprint
image_bp = Blueprint("image_bp", __name__)

@image_bp.route("/generate", methods=["POST"])
def generate():

    image_file = request.files.get('image')

    image = Image.open(image_file.stream).convert("RGB")
    blipDescription = blip_processor.blip_analyze(image)

    try:
        base64_img = stable_diffusion_service.generate_image(blipDescription)  # ✅ 正確用法
        return jsonify({"image": base64_img})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 註冊藍圖
app.register_blueprint(image_bp, url_prefix="/api")

if __name__ == "__main__":
    app.run(debug=True, port=5002)