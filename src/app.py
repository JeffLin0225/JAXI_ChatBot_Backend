from flask import Flask, Blueprint , jsonify
from flask_cors import CORS

from config import Config
from src.Controller.ask_controller import jaxi_bp

'''
初始化 Flask 
'''
app = Flask(__name__)
app.config.from_object(Config)  # 設定檔 class Config:
CORS(app)                       # 啟用 CORS

app.register_blueprint(jaxi_bp , url_prefix =  "/jaxiapi")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)