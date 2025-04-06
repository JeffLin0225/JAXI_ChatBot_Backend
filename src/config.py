
class Config:
    MAX_CONTENT_LENGTH = 10 * 1024 * 1024   # 10MB 限制
    JSON_AS_ASCII = False                   # 禁用 ASCII 編碼，讓中文直接輸出

    HOST = '0.0.0.0'
    PORT = 5001
    DEBUG = True