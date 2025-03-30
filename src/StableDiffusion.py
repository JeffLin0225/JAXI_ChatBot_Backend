from diffusers import StableDiffusionPipeline
import configparser
import base64
from io import BytesIO
import torch
import gc

config = configparser.ConfigParser()
config.read("config.ini")

class StableDiffusion:

    '''初始化'''
    def __init__(self):
        # config.ini
        self.pipe_negative_prompt = config["StableDiffusion"]["pipe_negative_prompt"]
        self.pipe_height = int(config["StableDiffusion"]["pipe_height"])
        self.pipe_width = int(config["StableDiffusion"]["pipe_width"])
        self.pipe_num_inference_steps = int(config["StableDiffusion"]["pipe_num_inference_steps"])
        self.stableDiffusionModel_Path = config["StableDiffusion"]["stableDiffusionModel_Path"]
        
        # 初始 pipe 設為 None
        self.pipe = None

    '''生成,儲存 圖像'''
    def generate_image(self , blip_prompt):
        try:
            # 如果模型未加載，則進行加載
            if self.pipe is None:
                self.pipe = StableDiffusionPipeline.from_pretrained(        
                    self.stableDiffusionModel_Path,
                    low_cpu_mem_usage=True                                  # 優化 CPU 記憶體使用
                )
                self.pipe = self.pipe.to("mps")  # 移動模型到 MPS 計算設備
                self.pipe.enable_attention_slicing()  # 啟用注意力切片技術

            # 生成圖像
            image = self.pipe(
                        prompt=blip_prompt, 
                        negative_prompt=self.pipe_negative_prompt,
                        height=self.pipe_height,
                        width=self.pipe_width, 
                        num_inference_steps=self.pipe_num_inference_steps
                    ).images[0]
            image.save("output3.png")
            image.show()
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            base64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            return base64_str
        finally:
            self.clean()
        
    '''清理記憶體'''
    def clean(self):
        gc.collect()  # 釋放未使用的物件
        torch.mps.empty_cache()  # 清除 Metal Performance Shaders 內存快取
