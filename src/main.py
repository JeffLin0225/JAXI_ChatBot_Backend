from StableDiffusion import StableDiffusion
import base64
from io import BytesIO

'''測試檔案'''
def main():
    # 初始化 StableDiffusion 類
    stable_diffusion = StableDiffusion()

    # 呼叫 generate_image 方法來生成圖像
    generated_image = stable_diffusion.generate_image()
    
    # 顯示圖像生成成功的訊息
    if generated_image:
        print("Image generated and saved as output3.png.")
        generated_image.show()
        buffered = BytesIO()
        generated_image.save(buffered, format="PNG") 
        base64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return base64_str 
    else:
        print("Image generation failed.")

if __name__ == "__main__":
    main()