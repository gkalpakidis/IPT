from realesrgan import RealESRGANer
from PIL import Image

def upscale(img_path, output_path):
    model = RealESRGANer(device="cuda", #"cpu"
                        scale=4,
                        model_path="/Imaging/weights/RealESRGAN_x4.pth")
    model.load_weights()
    
    image = Image.open(img_path).convert("RGB")
    upscaled = model.predict(image)
    upscaled.save(output_path)

upscale("/Imaging/pixelated.png", "/Imaging/depixel.png")