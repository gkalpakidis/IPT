import torch, requests, os
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from io import BytesIO

#Load pretrained model (DeblurGAN-v2)
def load_model():
    url = "" #URL of repo
    path = "" #Path of .pth file

    if not os.path.exists(path):
        print("Downloading model...")
        response = requests.get(url)
        with open(path, "wb") as f:
            f.write(response.content)
    
    model = torch.load(path, map_location="cpu") #"cuda"
    model.eval()
    return model

#Preprocessing image
def preprocess(img_path):
    image = Image.open(img_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return transform(image).unsqueeze(0), image

#Postprocessing image
def postprocess(tensor_img):
    tensor_img = tensor_img.squeeze().detach().cpu()
    tensor_img = (tensor_img * 0.5 + 0.5).clamp(0, 1) #Undo normalization
    return transforms.ToPILImage()(tensor_img)

#Deblurring
def deblur(model, input_img):
    with torch.no_grad():
        output = model(input_img)
    return output

#Main function