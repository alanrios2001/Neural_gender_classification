import torch
from torchvision import transforms
from PIL import Image
from model_builder import GenderModel
import typer
import json

'''
uso:
python predict.py "model_dir" "img_path"
'''

app = typer.Typer()

img_transform = transforms.Compose([
    transforms.Resize(size=(128, 128)),
    transforms.ToTensor()
])


@app.command()
def predict(model_dir: str, img_path: str):
    model_path = model_dir + "model.pth"
    with open(model_dir+"metrics.json") as file:
        metrics = json.load(file)

    classes = metrics["classes"]
    model = GenderModel(3, 20, 1)
    model.load_state_dict(torch.load(model_path))

    image = Image.open(img_path)
    img_transformed = img_transform(image).unsqueeze(dim=0)

    with torch.inference_mode():
        pred = model(img_transformed)
        pred = int(torch.round(torch.sigmoid(pred)))

    print(f"pred: {classes[pred]}")


if __name__ == "__main__":
    app()
