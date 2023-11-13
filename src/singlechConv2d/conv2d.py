import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
from PIL import Image
import numpy as np
import os


def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = preprocess(image)
    image = image.unsqueeze(0)
    return image


def process_image(img_path):
    model = resnet50(pretrained=True)
    model = model.eval()

    image = load_image(img_path)

    with torch.no_grad():
        features = model(image)

    features = features.view(1, 1024)
    return features


def process_dir(input_dir, output_dir):
    for video_name in os.listdir(input_dir):
        video_dir = os.path.join(input_dir, video_name)
        features = []
        for img_name in sorted(os.listdir(video_dir)):
            img_path = os.path.join(video_dir, img_name)

            img_features = process_image(img_path)
            features.append(img_features)

        features = np.concatenate(features, axis=0)
        np.savez(os.path.join(output_dir, f"{video_name}.npz"), features)


if __name__ == '__main__':
    process_dir(input_dir, output_dir)
