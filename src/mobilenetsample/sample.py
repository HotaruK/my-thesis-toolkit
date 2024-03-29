from PIL import Image
from torchvision import transforms
import torch
from torchvision import models
import os

mobilenet = models.mobilenet_v3_small(pretrained=True)
mobilenet.train()
features = mobilenet.features


def process_image(image_path):
    image = Image.open(image_path)

    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)

    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        mobilenet.to('cuda')

    def hook(module, input, output):
        global layer_output
        layer_output = output

    # register hook
    handle = mobilenet.features[-2].register_forward_hook(hook)

    with torch.no_grad():
        mobilenet(input_batch)
    handle.remove()

    lo = layer_output.view(1, -1)
    return lo


def process(name, root_path):
    video_path = os.path.join(root_path, name)

    files = os.listdir(video_path)
    v = []
    for file in files:
        if file.endswith('.png'):
            img_path = os.path.join(video_path, file)
            v.append(process_image(img_path))

    return torch.stack(v).permute(1, 0, 2)


if __name__ == '__main__':
    root_path = ''
    a = process('01April_2010_Thursday_heute-6694', root_path)
    print(a.shape)
