from PIL import Image
from torchvision import transforms
from torchvision import models
import os
import torch
from tqdm import tqdm


def _extract(file_name):
    image = Image.open(file_name)

    preprocess = transforms.Compose([
        transforms.Resize((210, 260)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)

    model = models.resnet50(pretrained=True)
    model.eval()

    # Use GPU if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    output = model(input_batch)

    return output.squeeze()


def _process_video(video_dir, output_dir_root):
    # this function process each video frames

    # read all .png files in video_dir
    files = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith('.png')]
    result = []
    for f in files:
        output = _extract(f)
        result.append(output)

    # marge all tensors in result and make one (1, <count_of_files>, 1000) tensor
    marged_result = torch.stack(result).unsqueeze(0)

    # export marged_result named <video_dir>.pt
    output_file_path = os.path.join(output_dir_root, os.path.basename(video_dir) + '.pt')
    torch.save(marged_result, output_file_path)


if __name__ == '__main__':
    input_base_dir = ''
    output_base_dir = ''
    train_dir = input_base_dir + '/train'
    test_dir = input_base_dir + '/test'
    dev_dir = input_base_dir + '/dev'
    output_train_dir = output_base_dir + '/train'
    output_test_dir = output_base_dir + '/test'
    output_dev_dir = output_base_dir + '/dev'

    for f in tqdm(os.listdir(train_dir), desc="Processing train videos"):
        video_p = os.path.join(train_dir, f)
        _process_video(video_p, output_train_dir)

    for f in tqdm(os.listdir(test_dir), desc="Processing test videos"):
        video_p = os.path.join(test_dir, f)
        _process_video(video_p, output_test_dir)

    for f in tqdm(os.listdir(dev_dir), desc="Processing dev videos"):
        video_p = os.path.join(dev_dir, f)
        _process_video(video_p, output_dev_dir)
