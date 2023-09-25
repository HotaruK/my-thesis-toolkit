import torch
import cv2
import os
from pytorchi3d import InceptionI3d
import gzip
import pickle


def extract_features(base_dir, video_dir, span, model_path, device, origin_data):
    loaded_object = torch.load(model_path)
    model = InceptionI3d(400, in_channels=3)
    model.load_state_dict(loaded_object)
    model.eval()
    model = model.to(device)

    frames = []
    video_path = os.path.join(base_dir, video_dir)
    for file in sorted(os.listdir(video_path)):
        if not file.endswith('.png'):
            continue
        frame = cv2.imread(os.path.join(video_path, file), cv2.IMREAD_COLOR)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (210, 260))
        frame = torch.from_numpy(frame)
        frames.append(frame)
    frames = torch.stack(frames)
    print(f"Shape of frames tensor: {frames.shape}")

    frames = frames.permute(3, 0, 1, 2).unsqueeze(0).float()
    frames = frames.to(device)
    features = model(frames)

    output_dir = os.path.join(base_dir, 'output')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        os.makedirs(output_dir + f'\\span={span}')

    op = {
        'name': origin_data['name'],
        'signer': origin_data['signer'],
        'gloss': origin_data['gloss'],
        'text': origin_data['text'],
        'sign': features,
    }

    output_file = os.path.join(output_dir, f'span={span}\\{video_dir}.pt')
    torch.save(op, output_file)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device={device}')

    base_dir = ''
    model_path = ''
    original_pklgz = ''
    prefix = 'dev/'

    dataset = None
    with gzip.open(original_pklgz, 'rb') as f:
        dataset = pickle.load(f)

    for video_dir in os.listdir(base_dir):
        if not os.path.isdir(os.path.join(base_dir, video_dir)):
            continue
        origin = None
        for i in dataset:
            if i.get('name') == prefix + video_dir:
                origin = i
        extract_features(base_dir=base_dir, video_dir=video_dir, span=8, model_path=model_path, device=device,
                         origin_data=origin)
