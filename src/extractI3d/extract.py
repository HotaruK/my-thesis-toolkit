import torch
import cv2
import os
from pytorchi3d import InceptionI3d
import gzip
import pickle
from tqdm import tqdm


def extract_features(base_dir, video_dir, model_path, device, origin_data):
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

    frames = frames.permute(3, 0, 1, 2).unsqueeze(0).float()
    frames = frames.to(device)

    features = model.extract_features(frames).permute(0, 2, 1, 3, 4).contiguous().view(-1, 1024)
    features = features.to('cpu')

    output_dir = os.path.join(base_dir, 'output')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    op = {
        'name': origin_data['name'],
        'signer': origin_data['signer'],
        'gloss': origin_data['gloss'],
        'text': origin_data['text'],
        'sign': features,
    }

    output_file = os.path.join(output_dir, f'{video_dir}.pt')
    torch.save(op, output_file)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device={device}')

    base_dir = ''
    model_path = ''
    original_pklgz = ''
    prefix = ''

    dataset = None
    with gzip.open(original_pklgz, 'rb') as f:
        dataset = pickle.load(f)

    video_dirs = os.listdir(base_dir)
    video_dirs = [video_dir for video_dir in video_dirs if os.path.isdir(os.path.join(base_dir, video_dir))]

    for video_dir in tqdm(video_dirs):
        origin = None
        for i in dataset:
            if i.get('name') == prefix + video_dir:
                origin = i
        extract_features(base_dir=base_dir, video_dir=video_dir, model_path=model_path, device=device,
                         origin_data=origin)
