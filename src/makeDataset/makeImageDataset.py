import os
from PIL import Image
import numpy as np
import torch
import gzip
import pickle


def find_and_pop_name_by_object(objects, target_name):
    for i, obj in enumerate(objects):
        if obj.get('name') == target_name:
            return objects.pop(i)
    return None


def _pack_dataset(input_dir, output_dir, gaslt_file_path, type):
    files = [f for f in os.listdir(input_dir) if f.endswith('.npz')]

    with gzip.open(gaslt_file_path, 'rb') as f:
        i3d_ds = pickle.load(f)

    result = []
    for ff in files:
        video_file_path = os.path.join(input_dir, ff)
        file_name, _ = os.path.splitext(ff)

        i3d_obj = find_and_pop_name_by_object(i3d_ds, f'{type}/{file_name}')
        v = np.load(video_file_path, allow_pickle=True)
        i3d_obj['sign'] = torch.tensor(v)
        result.append(i3d_obj)

    with gzip.open(os.path.join(output_dir, f'img.{type}.pkl.gz'), 'wb') as f:
        pickle.dump(result, f)


def load_frame(frame_file):
    """
    https://github.com/Finspire13/pytorch-i3d-feature-extraction/blob/master/extract_features.py
    """
    data = Image.open(frame_file)

    data = data.resize((224, 224), Image.ANTIALIAS)

    data = np.array(data)
    data = data.astype(float)
    data = (data * 2 / 255) - 1

    assert (data.max() <= 1.0)
    assert (data.min() >= -1.0)

    return data


def _extract(input_dir, output_dir):
    video_names = [i for i in os.listdir(input_dir)]
    for video_name in video_names:
        save_file = f'{video_name}.npz'
        frames_dir = os.path.join(input_dir, video_name)
        rgb_files = [i for i in os.listdir(frames_dir) if i.startswith('image')]
        rgb_files.sort()

        frames = [load_frame(i) for i in rgb_files]
        np.savez(os.path.join(output_dir, save_file), frames)


if __name__ == '__main__':
    root_dir = ''
    gaslt_origin_ds_dir = ''
    video_output_root_dir = ''
    gaslt_ds_output_dir = ''
    type = ['train', 'test', 'dev']

    for t in type:
        input_dir = os.path.join(root_dir, t)
        output_dir = os.path.join(video_output_root_dir, t)
        _extract(input_dir, output_dir)
        _pack_dataset(input_dir=output_dir,
                      output_dir=gaslt_ds_output_dir,
                      gaslt_file_path=os.path.join(gaslt_origin_ds_dir, f'{t}filename'),
                      type=t)
