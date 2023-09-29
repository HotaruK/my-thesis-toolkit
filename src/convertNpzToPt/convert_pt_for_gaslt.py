import numpy as np
import torch
import os
import gzip
import pickle
import shutil
from tqdm import tqdm


def _open_npz(filename: str):
    with np.load(filename) as data:
        f = data.f.feature
        video_name = data.f.video_name.item()
        return f, video_name


def _get_original_dataset(filename: str):
    with gzip.open(filename, 'rb') as f:
        dataset = pickle.load(f)
        v_names = {e['name']: i for i, e in enumerate(dataset)}
        return dataset, v_names


def _convert(feature, origin, output_dir, video_name):
    f_sq = np.squeeze(feature, axis=0)
    f_tensor = torch.from_numpy(f_sq)

    op = {
        'name': origin['name'],
        'signer': origin['signer'],
        'gloss': origin['gloss'],
        'text': origin['text'],
        'sign': f_tensor,
    }

    output_file = os.path.join(output_dir, 'tmp', f'{video_name}.pt')
    torch.save(op, output_file)


def _process(npz_dir, original_pklgz, output_dir, prefix):
    print('Initial process...')
    tmp_dir = os.path.join(output_dir, 'tmp')
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    print('Reading original ds...')
    origin_ds, origin_v_names = _get_original_dataset(original_pklgz)

    def _get_origin(vn):
        n = prefix + '/' + vn
        origin = origin_ds[origin_v_names[n]]
        del origin_v_names[n]
        return origin

    # make each pt file
    print('processing each files...')
    filenames = os.listdir(npz_dir)
    for filename in tqdm(filenames):
        if filename.endswith('.npz'):
            full_path = os.path.join(npz_dir, filename)
            feature, video_name = _open_npz(full_path)
            origin = _get_origin(video_name)
            _convert(feature, origin, output_dir, video_name)

    # compile all pt files
    print("Making a pickle gzip file...")

    data_list = []
    for file in tqdm(os.listdir(tmp_dir)):
        if file.endswith('.pt'):
            file_path = os.path.join(tmp_dir, file)
            data = torch.load(file_path)
            data_list.append(data)

    print("Finalizing...")
    with gzip.open(os.path.join(output_dir, f'{prefix}.pkl.gz'), 'wb') as f:
        pickle.dump(data_list, f)
    shutil.rmtree(tmp_dir)

    print(f"Done! Check {output_dir}")


if __name__ == '__main__':
    npz_dir = ''
    original_pklgz = ''
    output_dir = ''
    prefix = 'train'  # train, dev, test

    _process(npz_dir, original_pklgz, output_dir, prefix)
