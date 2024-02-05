import os
from tqdm import tqdm
import gzip
import numpy as np
import pickle
import torch


def marge_tensor(i3d, lmfile):
    lm = np.load(lmfile, allow_pickle=True)
    i3d_t = i3d['sign']

    try:
        assert len(i3d_t) == len(lm)
    except AssertionError:
        with open('../generateLandmarkDataset/error_features.txt', 'a') as f:
            f.write(lmfile + '\n')
            return

    result = []
    i = 0
    while i < len(lm):
        d1 = lm[i]['pose'] + lm[i]['left_hand'] + lm[i]['right_hand'] + lm[i]['face']
        d1 = np.array([[p['x'], p['y']] for p in d1]).reshape(-1)
        d1 = torch.from_numpy(d1)
        i3d_tt = i3d_t[i]
        d1 = np.concatenate((i3d_tt, d1.numpy()), axis=0)
        result.append(d1)
        i += 1

    result = torch.tensor(result)
    assert result.shape == torch.Size((len(lm), 1024 + 553 * 2,))
    i3d['sign'] = result

    return i3d


def find_and_pop_name_by_object(objects, target_name):
    for i, obj in enumerate(objects):
        if obj.get('name') == target_name:
            return objects.pop(i)
    return None


def _process_dataset(input_dir, output_dir, i3d_file_path, type):
    files = [f for f in os.listdir(input_dir) if f.endswith('.pickle')]
    progress_bar = tqdm(total=len(files)+1)

    with gzip.open(i3d_file_path, 'rb') as f:
        i3d_ds = pickle.load(f)

    result = []
    for ff in files:
        lm_file_path = os.path.join(input_dir, ff)
        file_name, _ = os.path.splitext(ff)

        i3d_obj = find_and_pop_name_by_object(i3d_ds, f'{type}/{file_name}')
        result.append(marge_tensor(i3d_obj, lm_file_path))
        progress_bar.update(1)

    with gzip.open(os.path.join(output_dir, f'{type}.pkl.gz'), 'wb') as f:
        pickle.dump(result, f)
    progress_bar.update(1)
    progress_bar.close()


if __name__ == '__main__':
    root_dir = ''
    i3d_dir = ''
    output_root_dir = ''
    type = ['train', 'test', 'dev']

    for t in types:
        work_dir = f'{root_dir}/{t}'
        output_dir = f'{output_root_dir}'
        i3d_file_path = f'{i3d_dir}/{t}.pkl.gz'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        _process_dataset(work_dir, output_dir, i3d_file_path, t)
