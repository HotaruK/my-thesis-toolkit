import os
import gzip
import pickle
import torch


def dataset_norm(ds_dir: str, ds_name: str, output_dir: str, output_name: str):
    with gzip.open(os.path.join(ds_dir, ds_name), 'rb') as f:
        i3d_ds = pickle.load(f)

    all_tensors = torch.cat([d['sign'] for d in i3d_ds])
    mean = all_tensors.mean()
    std = all_tensors.std()
    del all_tensors

    for d in i3d_ds:
        d['sign'] = (d['sign'] - mean) / std

    with gzip.open(os.path.join(output_dir, output_name), 'wb') as f:
        pickle.dump(i3d_ds, f)


if __name__ == '__main__':
    ds_dir = ''
    ds_name = ''
    output_dir = ''
    output_name = ''
