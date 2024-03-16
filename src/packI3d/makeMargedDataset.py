import os
from tqdm import tqdm
import gzip
import pickle
import torch

TOTAL_FULL_LANDMARKS = 2130
TOTAL_MINIMIZE_LANDMARKS = 1208


def marge_tensor(base_obj, additional_obj):
    new_sign = torch.cat((additional_obj['sign'], base_obj['sign']), dim=2)
    base_obj['sign'] = new_sign

    return base_obj


def find_and_pop_name_by_object(objects, target_name):
    for i, obj in enumerate(objects):
        if obj.get('name') == target_name:
            return objects.pop(i)
    return None


def _process_dataset(base_ds_path, output_dir, additional_dataset_path, type, prefix):
    with gzip.open(base_ds_path, 'rb') as f:
        base_ds = pickle.load(f)

    with gzip.open(additional_dataset_path, 'rb') as f:
        additional_ds = pickle.load(f)

    progress_bar = tqdm(total=len(base_ds) + 1)

    for k, d in base_ds.items():
        base_ds[k] = marge_tensor(base_obj=d,
                                  additional_obj=find_and_pop_name_by_object(additional_ds, d.get('name')))
        progress_bar.update(1)

    del additional_ds

    for i in base_ds:
        assert i['sign'].shape[2] == TOTAL_MINIMIZE_LANDMARKS

    with gzip.open(os.path.join(output_dir, f'{prefix}.{type}.pkl.gz'), 'wb') as f:
        pickle.dump(base_ds, f)
    progress_bar.update(1)
    progress_bar.close()


if __name__ == '__main__':
    additional_dataset_dir = ''
    base_dataset_dir = ''
    output_root_dir = ''
    type = ['train', 'test', 'dev']
    prefix = ''

    for t in type:
        work_dir = f'{base_dataset_dir}/{t}.pkl.gz'
        output_dir = f'{output_root_dir}'
        additional_dataset_file_path = f'{additional_dataset_dir}/{t}.pkl.gz'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        _process_dataset(work_dir, output_dir, additional_dataset_file_path, t, prefix)
