import os
import gzip
import pickle


def _pack_dataset(dataset_path: str, gaslt_ds_path: str, output_dir: str, ds_type: str):
    with gzip.open(gaslt_ds_path, 'rb') as f:
        i3d_ds = pickle.load(f)

    for idx, v in enumerate(i3d_ds):
        i3d_ds[idx].pop('sign')
        video_name = i3d_ds[idx]['name'].split('/')[-1]
        video_path = os.path.join(dataset_path, video_name)
        sign_length = len([i for i in os.listdir(video_path) if i.startswith('image')])
        i3d_ds[idx]['sgn_len'] = sign_length

    with gzip.open(os.path.join(output_dir, f'sgnlen.{ds_type}.pkl.gz'), 'wb') as f:
        pickle.dump(i3d_ds, f)


if __name__ == '__main__':
    dataset_root_dir = ''
    gaslt_original_ds_dir = ''
    output_dir = ''
    type = [
        'test',
        'dev',
        'train'
    ]

    for t in type:
        input_dir = os.path.join(dataset_root_dir, t)
        print(f'start {t}')
        _pack_dataset(dataset_path=input_dir,
                      gaslt_ds_path=os.path.join(gaslt_original_ds_dir, f'phoenix14t.pami0.{t}'),
                      output_dir=output_dir,
                      ds_type=t)
        print(f'done pack dataset: {t}')
