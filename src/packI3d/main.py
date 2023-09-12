from tqdm import tqdm
import os
import torch
import pickle
import gzip


def convert_pt_to_pkl_gz(input_dir, output_file):
    data = []

    filenames = [filename for filename in os.listdir(input_dir) if filename.endswith('.pt')]
    pbar = tqdm(total=len(filenames))

    for filename in filenames:
        filepath = os.path.join(input_dir, filename)
        tensor = torch.load(filepath)
        data.append(tensor)
        pbar.update(1)

    with gzip.open(output_file, 'wb') as f:
        pickle.dump(data, f)
        pbar.close()


if __name__ == '__main__':
    input_dir = ''
    output_name = ''
    convert_pt_to_pkl_gz(input_dir, output_name)
