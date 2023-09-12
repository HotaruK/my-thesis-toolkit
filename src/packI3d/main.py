import os
import torch
import pickle
import gzip


def convert_pt_to_pkl_gz(input_dir, output_file):
    data = {}

    for filename in os.listdir(input_dir):
        if filename.endswith('.pt'):
            filepath = os.path.join(input_dir, filename)
            tensor = torch.load(filepath)
            data[filename] = tensor

    with gzip.open(output_file, 'wb') as f:
        pickle.dump(data, f)


if __name__ == '__main__':
    input_dir = ''
    output_name = ''
    convert_pt_to_pkl_gz(input_dir, output_name)
