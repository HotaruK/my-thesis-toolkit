import os
import numpy as np
import torch
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


def npz_process(npz_file):
    data = np.load(npz_file)
    original_array = torch.from_numpy(data['feature'])
    converted_list = []
    for i in range(original_array.shape[1]):
        a = original_array[:, i, :]
        converted_list.append(a)
    return converted_list

def process_file(file):
    if file.endswith('.npz'):
        npz_file = os.path.join(input_dir, file)
        tensor = npz_process(npz_file)
        torch.save(tensor, os.path.join(output_dir, file.replace('.npz', '.pt')))


if __name__ == '__main__':
    input_dir = ''
    span = 8

    output_dir = os.path.join(os.path.dirname(input_dir), f'tspnet_span{span}')
    os.makedirs(output_dir, exist_ok=True)

    with ThreadPoolExecutor() as executor:
        list(tqdm(executor.map(process_file, os.listdir(input_dir)), total=len(os.listdir(input_dir))))
