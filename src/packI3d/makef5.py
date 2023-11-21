import os
import numpy as np

def marge_tensor(filename, file_path1, file_path2, output1, output2):
    data1 = np.load(file_path1).f.feature
    data2 = np.load(file_path2).f.feature

    # (2, frames, 1024)
    new_tensor1 = np.stack((data1, data2))
    # (1, frames, 2048)
    new_tensor2 = np.concatenate((data1, data2), axis=2)

    np.savez(f'{output1}/{filename}.npz', new_tensor1)
    np.savez(f'{output2}/{filename}.npz', new_tensor2)

def process(input_dir1, input_dir2, output_dir1, output_dir2):
    filenames = os.listdir(input_dir1)
    for filename in filenames:
        file_path1 = os.path.join(input_dir1, filename)
        file_path2 = os.path.join(input_dir2, filename)
        marge_tensor(filename, file_path1, file_path2, output_dir1, output_dir2)

if __name__ == '__main__':
    input_dir1 = ''
    input_dir2 = ''
    output_dir1 = ''
    output_dir2 = ''
    process(input_dir1, input_dir2, output_dir1, output_dir2)