import os
import numpy as np


def marge_tensor(filename, file_path1, file_path2, output_dir1, output_dir2):
    data1 = np.load(file_path1, allow_pickle=True)
    data2 = np.load(file_path2, allow_pickle=True)
    data1 = data1.f.feature
    data2 = data2.f.feature
    try:
        assert data1.shape == data2.shape
    except AssertionError:
        with open('error_features.txt', 'a') as f:
            f.write(filename + '\n')
            return

    # (2, frames, 1024)
    new_tensor1 = np.concatenate((data1[np.newaxis, :], data2[np.newaxis, :]), axis=0)
    assert new_tensor1.shape == (2,) + data1.shape

    # (1, frames, 2048)
    new_tensor2 = np.concatenate((data1, data2), axis=2)
    assert new_tensor2.shape == data1.shape[:-1] + (2048,)

    np.savez(f'{output_dir1}/{filename}', new_tensor1)
    np.savez(f'{output_dir2}/{filename}', new_tensor2)


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