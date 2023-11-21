import numpy as np

file1 = 'path_to_your_file1.npz'
file2 = 'path_to_your_file2.npz'
output_dir = 'path_to_your_output_directory'

tensor1 = np.load(file1)['arr_0']
tensor2 = np.load(file2)['arr_0']

tensor_combined_1 = np.concatenate((tensor1, tensor2), axis=0)  # (2,1024)
tensor_combined_2 = np.concatenate((tensor1, tensor2), axis=1)  # (1,2048)

np.savez(output_dir + '/combined_tensor_1.npz', tensor_combined_1)
np.savez(output_dir + '/combined_tensor_2.npz', tensor_combined_2)

if __name__ == '__main__':
