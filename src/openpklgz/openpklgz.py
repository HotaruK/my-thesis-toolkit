import pickle
import gzip
import torch


def open_pkl_gz(file: str):
    with gzip.open(file, 'rb') as f:
        dataset = pickle.load(f)
        print(dataset)


def open_pt(file: str):
    data = torch.load(file)
    print(data)
