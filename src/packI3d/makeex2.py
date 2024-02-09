import os
from tqdm import tqdm
import gzip
import numpy as np
import pickle
import torch

VALID_FACE_LANDMARKS = [
    57, 13, 291, 14, 57,  # mouth
    70, 53, 52, 65, 55,  # left eyebrow
    285, 295, 282, 283, 300,  # right eyebrow
    468,  # left eye
    473,  # right eye
]


def _get_face_points(landmarks, minimize_face_points=False):
    if not minimize_face_points:
        return landmarks
    result = [landmarks[i] for i in VALID_FACE_LANDMARKS]
    return result


def marge_tensor(i3d, lmfile, minimize_face_points=False):
    lm = np.load(lmfile, allow_pickle=True)

    pose = []
    left_hand = []
    right_hand = []
    face = []
    for i in range(len(lm)):
        pose.append(lm[i]['pose'])
        left_hand.append(lm[i]['left_hand'])
        right_hand.append(lm[i]['right_hand'])
        face.append(_get_face_points(lm[i]['face'], minimize_face_points))

    i3d['pose'] = torch.tensor(np.array(pose))
    i3d['left_hand'] = torch.tensor(np.array(left_hand))
    i3d['right_hand'] = torch.tensor(np.array(right_hand))
    i3d['face'] = torch.tensor(np.array(face))

    return i3d


def find_and_pop_name_by_object(objects, target_name):
    for i, obj in enumerate(objects):
        if obj.get('name') == target_name:
            return objects.pop(i)
    return None


def _process_dataset(input_dir, output_dir, i3d_file_path, type, minimize_face_points):
    files = [f for f in os.listdir(input_dir) if f.endswith('.pickle')]
    progress_bar = tqdm(total=len(files) + 1)

    with gzip.open(i3d_file_path, 'rb') as f:
        i3d_ds = pickle.load(f)

    result = []
    for ff in files:
        lm_file_path = os.path.join(input_dir, ff)
        file_name, _ = os.path.splitext(ff)

        i3d_obj = find_and_pop_name_by_object(i3d_ds, f'{type}/{file_name}')
        result.append(marge_tensor(i3d_obj, lm_file_path, minimize_face_points))
        progress_bar.update(1)

    with gzip.open(os.path.join(output_dir, f'ex2.{type}.pkl.gz'), 'wb') as f:
        pickle.dump(result, f)
    progress_bar.update(1)
    progress_bar.close()


if __name__ == '__main__':
    root_dir = ''
    i3d_dir = ''
    output_root_dir = ''
    type = ['train', 'test', 'dev']
    minimize_face_points = True

    for t in type:
        work_dir = f'{root_dir}/{t}'
        output_dir = f'{output_root_dir}'
        i3d_file_path = f'{i3d_dir}/{t}.pkl.gz'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        _process_dataset(work_dir, output_dir, i3d_file_path, t, minimize_face_points)
