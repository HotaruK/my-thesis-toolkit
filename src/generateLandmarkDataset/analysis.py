import os
from tqdm import tqdm
import pickle
import json

NUMBER_OF_POSE_LANDMARKS = 33
NUMBER_OF_FACE_LANDMARKS = 478
NUMBER_OF_HAND_LANDMARKS = 21

NULL_OBJ_POSE = [{'x': None, 'y': None, 'z': None} for _ in range(NUMBER_OF_POSE_LANDMARKS)]
NULL_OBJ_FACE = [{'x': None, 'y': None, 'z': None} for _ in range(NUMBER_OF_FACE_LANDMARKS)]
NULL_OBJ_HAND = [{'x': None, 'y': None, 'z': None} for _ in range(NUMBER_OF_HAND_LANDMARKS)]


def _check_file(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    missings = {'pose': [], 'face': [], 'left_hand': [], 'right_hand': []}
    for key, img in enumerate(data):
        if img['pose'] == NULL_OBJ_POSE:
            missings['pose'].append(key)
        if img['face'] == NULL_OBJ_FACE:
            missings['face'].append(key)
        if img['left_hand'] == NULL_OBJ_HAND:
            missings['left_hand'].append(key)
        if img['right_hand'] == NULL_OBJ_HAND:
            missings['right_hand'].append(key)

    if not missings['pose'] and not missings['face'] and not missings['left_hand'] and not missings['right_hand']:
        return None, None
    else:
        data_len = len(data)
        return missings, data_len


def process_dataset(input_dir, output_dir, type):
    files = [f for f in os.listdir(input_dir) if f.endswith('.pickle')]
    progress_bar = tqdm(total=len(files) + 1)
    detail_log = output_dir + f'{type}_analysis.log'

    cnt = {'pose': 0, 'face': 0, 'left_hand': 0, 'right_hand': 0}
    video_cnt = {'pose': 0, 'face': 0, 'left_hand': 0, 'right_hand': 0}
    no_error_cnt = 0
    all_missings = []

    for file_name in files:
        file_path = os.path.join(input_dir, file_name)
        result, l = _check_file(file_path)
        if result:
            cnt['pose'] += len(result['pose'])
            cnt['face'] += len(result['face'])
            cnt['left_hand'] += len(result['left_hand'])
            cnt['right_hand'] += len(result['right_hand'])

            if len(result['pose']) > 0:
                video_cnt['pose'] += 1
            if len(result['face']) > 0:
                video_cnt['face'] += 1
            if len(result['left_hand']) > 0:
                video_cnt['left_hand'] += 1
            if len(result['right_hand']) > 0:
                video_cnt['right_hand'] += 1

            all_missing = [file_name, []]
            if len(result['pose']) == l:
                all_missing[1].append('pose')
            if len(result['face']) == l:
                all_missing[1].append('face')
            if len(result['left_hand']) == l:
                all_missing[1].append('left_hand')
            if len(result['right_hand']) == l:
                all_missing[1].append('right_hand')
            if all_missing[1]:
                all_missings.append(all_missing)

            with open(detail_log, "a") as f:
                f.write(f"[Found None] file name={file_name}\n")
                f.write(json.dumps(result))
                f.write("\n")
        else:
            no_error_cnt += 1
        progress_bar.update(1)

    with open(detail_log, "a") as f:
        f.write("----------------------------------------------------------\n")
        f.write(f"[Error count] total_videos={len(files)}, no_error_videos={no_error_cnt}\n")
        f.write(f"[Error video count] pose={video_cnt['pose']}, face={video_cnt['face']}, left_hand={video_cnt['left_hand']}, right_hand={video_cnt['right_hand']}\n")
        f.write(f"[None Count] pose={cnt['pose']}, face={cnt['face']}, left_hand={cnt['left_hand']}, right_hand={cnt['right_hand']}\n")
        f.write(f"\n\n[All missing list]:\n")
        for k, i in enumerate(all_missings):
            f.write(f"[{k}] video_name={i[0]}: {','.join(i[1])}\n")
    progress_bar.update(1)
    progress_bar.close()


if __name__ == '__main__':
    root_dir = ''
    output_root_dir = ''
    types = ['test', 'train', 'dev']

    for t in types:
        work_dir = f'{root_dir}/{t}'
        process_dataset(work_dir, output_root_dir, t)
