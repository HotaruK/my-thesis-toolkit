import os
from tqdm import tqdm
import multiprocessing
import pickle

num_workers = multiprocessing.cpu_count()

NUMBER_OF_POSE_LANDMARKS = 33
NUMBER_OF_FACE_LANDMARKS = 478
NUMBER_OF_HAND_LANDMARKS = 21

NULL_OBJ_POSE = [{'x': None, 'y': None, 'z': None} for _ in range(NUMBER_OF_POSE_LANDMARKS)]
NULL_OBJ_FACE = [{'x': None, 'y': None, 'z': None} for _ in range(NUMBER_OF_FACE_LANDMARKS)]
NULL_OBJ_HAND = [{'x': None, 'y': None, 'z': None} for _ in range(NUMBER_OF_HAND_LANDMARKS)]


def _get_blacklist(blacklist_dir, key):
    p = os.path.join(blacklist_dir, key + '.txt')
    with open(p, 'r') as f:
        lines = f.readlines()
        return [line.strip() for line in lines]


def _extract_x_and_y(landmarks):
    return [{k: v for k, v in p.items() if k in ['x', 'y']} for p in landmarks]


def _get_average(start, end):
    assert len(start) == len(end)
    average = []
    for i in range(len(start)):
        o = {}
        for j in ['x', 'y']:
            try:
                o[j] = (start[i][j] + end[i][j]) / 2
            except TypeError as e:
                print(start[i], end[i])
                raise e
        average.append(o)
    return average


def _filling_missing_frames(data, missings, key):
    data_len = len(data)
    processed = []
    for frame_no in missings:
        # Frame is processed only once
        if frame_no in processed:
            continue
        else:
            processed.append(frame_no)

        # Check for continuous missing ranges
        start = frame_no
        end = frame_no
        while end < data_len:
            if end + 1 not in missings:
                break
            else:
                processed.append(end + 1)
                end += 1

        frame_list = list(range(start, end + 1))
        total_frames = len(frame_list)

        # missing only single frame
        if total_frames == 1:
            if start > 0 and end < (data_len - 1):
                # missing the middle frame
                data[start][key] = _get_average(data[start - 1][key], data[end + 1][key])
            elif start == 0:
                # missing the first frame (just copy)
                data[start][key] = data[start + 1][key]
            elif start == (data_len - 1):
                # missing the last frame (just copy)
                data[start][key] = data[start - 1][key]
        # missing multiple frames
        else:
            if start <= 0:
                # missing the first frame (just copy)
                try:
                    copy_base = data[end + 1][key]
                except IndexError as e:
                    print(start, end, data_len)
                    raise e
                for frame_idx in frame_list:
                    data[frame_idx][key] = copy_base
            elif end >= (data_len - 1):
                # missing the last frame (just copy)
                copy_base = data[start - 1][key]
                for frame_idx in frame_list:
                    try:
                        data[frame_idx][key] = copy_base
                    except IndexError as e:
                        print(frame_idx, key)
                        raise e
            else:
                mid_idx = total_frames // 2
                prev = start - 1
                nxt = end + 1
                for i in range(mid_idx, -1, -1):
                    current = frame_list[i]
                    if prev >= 0:
                        try:
                            data[current][key] = _get_average(data[prev][key], data[nxt][key])
                        except IndexError as e:
                            print(f"current={current}, key={key}, prev={prev}, next={nxt}, data_len={data_len}")
                            raise e
                    else:
                        data[current][key] = data[current + 1][key]
                    nxt = current

                prev = frame_list[mid_idx]
                nxt = end + 1
                for i in range(mid_idx + 1, total_frames):
                    current = frame_list[i]
                    if nxt < data_len:
                        data[current][key] = _get_average(data[prev][key], data[nxt][key])
                    else:
                        data[current][key] = data[current - 1][key]
                    prev = current


def _process_file(file_path, file_name, output_dir):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    # make lists of missing frames
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

    # process missing frames
    for k in ['pose', 'face', 'left_hand', 'right_hand']:
        _filling_missing_frames(data, missings[k], k)

    # data cleaning
    clean_data = [{
        'pose': _extract_x_and_y(img['pose']),
        'face': _extract_x_and_y(img['face']),
        'left_hand': _extract_x_and_y(img['left_hand']),
        'right_hand': _extract_x_and_y(img['right_hand']),
    } for img in data]

    # output
    with open(os.path.join(output_dir, file_name), 'wb') as ff:
        pickle.dump(clean_data, ff)


def process_dataset(input_dir, output_dir, blacklist):
    files = [f for f in os.listdir(input_dir) if f.endswith('.pickle')]
    progress_bar = tqdm(total=len(files))

    for f in files:
        # Videos in blacklist are excluded from the dataset due to low quality of pose estimation
        fn, _ = os.path.splitext(f)
        if fn not in blacklist:
            file_path = os.path.join(input_dir, f)
            _process_file(file_path, f, output_dir)
        progress_bar.update(1)

    progress_bar.close()


if __name__ == '__main__':
    root_dir = ''
    output_root_dir = ''
    types = ['test', 'train', 'dev']
    blacklist_dir = ''

    for t in types:
        work_dir = f'{root_dir}/{t}'
        output_dir = f'{output_root_dir}/{t}'
        blacklist = _get_blacklist(blacklist_dir, t)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        process_dataset(work_dir, output_dir, blacklist)
