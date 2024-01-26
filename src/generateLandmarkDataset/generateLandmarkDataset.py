import cv2
import mediapipe as mp
import os
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import pickle
from google.protobuf.json_format import MessageToDict

num_workers = multiprocessing.cpu_count()
mp_holistic = mp.solutions.holistic

NUMBER_OF_POSE_LANDMARKS = 33
NUMBER_OF_FACE_LANDMARKS = 478
NUMBER_OF_HAND_LANDMARKS = 21


def _process_image(image_file: str):
    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    landmarks = {}

    with mp_holistic.Holistic(static_image_mode=True, refine_face_landmarks=True) as holistic:
        results = holistic.process(image)

        # pose landmarks
        if results.pose_landmarks:
            landmarks['pose'] = MessageToDict(results.pose_landmarks)['landmark']
        else:
            landmarks['pose'] = [{'x': None, 'y': None, 'z': None} for _ in range(NUMBER_OF_POSE_LANDMARKS)]

        # face landmarks
        if results.face_landmarks:
            landmarks['face'] = MessageToDict(results.face_landmarks)['landmark']
        else:
            landmarks['face'] = [{'x': None, 'y': None, 'z': None} for _ in range(NUMBER_OF_FACE_LANDMARKS)]

        # hands landmarks
        if results.left_hand_landmarks:
            landmarks['left_hand'] = MessageToDict(results.left_hand_landmarks)['landmark']
        else:
            landmarks['left_hand'] = [{'x': None, 'y': None, 'z': None} for _ in range(NUMBER_OF_HAND_LANDMARKS)]

        if results.right_hand_landmarks:
            landmarks['right_hand'] = MessageToDict(results.right_hand_landmarks)['landmark']
        else:
            landmarks['right_hand'] = [{'x': None, 'y': None, 'z': None} for _ in range(NUMBER_OF_HAND_LANDMARKS)]

    return landmarks


def _process_video(input_dir, dir_name, output_dir):
    landmarks = {}
    files = os.listdir(input_dir)
    for file in files:
        if file.endswith('.png'):
            input_file = os.path.join(input_dir, file)
            landmarks[file] = _process_image(input_file)

    landmarks_values = [v for k, v in sorted(landmarks.items(), key=lambda x: x[0])]

    # check
    for i in landmarks_values:
        if len(i['pose']) == NUMBER_OF_POSE_LANDMARKS and \
                len(i['face']) == NUMBER_OF_FACE_LANDMARKS and \
                len(i['left_hand']) == NUMBER_OF_HAND_LANDMARKS and \
                len(i['right_hand']) == NUMBER_OF_HAND_LANDMARKS:
            pass
        else:
            print(f'Error! video_name={dir_name}')
            raise Exception()

    with open(os.path.join(output_dir, dir_name + '.pickle'), 'wb') as f:
        pickle.dump(landmarks_values, f)


def process_dataset(input_dir, output_dir):
    dirs = os.listdir(input_dir)
    total_files = len(dirs)
    progress_bar = tqdm(total=total_files)

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for d in dirs:
            dir_path = os.path.join(input_dir, d)
            future = executor.submit(_process_video, dir_path, d, output_dir)
            futures.append(future)
        for future in as_completed(futures):
            progress_bar.update(1)

    progress_bar.close()


if __name__ == '__main__':
    root_dir = ''
    output_root_dir = ''

    types = ['test', 'train', 'dev']
    for t in types:
        work_dir = f'{root_dir}/{t}'
        output_dir = f'{output_root_dir}/{t}'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        process_dataset(work_dir, output_dir)
