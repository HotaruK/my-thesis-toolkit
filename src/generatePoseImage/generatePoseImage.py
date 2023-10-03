import cv2
import mediapipe as mp
import os
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
landmark_drawing_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=0, circle_radius=0)
connection_drawing_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)
num_workers = multiprocessing.cpu_count()
mp_face_mesh = mp.solutions.face_mesh

# landmark data for pose(body and arms)
pose_landmarks = [mp_holistic.PoseLandmark.LEFT_SHOULDER,
                  mp_holistic.PoseLandmark.RIGHT_SHOULDER,
                  mp_holistic.PoseLandmark.LEFT_ELBOW,
                  mp_holistic.PoseLandmark.RIGHT_ELBOW,
                  mp_holistic.PoseLandmark.LEFT_WRIST,
                  mp_holistic.PoseLandmark.RIGHT_WRIST,
                  mp_holistic.PoseLandmark.LEFT_HIP,
                  mp_holistic.PoseLandmark.RIGHT_HIP]
pose_connections = [(mp_holistic.PoseLandmark.LEFT_SHOULDER, mp_holistic.PoseLandmark.RIGHT_SHOULDER),
                    (mp_holistic.PoseLandmark.LEFT_SHOULDER, mp_holistic.PoseLandmark.LEFT_ELBOW),
                    (mp_holistic.PoseLandmark.RIGHT_SHOULDER, mp_holistic.PoseLandmark.RIGHT_ELBOW),
                    (mp_holistic.PoseLandmark.LEFT_ELBOW, mp_holistic.PoseLandmark.LEFT_WRIST),
                    (mp_holistic.PoseLandmark.RIGHT_ELBOW, mp_holistic.PoseLandmark.RIGHT_WRIST),
                    (mp_holistic.PoseLandmark.LEFT_HIP, mp_holistic.PoseLandmark.RIGHT_HIP),
                    (mp_holistic.PoseLandmark.LEFT_SHOULDER, mp_holistic.PoseLandmark.LEFT_HIP),
                    (mp_holistic.PoseLandmark.RIGHT_SHOULDER, mp_holistic.PoseLandmark.RIGHT_HIP)]
face_landmarks = [
    [57, 13, 291, 14, 57],  # mouth
    [70, 53, 52, 65, 55],  # left eyebrow
    [285, 295, 282, 283, 300],  # right eyebrow
    [468],  # left eye
    [473],  # right eye
]


def __drawing_pose(image, results):
    for landmark in pose_landmarks:
        landmark_point = results.pose_landmarks.landmark[landmark]
        loc = (int(landmark_point.x * image.shape[1]), int(landmark_point.y * image.shape[0]))
        cv2.circle(image, loc, landmark_drawing_spec.circle_radius, landmark_drawing_spec.color,
                   landmark_drawing_spec.thickness)

    for connection in pose_connections:
        start_point = results.pose_landmarks.landmark[connection[0]]
        end_point = results.pose_landmarks.landmark[connection[1]]
        start_loc = (int(start_point.x * image.shape[1]), int(start_point.y * image.shape[0]))
        end_loc = (int(end_point.x * image.shape[1]), int(end_point.y * image.shape[0]))
        cv2.line(image, start_loc, end_loc, connection_drawing_spec.color,
                 connection_drawing_spec.thickness)


def __drawing_face(image, results):
    for landmark_group in face_landmarks:
        for i in range(len(landmark_group) - 1):
            start_point = results.face_landmarks.landmark[landmark_group[i]]
            end_point = results.face_landmarks.landmark[landmark_group[i + 1]]
            start_loc = (int(start_point.x * image.shape[1]), int(start_point.y * image.shape[0]))
            end_loc = (int(end_point.x * image.shape[1]), int(end_point.y * image.shape[0]))
            cv2.line(image, start_loc, end_loc, connection_drawing_spec.color,
                     connection_drawing_spec.thickness)
        # Draw circles for the eyes
        if len(landmark_group) == 1:
            landmark_point = results.face_landmarks.landmark[landmark_group[0]]
            loc = (int(landmark_point.x * image.shape[1]), int(landmark_point.y * image.shape[0]))
            cv2.circle(image, loc, landmark_drawing_spec.circle_radius * 2, landmark_drawing_spec.color,
                       landmark_drawing_spec.thickness)

def _process_image(output_options: dict, image_file: str, output_name: str):
    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with mp_holistic.Holistic(static_image_mode=True, refine_face_landmarks=True) as holistic:
        results = holistic.process(image)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if output_options['pose']:
        __drawing_pose(image, results)

    if output_options['hands']:
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  landmark_drawing_spec, connection_drawing_spec)
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  landmark_drawing_spec, connection_drawing_spec)

    if output_options['face']:
        __drawing_face(image, results)

    cv2.imwrite(output_name, image)


def process_images(option: dict, input_dir: str, output_dir: str):
    total_files = sum([len(files) for r, d, files in os.walk(input_dir)])
    progress_bar = tqdm(total=total_files)

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if file.endswith('.png'):
                    input_file = os.path.join(root, file)
                    rel_path = os.path.relpath(input_file, input_dir)
                    output_file = os.path.join(output_dir, rel_path)
                    os.makedirs(os.path.dirname(output_file), exist_ok=True)
                    future = executor.submit(_process_image, option, input_file, output_file)
                    futures.append(future)

        for future in as_completed(futures):
            progress_bar.update(1)

    progress_bar.close()


if __name__ == '__main__':
    option = {
        'hands': True,
        'pose': True,
        'face': True,
    }
    work_dir = ''
    input_dir = work_dir + '/train'
    output_dir = os.path.join(work_dir, f'landmark_over_image_{"_".join([f"{k}_{v}" for k, v in option.items()])}')
    process_images(option, input_dir, output_dir)
