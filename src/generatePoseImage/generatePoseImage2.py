import cv2
import mediapipe as mp
import os
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
sc_landmark_drawing_spec = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=0, circle_radius=0)
sc_connection_drawing_spec = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1)
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

# body
shoulder_drawing_spec = mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=0)
upper_arm_drawing_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=0)
forearm_drawing_spec = mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=0)
body_drawing_spec = mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=1, circle_radius=0)
# hands
palm_drawing_spec = mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=1, circle_radius=0)  # Magenta
thumb_drawing_spec = mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=1, circle_radius=0)  # Cyan
index_finger_drawing_spec = mp_drawing.DrawingSpec(color=(128, 0, 0), thickness=1, circle_radius=0)  # Maroon
middle_finger_drawing_spec = mp_drawing.DrawingSpec(color=(0, 128, 0), thickness=1, circle_radius=0)  # Dark Green
ring_finger_drawing_spec = mp_drawing.DrawingSpec(color=(0, 0, 128), thickness=1, circle_radius=0)  # Navy
pinky_drawing_spec = mp_drawing.DrawingSpec(color=(128, 128, 0), thickness=1, circle_radius=0)  # Olive
# face
mouth_drawing_spec = mp_drawing.DrawingSpec(color=(0, 128, 128), thickness=1, circle_radius=1)  # Teal
eyebrow_drawing_spec = mp_drawing.DrawingSpec(color=(128, 0, 128), thickness=1, circle_radius=1)  # Purple
eye_drawing_spec = mp_drawing.DrawingSpec(color=(128, 128, 128), thickness=1, circle_radius=1)  # Gray

palm_connections = [(0, 17), (0, 5), (13, 17), (9, 13), (5, 9)]
palm_landmarks = [0, 5, 9, 13, 17]

face_landmarks = [
    [57, 13, 291, 14, 57],  # mouth
    [70, 53, 52, 65, 55],  # left eyebrow
    [285, 295, 282, 283, 300],  # right eyebrow
    [468],  # left eye
    [473],  # right eye
]


def __sc_drawing_pose(image, results):
    if results.pose_landmarks is None:
        return

    for landmark in pose_landmarks:
        landmark_point = results.pose_landmarks.landmark[landmark]
        loc = (int(landmark_point.x * image.shape[1]), int(landmark_point.y * image.shape[0]))
        cv2.circle(image, loc, sc_landmark_drawing_spec.circle_radius, sc_landmark_drawing_spec.color,
                   sc_landmark_drawing_spec.thickness)

    for connection in pose_connections:
        start_point = results.pose_landmarks.landmark[connection[0]]
        end_point = results.pose_landmarks.landmark[connection[1]]
        start_loc = (int(start_point.x * image.shape[1]), int(start_point.y * image.shape[0]))
        end_loc = (int(end_point.x * image.shape[1]), int(end_point.y * image.shape[0]))
        cv2.line(image, start_loc, end_loc, sc_connection_drawing_spec.color,
                 sc_connection_drawing_spec.thickness)


def __sc_drawing_face(image, results):
    if results.face_landmarks is None:
        return

    for landmark_group in face_landmarks:
        for i in range(len(landmark_group) - 1):
            start_point = results.face_landmarks.landmark[landmark_group[i]]
            end_point = results.face_landmarks.landmark[landmark_group[i + 1]]
            start_loc = (int(start_point.x * image.shape[1]), int(start_point.y * image.shape[0]))
            end_loc = (int(end_point.x * image.shape[1]), int(end_point.y * image.shape[0]))
            cv2.line(image, start_loc, end_loc, sc_connection_drawing_spec.color,
                     sc_connection_drawing_spec.thickness)
        # Draw circles for the eyes
        if len(landmark_group) == 1:
            landmark_point = results.face_landmarks.landmark[landmark_group[0]]
            loc = (int(landmark_point.x * image.shape[1]), int(landmark_point.y * image.shape[0]))
            cv2.circle(image, loc, sc_landmark_drawing_spec.circle_radius * 2, sc_landmark_drawing_spec.color,
                       sc_landmark_drawing_spec.thickness)


def _generate_single_channel_image(output_options, image_size, results, output_name):
    annotated_image = np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)
    if output_options['pose']:
        __sc_drawing_pose(annotated_image, results)

    if output_options['hands']:
        mp_drawing.draw_landmarks(annotated_image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  sc_landmark_drawing_spec, sc_connection_drawing_spec)
        mp_drawing.draw_landmarks(annotated_image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  sc_landmark_drawing_spec, sc_connection_drawing_spec)

    if output_options['face']:
        __sc_drawing_face(annotated_image, results)

    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(output_name, annotated_image)


def __drawing_pose(image, results):
    if results.pose_landmarks is None:
        return

    for landmark in pose_landmarks:
        landmark_point = results.pose_landmarks.landmark[landmark]
        loc = (int(landmark_point.x * image.shape[1]), int(landmark_point.y * image.shape[0]))
        if landmark in [mp_holistic.PoseLandmark.LEFT_SHOULDER, mp_holistic.PoseLandmark.RIGHT_SHOULDER]:
            cv2.circle(image, loc, shoulder_drawing_spec.circle_radius, shoulder_drawing_spec.color,
                       shoulder_drawing_spec.thickness)
        elif landmark in [mp_holistic.PoseLandmark.LEFT_ELBOW, mp_holistic.PoseLandmark.RIGHT_ELBOW]:
            cv2.circle(image, loc, upper_arm_drawing_spec.circle_radius, upper_arm_drawing_spec.color,
                       upper_arm_drawing_spec.thickness)
        elif landmark in [mp_holistic.PoseLandmark.LEFT_WRIST, mp_holistic.PoseLandmark.RIGHT_WRIST]:
            cv2.circle(image, loc, forearm_drawing_spec.circle_radius, forearm_drawing_spec.color,
                       forearm_drawing_spec.thickness)
        else:
            cv2.circle(image, loc, body_drawing_spec.circle_radius, body_drawing_spec.color,
                       body_drawing_spec.thickness)

    for connection in pose_connections:
        start_point = results.pose_landmarks.landmark[connection[0]]
        end_point = results.pose_landmarks.landmark[connection[1]]
        start_loc = (int(start_point.x * image.shape[1]), int(start_point.y * image.shape[0]))
        end_loc = (int(end_point.x * image.shape[1]), int(end_point.y * image.shape[0]))
        if connection == (mp_holistic.PoseLandmark.LEFT_SHOULDER, mp_holistic.PoseLandmark.RIGHT_SHOULDER):
            cv2.line(image, start_loc, end_loc, shoulder_drawing_spec.color,
                     shoulder_drawing_spec.thickness)
        elif connection in [(mp_holistic.PoseLandmark.LEFT_HIP, mp_holistic.PoseLandmark.RIGHT_HIP),
                            (mp_holistic.PoseLandmark.LEFT_SHOULDER, mp_holistic.PoseLandmark.LEFT_HIP),
                            (mp_holistic.PoseLandmark.RIGHT_SHOULDER, mp_holistic.PoseLandmark.RIGHT_HIP)]:
            cv2.line(image, start_loc, end_loc, body_drawing_spec.color,
                     body_drawing_spec.thickness)
        elif connection in [(mp_holistic.PoseLandmark.LEFT_SHOULDER, mp_holistic.PoseLandmark.LEFT_ELBOW),
                            (mp_holistic.PoseLandmark.RIGHT_SHOULDER, mp_holistic.PoseLandmark.RIGHT_ELBOW)]:
            cv2.line(image, start_loc, end_loc, upper_arm_drawing_spec.color,
                     upper_arm_drawing_spec.thickness)
        else:
            cv2.line(image, start_loc, end_loc, forearm_drawing_spec.color,
                     forearm_drawing_spec.thickness)


def __drawing_hand_landmarks(image, hand_landmarks, hand_connections):
    if hand_landmarks is None or hand_landmarks.landmark is None:
        return
    # Draw the landmarks
    for i in range(0, 21):
        landmark_point = hand_landmarks.landmark[i]
        loc = (int(landmark_point.x * image.shape[1]), int(landmark_point.y * image.shape[0]))
        if i in palm_landmarks:
            cv2.circle(image, loc, palm_drawing_spec.circle_radius, palm_drawing_spec.color,
                       palm_drawing_spec.thickness)
        elif i <= 4:
            cv2.circle(image, loc, thumb_drawing_spec.circle_radius, thumb_drawing_spec.color,
                       thumb_drawing_spec.thickness)
        elif i <= 8:
            cv2.circle(image, loc, index_finger_drawing_spec.circle_radius, index_finger_drawing_spec.color,
                       index_finger_drawing_spec.thickness)
        elif i <= 12:
            cv2.circle(image, loc, middle_finger_drawing_spec.circle_radius, middle_finger_drawing_spec.color,
                       middle_finger_drawing_spec.thickness)
        elif i <= 16:
            cv2.circle(image, loc, ring_finger_drawing_spec.circle_radius, ring_finger_drawing_spec.color,
                       ring_finger_drawing_spec.thickness)
        else:
            cv2.circle(image, loc, pinky_drawing_spec.circle_radius, pinky_drawing_spec.color,
                       pinky_drawing_spec.thickness)

    # Draw the connections
    for connection in hand_connections:
        start_point = hand_landmarks.landmark[connection[0]]
        end_point = hand_landmarks.landmark[connection[1]]
        start_loc = (int(start_point.x * image.shape[1]), int(start_point.y * image.shape[0]))
        end_loc = (int(end_point.x * image.shape[1]), int(end_point.y * image.shape[0]))
        if (connection[0], connection[1]) in palm_connections or (connection[1], connection[0]) in palm_connections:
            cv2.line(image, start_loc, end_loc, palm_drawing_spec.color,
                     palm_drawing_spec.thickness)
        elif connection[0] <= 4 or connection[1] <= 4:
            cv2.line(image, start_loc, end_loc, thumb_drawing_spec.color,
                     thumb_drawing_spec.thickness)
        elif connection[0] <= 8 or connection[1] <= 8:
            cv2.line(image, start_loc, end_loc, index_finger_drawing_spec.color,
                     index_finger_drawing_spec.thickness)
        elif connection[0] <= 12 or connection[1] <= 12:
            cv2.line(image, start_loc, end_loc, middle_finger_drawing_spec.color,
                     middle_finger_drawing_spec.thickness)
        elif connection[0] <= 16 or connection[1] <= 16:
            cv2.line(image, start_loc, end_loc, ring_finger_drawing_spec.color,
                     ring_finger_drawing_spec.thickness)
        else:
            cv2.line(image, start_loc, end_loc, pinky_drawing_spec.color,
                     pinky_drawing_spec.thickness)


def __drawing_face(image, results):
    if results.face_landmarks is None:
        return

    for landmark_group in face_landmarks:
        if len(landmark_group) == 1:
            # Draw circles for the eyes
            landmark_point = results.face_landmarks.landmark[landmark_group[0]]
            loc = (int(landmark_point.x * image.shape[1]), int(landmark_point.y * image.shape[0]))
            cv2.circle(image, loc, 0, eye_drawing_spec.color,
                       eye_drawing_spec.thickness)
            continue
        for i in range(len(landmark_group) - 1):
            start_point = results.face_landmarks.landmark[landmark_group[i]]
            end_point = results.face_landmarks.landmark[landmark_group[i + 1]]
            start_loc = (int(start_point.x * image.shape[1]), int(start_point.y * image.shape[0]))
            end_loc = (int(end_point.x * image.shape[1]), int(end_point.y * image.shape[0]))
            if landmark_group == face_landmarks[0]:  # mouth
                cv2.line(image, start_loc, end_loc, mouth_drawing_spec.color,
                         mouth_drawing_spec.thickness)
            elif landmark_group == face_landmarks[1] or landmark_group == face_landmarks[2]:  # eyebrow
                cv2.line(image, start_loc, end_loc, eyebrow_drawing_spec.color,
                         eyebrow_drawing_spec.thickness)


def _generate_bkbg_rgb(output_options, image_size, results, output_name):
    image = np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)
    if output_options['pose']:
        __drawing_pose(image, results)

    if output_options['hands']:
        __drawing_hand_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        __drawing_hand_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

    if output_options['face']:
        __drawing_face(image, results)

    cv2.imwrite(output_name, image)


def _generate_original_rgb(output_options, image, results, output_name):
    if output_options['pose']:
        __drawing_pose(image, results)

    if output_options['hands']:
        __drawing_hand_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        __drawing_hand_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

    if output_options['face']:
        __drawing_face(image, results)
    cv2.imwrite(output_name, image)


def _process_image(output_options: dict, image_file: str, output_dir: str, rel_path: str):
    try:
        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        with mp_holistic.Holistic(static_image_mode=True, refine_face_landmarks=True) as holistic:
            results = holistic.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image_height, image_width, _ = image.shape
        image_size = (image_height, image_width)

        # single channel image
        output_name = os.path.join(output_dir, 'sc', rel_path)
        os.makedirs(os.path.dirname(output_name), exist_ok=True)
        _generate_single_channel_image(output_options, image_size, results, output_name)

        # black background + color annotation
        output_name = os.path.join(output_dir, 'bkbg_rgb', rel_path)
        os.makedirs(os.path.dirname(output_name), exist_ok=True)
        _generate_bkbg_rgb(output_options, image_size, results, output_name)

        # original image + color annotation
        output_name = os.path.join(output_dir, 'rgb', rel_path)
        os.makedirs(os.path.dirname(output_name), exist_ok=True)
        _generate_original_rgb(output_options, image, results, output_name)
    except Exception as e:
        print(e)


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
                    future = executor.submit(_process_image, option, input_file, output_dir, rel_path)
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
    process_images(option, input_dir, output_dir)
