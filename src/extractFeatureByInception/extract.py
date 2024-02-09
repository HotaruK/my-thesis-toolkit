from keras.applications import xception
from keras.applications import inception_resnet_v2
from keras.preprocessing import image
from PIL import Image
import numpy as np
import os
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

num_workers = multiprocessing.cpu_count()
xception_model = xception.Xception(weights='imagenet', include_top=False)
inception_model = inception_resnet_v2.InceptionResNetV2(weights='imagenet', include_top=False)


def extract_feature_by_xception(img_path):
    im = Image.open(img_path)
    im.thumbnail((224, 224))
    x = image.img_to_array(im)
    x = np.expand_dims(x, axis=0)
    x = xception.preprocess_input(x)
    f = xception_model.predict(x)
    return f


def extract_feature_by_inception_resnet_v2(img_path):
    im = Image.open(img_path)
    im.thumbnail((299, 299))
    x = image.img_to_array(im)
    x = np.expand_dims(x, axis=0)
    x = inception_resnet_v2.preprocess_input(x)
    f = inception_model.predict(x)
    return f


def _process_video(input_dir, video_name, output_dir, model_func):
    frames = {}
    files = os.listdir(input_dir)
    for file in files:
        if file.endswith('.png'):
            input_file = os.path.join(input_dir, file)
            frames[file] = model_func(input_file)

    frames = [v for k, v in sorted(frames.items(), key=lambda x: x[0])]

    np.save(os.path.join(output_dir, f'{video_name}.npy'), frames)


def process_dataset(input_dir, output_dir, model_func):
    print(f'start: input_dir={input_dir}')
    dirs = os.listdir(input_dir)
    total_videos = len(dirs)
    progress_bar = tqdm(total=total_videos)

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for d in dirs:
            dir_path = os.path.join(input_dir, d)
            future = executor.submit(_process_video, dir_path, d, output_dir, model_func)
            futures.append(future)
        for future in as_completed(futures):
            progress_bar.update(1)

    progress_bar.close()
    print(f'finish: input_dir={input_dir}')


if __name__ == '__main__':
    input_dir = ''
    output_root_dir = ''
    types = ['train', 'test', 'dev']
    model_func = extract_feature_by_xception  # or, model = extract_feature_by_inception_resnet_v2

    for t in types:
        work_dir = f'{input_dir}/{t}'
        output_dir = f'{output_root_dir}/{t}'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        process_dataset(work_dir, output_dir, model_func)
