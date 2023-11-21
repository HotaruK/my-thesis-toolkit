import cv2
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import multiprocessing


def _process_image(input_file: str, output_file: str):
    img = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    cv2.imwrite(output_file, img_rgb)


def process_images(input_dir: str, output_dir: str, num_workers: int):
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
                    future = executor.submit(_process_image, input_file, output_file)
                    futures.append(future)

        for future in as_completed(futures):
            progress_bar.update(1)

    progress_bar.close()


if __name__ == '__main__':
    input_dir = ''
    output_dir = ''
    process_images(input_dir, output_dir, multiprocessing.cpu_count())
