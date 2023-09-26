import imageio.v2 as imageio
import os
import glob


def create_video_from_images(directory, output_name):
    images = []
    for file_name in sorted(glob.glob(os.path.join(directory, '*.png'))):
        images.append(imageio.imread(file_name))
    imageio.mimsave(output_name, images, fps=25)


if __name__ == '__main__':
    input_dir = ''
    output_dir = ''
    create_video_from_images(input_dir, output_dir)
