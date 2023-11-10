import os
import shutil


def rename_files(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith("-rgb.pt"):
            new_filename = filename.replace("-rgb.pt", ".pt")
            shutil.copy(os.path.join(input_dir, filename), os.path.join(output_dir, new_filename))


if __name__ == '__main__':
    rename_files(input_dir, output_dir)
