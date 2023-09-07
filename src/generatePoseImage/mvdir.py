import os
import shutil


def get_directory_list(directory_path):
    directory_list = []
    for entry in os.scandir(directory_path):
        if entry.is_dir():
            directory_list.append(entry.name)
    return directory_list


if __name__ == '__main__':
    src_dir = ""
    src_dir_list = get_directory_list(src_dir)
    dev_dir_list = get_directory_list("\\dev")
    test_dir_list = get_directory_list("\\train")

    for d in src_dir_list:
        target_dir = f"{src_dir}\\{d}"
        if d in dev_dir_list:
            shutil.move(target_dir, f"{src_dir}\\dev\\{d}")
        elif d in test_dir_list:
            shutil.move(target_dir, f"{src_dir}\\test\\{d}")
