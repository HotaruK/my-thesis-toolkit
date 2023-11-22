import os


def check_dir_diff(original_dir, processed_dir, output_file):
    failed_files = []

    for video_name in os.listdir(original_dir):
        original_files = os.listdir(os.path.join(original_dir, video_name))
        processed_files = os.listdir(os.path.join(processed_dir, video_name))

        if len(original_files) != len(processed_files):
            original_files_set = set(original_files)
            processed_files_set = set(processed_files)

            failed_files.extend(original_files_set - processed_files_set)

    with open(output_file, 'w') as f:
        for file in failed_files:
            f.write(file + '\n')


if __name__ == '__main__':
    check_dir_diff(original_dir, processed_dir, output_file)
