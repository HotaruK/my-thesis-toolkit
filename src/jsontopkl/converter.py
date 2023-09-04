import os
import jsonpickle


def convert_json_to_pickle(base_directory):
    json_directory = os.path.join(base_directory, 'openpose_output', 'json')
    pkl_directory = os.path.join(base_directory, 'openpose_output', 'pkl')
    for subdir, _, files in os.walk(json_directory):
        for file in files:
            if file.endswith('.json'):
                json_file_path = os.path.join(subdir, file)
                with open(json_file_path, 'r') as f:
                    data = jsonpickle.decode(f.read())
                pkl_subdir = subdir.replace(json_directory, pkl_directory)
                os.makedirs(pkl_subdir, exist_ok=True)
                pkl_file_path = os.path.join(pkl_subdir, file.replace('.json', '.pkl'))
                with open(pkl_file_path, 'wb') as f:
                    f.write(jsonpickle.encode(data).encode('utf-8'))


if __name__ == '__main__':
    base_dir = "C:\\Users\\Administrator\\Documents\\GitHub\\GloFE\\How2Sign"
    convert_json_to_pickle(base_dir)
