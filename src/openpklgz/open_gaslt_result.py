import os
import csv
import pickle


def read_scores(directory):
    test_file_name = 'test_output.test_results.pkl'
    file_path = os.path.join(directory, test_file_name)
    with open(file_path, 'rb') as file:
        dataset = pickle.load(file)
        scores = dataset['valid_scores']
        return {
            'directory': directory,
            'bleu': scores['bleu'],
            'bleu1': scores['bleu_scores']['bleu1'],
            'bleu2': scores['bleu_scores']['bleu2'],
            'bleu3': scores['bleu_scores']['bleu3'],
            'bleu4': scores['bleu_scores']['bleu4'],
            'shrf': scores['chrf'],
            'rouge': scores['rouge']
        }


def write_to_csv(data, output_file):
    with open(output_file, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)


if __name__ == '__main__':
    data = []
    for directory in os.listdir(base_dir):
        if directory.startswith('results'):
            full_dir = os.path.join(base_dir, directory)
            data.append(read_scores(full_dir))
    write_to_csv(data, output_file)
