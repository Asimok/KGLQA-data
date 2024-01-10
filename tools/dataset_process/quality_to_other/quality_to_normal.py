import os

from tqdm import tqdm

from utils.io_json import read_jsonl, write_jsonl


def process_data(data, dataset_type):
    out = []
    i = 0
    for row in tqdm(data):
        context = row["article"]
        i += 1
        for question in row['questions']:
            query = question['question']
            options = question['options']
            # if i == 120:
            #     print(f'A#{options[0]}B#{options[1]}C#{options[2]}D#{options[3]}')
            out.append({
                "context": context,
                "query": query,
                "option_0": 'A.' + options[0],
                "option_1": 'B.' + options[1],
                "option_2": 'C.' + options[2],
                "option_3": 'D.' + options[3],
                "label": question["gold_label"] - 1 if dataset_type != 'test' else -1,
            })
            #
    return out


def process_file(input_path_, output_path_, dataset_type):
    data = read_jsonl(input_path_)
    out = process_data(data, dataset_type)
    write_jsonl(out, output_path_)
    print('save to ', output_path_)


if __name__ == '__main__':
    PHASES = ["train", "dev", "test"]

    input_base_path = '/data0/maqi/KGLQA-data/datasets/QuALITY/QuALITY.v1.0.1/QuALITY.v1.0.1.htmlstripped'
    output_base_path = f'/data0/maqi/KGLQA-data/datasets/quality_process/quality_normal_format'
    for phase in PHASES:
        input_path = f"{input_base_path}.{phase}"
        output_path = os.path.join(output_base_path, f"{phase}.jsonl")
        if not os.path.exists(output_base_path):
            os.makedirs(output_base_path)
        process_file(input_path_=input_path, output_path_=output_path, dataset_type=phase)
