import json
import os

from tqdm import tqdm

from option_3.utils import query_template, search_knowledge, instruction_template, DICT_TO_LABEL
from utils.io_json import write_jsonl


def read_ncr(data_path_):
    dataset_ = []
    with open(data_path_, 'r') as f:
        for line in f:
            dataset_.append(json.loads(line))
    return dataset_


def save_data(dataset_, save_path_):
    # 存储为jsonl
    write_jsonl(dataset_, save_path_)
    print(f'save to {save_path_}')


def process(dataset_, knowledge_base_name_):
    process_dataset_ = []
    for elem in tqdm(dataset_):
        options = elem['options']
        query = query_template(question=elem['question'], options=options)
        query = 'Base on:' + elem['file_name'] + '.txt answer the question.' + query
        passage_ = search_knowledge(query=query, kb_name=knowledge_base_name_, top_k=12)
        prompt = instruction_template(passage_=passage_, question=elem['question'], options=options)
        process_dataset_.append({
            'prompt': prompt,
            'label': elem['answer']
        })
    return process_dataset_


if __name__ == '__main__':
    knowledge_base_name = 'QuALITY+RACE'
    PHASES = ['test']
    for phase in PHASES:
        for dataset_type in ['middle', 'high']:
            data_path = f'/data0/maqi/KGLQA-data/datasets/RACE/LangChain/knowledge_base/{dataset_type}_{phase}.jsonl'
            save_path = f'/data0/maqi/KGLQA-data/datasets/RACE/LangChain/select_quality_and_race'
            # 文件夹不存在 则新建
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            dataset = read_ncr(data_path)
            process_dataset = process(dataset, knowledge_base_name_=knowledge_base_name)
            save_data(dataset_=process_dataset, save_path_=f'{save_path}/{dataset_type}_{phase}.jsonl')
