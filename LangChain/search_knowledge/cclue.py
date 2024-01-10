import json
import os

import requests
from tqdm import tqdm

from LangChain.tools import query_template, search_knowledge, instruction_template, DICT_TO_LABEL
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
        options = [elem['option_0'], elem['option_1'], elem['option_2'], elem['option_3']]
        query = query_template(question=elem['query'], options=options)
        query = '根据文件:' + elem['file_name'] + '.txt 回答问题。' + query
        passage_ = search_knowledge(query=query, kb_name=knowledge_base_name_)
        prompt = instruction_template(passage_=passage_, question=elem['query'], options=options)
        process_dataset_.append({
            'prompt': prompt,
            'label': DICT_TO_LABEL[elem['label']]
        })
    return process_dataset_


if __name__ == '__main__':
    knowledge_base_name = 'CCLUE-MRC'
    PHASES = ['dev', 'test']
    for phase in PHASES:
        data_path = f'/data0/maqi/KGLQA-data/datasets/CCLUE/LangChain/knowledge_base/{phase}.jsonl'
        save_path = f'/data0/maqi/KGLQA-data/datasets/CCLUE/LangChain/select'
        # 文件夹不存在 则新建
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(save_path)
        dataset = read_ncr(data_path)
        process_dataset = process(dataset, knowledge_base_name_=knowledge_base_name)
        save_data(dataset_=process_dataset, save_path_=f'{save_path}/{phase}.jsonl')
