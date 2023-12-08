"""
只保留原文，每一条文本保存成一个txt文件
"""
import json
import os.path
from typing import Set

from tqdm import tqdm

import re

from LangChain.utils import clean_str, gen_file_name
from utils.io_json import write_jsonl


def read_ncr(data_path_):
    dataset_ = []
    with open(data_path_, 'r') as f:
        for line in f:
            dataset_.append(json.loads(line))
    return dataset_


def process(dataset_):
    process_dataset_ = []
    kb = set()
    for elem in tqdm(dataset_):
        passage = elem['context']
        passage = passage.strip()
        kb, file_name = gen_file_name(kb, passage, split_num=30,repeat=True)
        elem['file_name'] = file_name
        process_dataset_.append(elem)
    return process_dataset_


def save_data(dataset_, save_path_, phase_):
    if not os.path.exists(f'{save_path_}/kb/{phase_}'):
        os.makedirs(f'{save_path_}/kb/{phase_}')
    for elem in tqdm(dataset_):
        with open(f'{save_path_}/kb/{phase_}/{elem["file_name"]}.txt', 'w') as f:
            f.write(elem['context'])
    print(f'save knowledge base to {save_path_}')
    # 存储为jsonl
    write_jsonl(dataset_, f'{save_path_}/{phase_}.jsonl')
    print(f'process dataset save to {save_path_}')


if __name__ == '__main__':
    PHASES = ['train', 'dev', 'test']
    for phase in PHASES:
        data_path = f'/data0/maqi/KGLQA-data/datasets/CCLUE/cclue_normal/{phase}.jsonl'
        save_path = '/data0/maqi/KGLQA-data/datasets/CCLUE/LangChain/knowledge_base'
        dataset = read_ncr(data_path)
        dataset = process(dataset)
        save_data(dataset, save_path, phase)
