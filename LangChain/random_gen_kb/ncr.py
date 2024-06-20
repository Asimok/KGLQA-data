"""
只保留原文，每一条文本保存成一个txt文件
"""
import json
import os.path
import random
from typing import Set

from tqdm import tqdm

import re

from LangChain.tools import clean_str, gen_file_name, instruction_template, DICT_TO_LABEL
from utils.io_json import write_jsonl


def read_ncr(data_path_='/data0/maqi/KGLQA-data/datasets/NCR/ncr_format/test.jsonl'):
    dataset_ = []
    with open(data_path_, 'r') as f:
        for line in f:
            dataset_.append(json.loads(line))
    return dataset_


def process(dataset_):
    process_dataset_ = []
    kb = set()
    for elem in tqdm(dataset_):
        passage = elem['article']
        passage = passage.strip()
        kb, file_name = gen_file_name(kb, passage, split_num=40)
        for question in elem['questions']:
            # 从 kb中随机选择三个元素
            if len(kb) < 3:
                passage_ = list(kb)
            else:
                passage_ = random.sample(kb, 3)
            options = [option[2:] for option in question['options']]
            prompt = instruction_template(passage_=' '.join(passage_), question=question['question'], options=options)
            process_dataset_.append({
                'prompt': prompt,
                'label': DICT_TO_LABEL[question['gold_label']]
            })
    return process_dataset_


def save_data(dataset_, save_path_):
    # 存储为jsonl
    write_jsonl(dataset_, save_path_)
    print(f'save to {save_path_}')


if __name__ == '__main__':
    PHASES = [ 'test']
    for phase in PHASES:
        data_path = f'/data0/maqi/KGLQA-data/datasets/NCR/ncr_format/{phase}.jsonl'
        save_path = f'/data0/maqi/KGLQA-data/datasets/NCR/LangChain/random_select/{phase}.jsonl'
        dataset = read_ncr(data_path)
        process_dataset = process(dataset)
        save_data(dataset_=process_dataset, save_path_=save_path)
