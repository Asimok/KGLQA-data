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
        for question in elem['questions']:
            passage = elem['article']
            passage = passage.strip()
            kb, file_name = gen_file_name(kb, passage, split_num=100, repeat=False, max_len=200)
            # 从 kb中随机选择三个元素
            if len(kb) < 3:
                passage_ = list(kb)
            else:
                passage_ = random.sample(kb, 3)
            options = question['options']
            prompt = instruction_template(passage_=' '.join(passage_), question=question['question'], options=options)
            process_dataset_.append({
                'prompt': prompt,
                'question_unique_id': question['question_unique_id'],
                'label': DICT_TO_LABEL[question['gold_label'] - 1]
                # 'label': '-1'
            })
    return process_dataset_


def save_data(dataset_, save_path_):
    # 存储为jsonl
    write_jsonl(dataset_, save_path_)
    print(f'save to {save_path_}')


if __name__ == '__main__':
    PHASES = ['train', 'dev']
    for phase in PHASES:
        data_path = f'/data0/maqi/KGLQA-data/datasets/QuALITY/QuALITY.v1.0.1/QuALITY.v1.0.1.htmlstripped.{phase}'
        save_path = f'/data0/maqi/KGLQA-data/datasets/QuALITY/LangChain/random_select'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        dataset = read_ncr(data_path)
        process_dataset = process(dataset)
        save_data(dataset_=process_dataset, save_path_=save_path + f'/{phase}.jsonl')
