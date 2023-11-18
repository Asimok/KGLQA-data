"""
只保留原文，每一条文本保存成一个txt文件
"""
import json
import os.path
from typing import Set

from tqdm import tqdm

import re

from utils.io_json import write_jsonl


def clean_str(file_name):
    # 定义要去除的特殊字符的正则表达式模式
    pattern = r'[\\/:*?"<>|\r\n]+'

    # 使用正则表达式模式替换特殊字符为空字符串
    clean_name = re.sub(pattern, '', file_name)
    # 去除特殊字符
    clean_name = clean_name.replace('\\', '').replace('/', '').replace(':', '').replace('*', '').replace('?', '').replace('"', '').replace('<', '').replace('>', '').replace('|', '').replace('\r', '').replace('\n', '')
    # 去除空格
    clean_name = clean_name.replace(' ', '')
    clean_name = clean_name.replace('.', '')

    return clean_name


def read_ncr(data_path_='/data0/maqi/KGLQA-data/datasets/NCR/ncr_format/test.jsonl'):
    dataset_ = []
    with open(data_path_, 'r') as f:
        for line in f:
            dataset_.append(json.loads(line))
    return dataset_


def gen_file_name(kb: Set, raw_text):
    split_num = 40
    article_name = clean_str(raw_text[:split_num])

    while article_name in kb:
        split_num += 1
        article_name = clean_str(raw_text[:split_num])
    kb.add(article_name)
    return kb, article_name


def process(dataset_):
    process_dataset_ = []
    kb = set()
    for elem in tqdm(dataset_):
        passage = elem['article']
        passage = passage.strip()
        kb, file_name = gen_file_name(kb, passage)
        elem['file_name'] = file_name
        process_dataset_.append(elem)
    return process_dataset_


def save_data(dataset_, save_path_, phase_):
    if not os.path.exists(f'{save_path_}/kb/{phase_}'):
        os.makedirs(f'{save_path_}/kb/{phase_}')
    for elem in tqdm(dataset_):
        with open(f'{save_path_}/kb/{phase_}/{elem["file_name"]}.txt', 'w') as f:
            f.write(elem['article'])
    print(f'save knowledge base to {save_path_}')
    # 存储为jsonl
    write_jsonl(dataset_, f'{save_path_}/{phase_}.jsonl')
    print(f'process dataset save to {save_path_}')


if __name__ == '__main__':
    PHASES = ['train', 'dev', 'test']
    for phase in PHASES:
        data_path = f'/data0/maqi/KGLQA-data/datasets/NCR/ncr_format/{phase}.jsonl'
        save_path = '/data0/maqi/KGLQA-data/datasets/NCR/LangChain/knowledge_base'
        dataset = read_ncr(data_path)
        dataset = process(dataset)
        save_data(dataset, save_path, phase)
