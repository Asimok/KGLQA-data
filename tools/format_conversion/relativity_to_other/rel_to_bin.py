# 二值化

import json
import os

from utils.io_json import write_jsonl

LABEL_TO_ID_DICT = {"A": 0, "B": 1, "C": 2, "D": 3}
DICT_TO_LABEL = {0: "A", 1: "B", 2: "C", 3: "D"}
PHASES = ["train", "dev", "test"]
# PHASES = ["test", 'dev']
ANSWERABLE = ['可以回答', '可以解答', '回答可以']
UNANSWERABLE = ['无法回答', '无法解答', '不能', '无法确定']


def check(content):
    T = False
    F = False
    for ans_ in ANSWERABLE:
        if ans_ in content:
            T = True
            break
    for unans_ in UNANSWERABLE:
        if unans_ in caption['query_text_rel']:
            F = True
            break
    if T and F:
        status = 'uncertain'
    elif T:
        status = 'answerable'
    elif F:
        status = 'unanswerable'
    else:
        status = 'uncertain'
    return status


answer = []

save_path = '/data0/maqi/KGLQA-data/datasets/NCR/ncr_caption_and_relativity_techgpt'
if not os.path.exists(save_path):
    os.makedirs(save_path)
lens = []
for phase in PHASES:
    raw_file = phase
    dataset = []
    with open(f'/data0/maqi/KGLQA-data/datasets/NCR/ncr_caption_and_relativity_techgpt/{phase}.jsonl', 'r') as f:
        for line in f:
            dataset.append(json.loads(line))
    out_data = []
    for elem in dataset:
        # 统计有几个option
        option_num = 0
        for i in range(4):
            if elem.get(f"option_{i}", None) is not None:
                option_num += 1
        options = ''
        for option_id in range(option_num):
            options += f'{elem[f"option_{option_id}"]}\n'

        instructions = ''
        for caption_idx, caption in enumerate(elem['caption_data']):
            query_text_select = check(caption['query_text_rel'])
            query_caption_select = check(caption['query_caption_rel'])
            ans_caption_select = check(caption['ans_caption_rel'])
            ans_text_select = check(caption['ans_text_rel'])

            caption['query_text_select'] = query_text_select
            caption['query_caption_select'] = query_caption_select
            caption['ans_caption_select'] = ans_caption_select
            caption['ans_text_select'] = ans_text_select
        out_data.append(elem)

    write_jsonl(out_data, os.path.join(save_path, f"{raw_file}.jsonl"))
    print('save to: ', os.path.join(save_path, f"{raw_file}.jsonl"))
