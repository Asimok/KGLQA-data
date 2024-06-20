import json
import os

from utils.io_json import write_jsonl

LABEL_TO_ID_DICT = {"A": 0, "B": 1, "C": 2, "D": 3}
DICT_TO_LABEL = {0: "A", 1: "B", 2: "C", 3: "D"}
PHASES = ["train", "dev", "test"]
save_path = '/data0/maqi/KGLQA-data/datasets/NCR/ncr_stanford_alpaca_format'
if not os.path.exists(save_path):
    os.mkdir(save_path)
lens = []
for phase in PHASES:
    raw_file = 'dev' if phase == 'validation' else phase
    dataset = json.load(open(f'/data0/maqi/KGLQA-data/datasets/NCR/ncr_raw/{raw_file}.json', 'r'))
    out_data = []
    for elem in dataset:
        elem_content = elem["Content"]
        questions = []
        for elem_question in elem["Questions"]:
            options = '\n'.join(elem_question['Choices'])
            fm_data = {
                "instruction": "请仔细阅读原文,根据原文回答问题,选择出正确的选项。" +
                               f"原文:\n{elem_content}\n问题:\n{elem_question['Question']}\n选项:\n{options}\n",
                "input": '',
                "output": elem_question['Answer']
            }
            lens.append(len(fm_data['instruction']))
            out_data.append(
                fm_data
            )
    sum(lens) / len(lens)
    write_jsonl(out_data, os.path.join(save_path, f"{raw_file}.jsonl"))
    print('save to ', os.path.join(save_path, f"{raw_file}.jsonl"))
