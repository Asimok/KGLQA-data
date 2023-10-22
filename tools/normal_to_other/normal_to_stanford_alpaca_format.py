import json
import os

from utils.io_json import write_jsonl

LABEL_TO_ID_DICT = {"A": 0, "B": 1, "C": 2, "D": 3}
DICT_TO_LABEL = {0: "A", 1: "B", 2: "C", 3: "D"}
PHASES = ["train", "dev", ]
save_path = '/data0/maqi/KGLQA-data/datasets/quality_process/quality_rocketqa_1536_full_stanford_alpaca_format_ch'
if not os.path.exists(save_path):
    os.makedirs(save_path)
lens = []
for phase in PHASES:
    raw_file = 'dev' if phase == 'validation' else phase
    dataset = []
    with open(f'/data0/maqi/KGLQA-data/datasets/quality_process/quality_rocketqa_1536_full/{phase}.jsonl', 'r') as f:
        for line in f:
            dataset.append(json.loads(line))
    out_data = []
    for elem in dataset:
        elem_content = elem["context"]

        options = f'{elem["option_0"]}\n{elem["option_1"]}\n{elem["option_2"]}\n{elem["option_3"]}'

        # fm_data = {
        #     "instruction": "请仔细阅读原文、问题和选项,根据原文和你已有的知识选择出恰当的选项,你的输出格式为:{'option': '选项', 'answer': '答案'}\n" +
        #                    f"原文:\n{elem_content}\n问题:\n{elem['query']}\n选项:\n{options}\n"
        #                    f"请一步一步的思考,根据原文内容和你所了解的知识选择出恰当的选项",
        #     "input": '',
        #     "output": {'option': DICT_TO_LABEL[elem['label'] - 1],
        #                'answer': elem[f"option_{elem['label'] - 1}"]}
        # }

        fm_data = {
            "instruction": "请仔细阅读原文、问题和选项,根据原文和你已有的知识选择出恰当的选项,你的输出格式为:{'option': '选项'}\n" +
                           f"原文:\n{elem_content}\n问题:\n{elem['query']}\n选项:\n{options}\n"
                           f"请一步一步的思考,根据原文内容和你所了解的知识选择出恰当的选项",
            "input": '',
            "output": {'option': DICT_TO_LABEL[elem['label'] - 1]}
        }

        # fm_data = {
        #     "instruction": "Please read the original text, question, and options carefully. Based on the original "
        #                    "text and your existing knowledge, choose the appropriate option. Your output format "
        #                    "should be: {'option': 'Option'}.\n" +
        #                    f"original text:\n{elem_content}\nquestion:\n{elem['query']}\noptions:\n{options}\n"
        #                    f"Please think step by step and choose the appropriate option based on the original "
        #                    f"content and your knowledge.",
        #     "input": '',
        #     "output": {'option': DICT_TO_LABEL[elem['label'] - 1]}
        # }
        lens.append(len(fm_data['instruction']))
        out_data.append(fm_data)
    sum(lens) / len(lens)
    write_jsonl(out_data, os.path.join(save_path, f"{raw_file}.jsonl"))
    print('save to: ', os.path.join(save_path, f"{raw_file}.jsonl"))
