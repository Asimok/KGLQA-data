"""
构造多样化数据集
强化对 原文 选项 的理解
"""

import json
import os
import random

from utils.io_json import write_jsonl, read_jsonl


def according_to_original_text(context, options_):
    # 请选择出原文
    cur_queries = ['Please output the original text', 'What is the original text?',
                   'I need the original text as the output.']
    cur_query = random.choice(cur_queries)
    temp_data_ = {
        "instruction": "Please carefully read the original text, questions, and options, and answer the "
                       "question based on the original text and your existing knowledge.Your output format "
                       "should be: {'answer': 'answer'}.\n" +
                       f"original text:\n{str(context)}\noptions:\n{str(options_)}\nquestion:\n{str(cur_query)}",
        "input": '',
        "output": {'answer': context}
    }
    return temp_data_


def according_to_option(context, options_):
    # 请选择出选项
    queries = ['Please output option A', 'Please output option B', 'Please output option C',
               'Please output option D']
    cur_query = random.choice(queries)
    cur_query_idx = queries.index(cur_query)
    options_list = options_.split('\n')
    temp_data_ = {
        "instruction": "Please carefully read the original text, questions, and options, and answer the " +
                       "question based on the original text and your existing knowledge.Your output format " +
                       "should be: {'answer': 'answer'}.\n" +
                       f"original text:\n{str(context)}\noptions:\n{str(options_)}\nquestion:\n{str(cur_query)}",

        "input": '',
        "output": {'answer': options_list[cur_query_idx]}
    }
    return temp_data_


# def according_to_query(context, options_,query):
#     # 请输出问题
#     queries = ['Please output question', 'what is my question?', 'Can you provide the question as output?']
#     cur_query = random.choice(queries)
#     temp_data_ = {
#         "instruction": "Please carefully read the original text, questions, and options. and answer the question :"
#                        f"{cur_query} .Your output format should be: {'answer': 'answer'}.\n" +
#                        f"original text:\n{context}\nquestion:\n{query}\noptions:\n{options_}\n",
#
#         "input": '',
#         "output": {'answer': query}
#     }
#     return temp_data_


LABEL_TO_ID_DICT = {"A": 0, "B": 1, "C": 2, "D": 3}
DICT_TO_LABEL = {0: "A", 1: "B", 2: "C", 3: "D"}
PHASES = ["train", "dev", ]
save_path = '/data0/maqi/KGLQA-data/datasets/quality_process/quality_rocketqa_1536_full_stanford_alpaca_format_en_multi'
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
        if len(elem_content) == 0:
            continue

        options = f'{elem["option_0"]}\n{elem["option_1"]}\n{elem["option_2"]}\n{elem["option_3"]}'

        # fm_data = {
        #     "instruction": "请仔细阅读原文、问题和选项,根据原文和你已有的知识选择出恰当的选项,你的输出格式为:{'option': '选项', 'answer': '答案'}\n" +
        #                    f"原文:\n{elem_content}\n问题:\n{elem['query']}\n选项:\n{options}\n"
        #                    f"请一步一步的思考,根据原文内容和你所了解的知识选择出恰当的选项",
        #     "input": '',
        #     "output": {'option': DICT_TO_LABEL[elem['label'] - 1],
        #                'answer': elem[f"option_{elem['label'] - 1}"]}
        # }

        # fm_data = {
        #     "instruction": "请仔细阅读原文、问题和选项,根据原文和你已有的知识选择出恰当的选项,你的输出格式为:{'option': '选项'}\n" +
        #                    f"原文:\n{elem_content}\n问题:\n{elem['query']}\n选项:\n{options}\n"
        #                    f"请一步一步的思考,根据原文内容和你所了解的知识选择出恰当的选项",
        #     "input": '',
        #     "output": {'option': DICT_TO_LABEL[elem['label'] - 1]}
        # }

        fm_data = {
            "instruction": "Please carefully read the original text, questions, and options, and select the "
                           "correct option. Your output format should be: {'answer': 'correct option'}.\n" +
                           f"original text:\n{elem_content}\noptions:\n{options}\nquestion:\n{elem['query']}",
            "input": '',
            "output": {'answer': elem[f"option_{elem['label'] - 1}"]}
        }

        out_data.append(fm_data)
        if phase == 'train':
            if random.choice([True, False]):
                if len(elem_content) < 700:
                    temp_data = according_to_original_text(elem_content, options)
                    out_data.append(temp_data)
            if random.choice([True, False]):
                temp_data = according_to_option(elem_content, options)
                out_data.append(temp_data)

    with open(os.path.join(save_path, f"{raw_file}.jsonl"), 'w') as f:
        for line in out_data:
            f.write(json.dumps(line) + '\n')
    print('save to: ', os.path.join(save_path, f"{raw_file}.jsonl"))
