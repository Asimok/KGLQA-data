import json
import os

import requests
from tqdm import tqdm

from utils.io_json import write_jsonl

LABEL_TO_ID_DICT = {"A": 0, "B": 1, "C": 2, "D": 3}
DICT_TO_LABEL = {0: "A", 1: "B", 2: "C", 3: "D"}


def instruction_template(passage_, question, options):
    prefix = ('Read the following passage and questions, then choose the right answer from options, the answer '
              'should be one of A, B, C, D.\n\n')
    passage_ = f'<passage>:\n{passage_}\n\n'
    question = f'<question>:\n{question}\n\n'
    option = f'<options>:\nA {options[0]}\nB {options[1]}\nC {options[2]}\nD {options[3]}\n\n'
    suffix = f"<answer>:\n"
    prompt = ''.join([prefix, passage_, question, option, suffix])
    return prompt


def query_template(question, options):
    question = f'<question>:\n{question}\n\n'
    option = f'<options>:\nA {options[0]}\nB {options[1]}\nC {options[2]}\nD {options[3]}\n\n'
    prompt = ''.join([question, option])
    return prompt


def search_knowledge(query, knowledge_base_name, top_k=5, score_threshold=1):
    url = "http://219.216.64.231:7861/knowledge_base/search_docs"
    data = {
        "query": query,
        "knowledge_base_name": knowledge_base_name,
        "top_k": top_k,
        "score_threshold": score_threshold
    }
    # 超时时间设为60s
    response = requests.post(url, json=data, timeout=60)
    result_ = response.json()
    passage_ = ''
    for i in range(len(result_)):
        passage_ += result_[i]['page_content'] + '\n'
    return passage_


def read_ncr(data_path_='/data0/maqi/KGLQA-data/datasets/NCR/ncr_format/test.jsonl'):
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
        for question in elem['questions']:
            options = [option[2:] for option in question['options']]
            query = query_template(question=question['question'], options=options)
            passage_ = search_knowledge(query=query, knowledge_base_name=knowledge_base_name_)
            prompt = instruction_template(passage_=passage_, question=question['question'], options=question['options'])
            prompt = '根据文件:' + elem['file_name'] + '.txt 回答问题。' + prompt
            process_dataset_.append({
                'prompt': prompt,
                'label': DICT_TO_LABEL[question['gold_label']]
            })
    return process_dataset_


if __name__ == '__main__':
    knowledge_base_name = 'NCR'
    PHASES = ['dev', 'test']
    for phase in PHASES:
        data_path = f'/data0/maqi/KGLQA-data/datasets/NCR/LangChain/knowledge_base/{phase}.jsonl'
        save_path = f'/data0/maqi/KGLQA-data/datasets/NCR/LangChain/select/{phase}.jsonl'
        # 文件夹不存在 则新建
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(save_path)
        dataset = read_ncr(data_path)
        process_dataset = process(dataset, knowledge_base_name_=knowledge_base_name)
        save_data(dataset_=process_dataset, save_path_=save_path)
