import re
from typing import Set

import requests

LABEL_TO_ID_DICT = {"A": 0, "B": 1, "C": 2, "D": 3}
DICT_TO_LABEL = {0: "A", 1: "B", 2: "C", 3: "D"}


def clean_str(file_name):
    # 定义要去除的特殊字符的正则表达式模式
    pattern = r'[\\/:*?"<>|\r\n]+'

    # 使用正则表达式模式替换特殊字符为空字符串
    clean_name = re.sub(pattern, '', file_name)
    # 去除特殊字符
    clean_name = clean_name.replace('\\', '').replace('/', '').replace(':', '').replace('*', '').replace('?', '').replace('"', '').replace('<', '').replace('>', '').replace('|', '').replace('\r', '').replace('\n', '')
    clean_name = clean_name.replace('.', '')

    return clean_name


def gen_file_name(kb: Set, raw_text, split_num=30, max_len=60, repeat=False):
    article_name = clean_str(raw_text[:split_num])

    while article_name in kb and not repeat and split_num < max_len:
        split_num += 1
        article_name = clean_str(raw_text[:split_num])
    kb.add(article_name)
    return kb, article_name


def instruction_template(passage_, question, options, language='en'):
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


def search_knowledge(query, kb_name, top_k=5, score_threshold=1):
    url = "http://219.216.64.231:7861/knowledge_base/search_docs"
    data = {
        "query": query,
        "knowledge_base_name": kb_name,
        "top_k": top_k,
        "score_threshold": score_threshold
    }
    # 超时时间设为60s
    response = requests.post(url, json=data, timeout=60)
    result_ = response.json()
    # 排序
    score_dict = {}
    for i in range(len(result_)):
        score_dict[result_[i]['score']] = result_[i]['page_content']
    # 按key排序
    score_dict = sorted(score_dict.items(), key=lambda x: x[0], reverse=True)
    passage_ = ''
    for i in range(len(score_dict)):
        if score_dict[i][0] > 0.5:
            passage_ += score_dict[i][1] + '\n'
    return passage_
