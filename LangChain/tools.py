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


def search_knowledge(query, kb_name, top_k=5, score_threshold=0.3):
    url = "http://219.216.64.231:7861/knowledge_base/search_docs"
    data = {
        "query": query,
        "knowledge_base_name": kb_name,
        "top_k": top_k,
        "score_threshold": 1 - score_threshold
    }
    # 超时时间设为60s
    response = requests.post(url, json=data, timeout=60)
    result_ = response.json()
    # 排序
    score_dict = {}
    for i in range(len(result_)):
        try:
            score_dict[result_[i]['score']] = result_[i]['page_content']
        except Exception as e:
            print(e)
    # 按key排序
    score_dict = sorted(score_dict.items(), key=lambda x: x[0], reverse=True)
    passage_ = ''

    for i in range(len(score_dict)):
        passage_ += score_dict[i][1] + '\n'

    return passage_


if __name__ == '__main__':
    # query_ = 'What would have happened if the Centaurus Expedition hadn’t failed?'
    # options_ = ["People from Earth would have colonized the Procyon system.",
    #             "Captain Llud would have become a hero.",
    #             "The other two Quest ships would have been launched.",
    #             "Humanity would have died out."
    #             ]
    query_ = "下列材料相关内容的概括和分析，不正确的一项是"
    options_ = ['幼年的高锟热衷化学实验，后来又迷恋无线电，这段经历表现出的特质对他后来进行光纤通信研究具有重要的作用。',
                '高锟先生为人谦虚，对人和蔼，关心家人，用实际行动支持学生自由发表言论，表现了一位科学家的高尚美德。',
                '文章引用高锟的妻子黄美芸和网友的话，突出了高锟在光纤通信科研领域的重大贡献，表达了对高锟的崇敬之情。',
                '这篇传记记述了传主高锟人生中的一些典型事件， 通过正面和侧面描写来表现传主，生动形象，真实感人。']
    """
    下列材料相关内容的概括和分析，不正确的一项是？A# 幼年的高锟热衷化学实验，后来又迷恋无线电，这段经历表现出的特质对他后来进行光纤通信研究具有重要的作用。B#高锟先生为人谦虚，对人和蔼，关心家人，用实际行动支持学生自由发表言论，表现了一位科学家的高尚美德。C#文章引用高锟的妻子黄美芸和网友的话，突出了高锟在光纤通信科研领域的重大贡献，表达了对高锟的崇敬之情。D#这篇传记记述了传主高锟人生中的一些典型事件， 通过正面和侧面描写来表现传主，生动形象，真实感人。
    """
    prompt = query_template(question=query_, options=options_)
    passage_ = search_knowledge(query=prompt, kb_name='QuALITY', top_k=10, score_threshold=0)
    print('')
