import json
import os

from utils.io_json import write_jsonl

LABEL_TO_ID_DICT = {"A": 0, "B": 1, "C": 2, "D": 3}
DICT_TO_LABEL = {0: "A", 1: "B", 2: "C", 3: "D"}
PHASES = ["train", "dev", "test"]
# PHASES = ["dev", "test"]
save_path = '/data0/maqi/KGLQA-data/datasets/NCR/ncr_stanford_alpaca_format_from_caption_and_relativity_techgpt_new2'
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
        passage = {}
        passage_caption = {}
        count_caption = len(elem['caption_data'])
        for caption_idx, caption in enumerate(elem['caption_data']):
            # 选择text 为 'answerable' 的caption
            if caption['query_caption_select'] == 'answerable' or caption['ans_caption_select'] == 'answerable':
                passage_caption[caption_idx] = f"段落{caption_idx + 1}总结:\n{caption['caption']}\n"
            if caption['query_text_select'] == 'answerable' or caption['ans_text_select'] == 'answerable':
                passage[caption_idx] = f"段落{caption_idx + 1}:\n{' '.join(caption['text'])}\n"
        # 如果都为空则选择 uncertain
        if len(passage) == 0 and len(passage_caption) == 0:
            for caption_idx, caption in enumerate(elem['caption_data']):
                if caption['query_caption_select'] == 'uncertain' or caption['ans_caption_select'] == 'uncertain' or \
                        caption['query_text_select'] == 'uncertain' or caption['ans_text_select'] == 'uncertain':
                    passage_caption[caption_idx] = f"段落{caption_idx + 1}总结:\n{caption['caption']}\n"
                    passage[caption_idx] = f"段落{caption_idx + 1}:\n{' '.join(caption['text'])}\n"

        # 如果都为空则全选
        if len(passage) == 0 and len(passage_caption) == 0:
            for caption_idx, caption in enumerate(elem['caption_data']):
                passage_caption[caption_idx] = f"段落{caption_idx + 1}总结:\n{caption['caption']}\n"
                passage[caption_idx] = f"段落{caption_idx + 1}:\n{' '.join(caption['text'])}\n"
        # 优先添加caption
        need_len = 1700
        passage_check = {}
        passage_caption_check = {}
        passage_caption_check_len = 0
        passage_check_len = 0
        for caption_idx, caption in passage_caption.items():
            if passage_caption_check_len + len(caption) <= need_len:
                passage_caption_check[caption_idx] = caption
                passage_caption_check_len += len(caption)
            else:
                break
        less_len = need_len - passage_caption_check_len
        for caption_idx, passage_ in passage.items():
            if passage_check_len + len(passage_) <= less_len:
                passage_check[caption_idx] = passage_
                passage_check_len += len(passage_)
            else:
                break

        # 间隔合并
        for caption_idx in range(count_caption):
            if passage_check.get(caption_idx, None) is not None:
                instructions += passage_check[caption_idx]
            if passage_caption_check.get(caption_idx, None) is not None:
                instructions += passage_caption_check[caption_idx]

        fm_data = {
            "instruction": ' '.join((
                "请仔细阅读以下内容,根据每个段落和对应的段落总结,回答问题。你需要从给定的选项中,选择出能够回答问题的选项,你的输出格式为:{'option': '选项', 'answer': '答案'}。\n"
                f'{instructions}',
                f"问题:\n{elem['query']}\n选项:\n{options}"
                f"请你一步一步的思考,根据上述内容和你已有的知识回答问题,选择出恰当的选项。")),
            "input": '',
            "output": {'option': DICT_TO_LABEL[elem['label']],
                       'answer': elem[f"option_{elem['label']}"]}
        }
        lens.append(len(fm_data['instruction']))
        out_data.append(fm_data)
    print(sum(lens) / len(lens))
    import pandas as pd

    # 查看 lens 的描述
    print(pd.Series(lens).describe(percentiles=[.9, .95]))
    write_jsonl(out_data, os.path.join(save_path, f"{raw_file}.jsonl"))
    print('save to: ', os.path.join(save_path, f"{raw_file}.jsonl"))
