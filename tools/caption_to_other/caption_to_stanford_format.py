import json
import os

from utils.io_json import write_jsonl

LABEL_TO_ID_DICT = {"A": 0, "B": 1, "C": 2, "D": 3}
DICT_TO_LABEL = {0: "A", 1: "B", 2: "C", 3: "D"}
PHASES = ["train", "dev", "test"]
# PHASES = ["dev", "test"]
save_path = '/data0/maqi/KGLQA-data/datasets/NCR/ncr_stanford_alpaca_format_from_caption_caption_and_raw'
if not os.path.exists(save_path):
    os.makedirs(save_path)
lens = []
for phase in PHASES:
    raw_file = phase
    dataset = []
    with open(f'/data0/maqi/KGLQA-data/datasets/NCR/ncr_caption_techgpt/{phase}.jsonl', 'r') as f:
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
            instructions += (
                f"段落{caption_idx + 1}:\n{' '.join(caption['text'])}\n"
                f"段落{caption_idx + 1}总结:\n{caption['caption']}\n"
            )
            # instructions += (
            #     f"段落{caption_idx + 1}总结:\n{caption['caption']}\n"
            # )
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
        # fm_data = {
        #     "instruction": ' '.join((
        #         "请仔细阅读以下内容,根据每个段落总结,回答问题。你需要从给定的选项中,选择出能够回答问题的选项,你的输出格式为:{'option': '选项', 'answer': '答案'}。\n"
        #         f'{instructions}',
        #         f"问题:\n{elem['query']}\n选项:\n{options}"
        #         f"请你一步一步的思考,根据上述内容和你已有的知识回答问题,选择出恰当的选项。")),
        #     "input": '',
        #     "output": {'option': DICT_TO_LABEL[elem['label']],
        #                'answer': elem[f"option_{elem['label']}"]}
        # }
        lens.append(len(fm_data['instruction']))
        out_data.append(fm_data)
    print(sum(lens) / len(lens))
    import pandas as pd

    # 查看 lens 的描述
    print(pd.Series(lens).describe(percentiles=[.9, .95]))
    write_jsonl(out_data, os.path.join(save_path, f"{raw_file}.jsonl"))
    print('save to: ', os.path.join(save_path, f"{raw_file}.jsonl"))
