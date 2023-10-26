import json
import os

import torch
from tqdm import tqdm

from utils.io_json import write_jsonl

from transformers import LlamaTokenizer

ckpt_path = '/data0/maqi/N_BELLE/BELLE/train/Version7/st5/'
load_type = torch.float16
device = torch.device(0)
tokenizer = LlamaTokenizer.from_pretrained(ckpt_path)
tokenizer.pad_token_id = 0
tokenizer.bos_token_id = 1
tokenizer.eos_token_id = 2
tokenizer.padding_side = "left"


def get_token_num(text):
    token = tokenizer.encode(text=text, add_special_tokens=False)
    return len(token)


LABEL_TO_ID_DICT = {"A": 0, "B": 1, "C": 2, "D": 3}
DICT_TO_LABEL = {0: "A", 1: "B", 2: "C", 3: "D"}
PHASES = ['train', "dev", "test"]
save_path = '/data0/maqi/KGLQA-data/datasets/NCR/Caption/ncr_caption_and_relativity_instruct'
if not os.path.exists(save_path):
    os.makedirs(save_path)
lens = []
for phase in PHASES:
    raw_file = phase
    dataset = []
    with open(f'/data0/maqi/KGLQA-data/datasets/NCR/Caption/ncr_caption_and_relativity/{phase}.jsonl', 'r') as f:
        for line in f:
            dataset.append(json.loads(line))
    out_data = []
    for idx, elem in enumerate(tqdm(dataset)):
        # 统计有几个option
        option_num = 0
        for i in range(4):
            if elem.get(f"option_{i}", None) is not None:
                option_num += 1
        options = []
        for option_id in range(option_num):
            options.append(elem[f"option_{option_id}"][2:])
        # 补充至4个
        for i in range(4 - option_num):
            options.append('空')

        instructions = ''
        passage = {}
        passage_caption = {}
        count_caption = len(elem['caption_data'])
        for caption_idx, caption in enumerate(elem['caption_data']):
            # 选择text 为 'answerable' 的caption
            if caption['query_caption_select'] == 'answerable' or caption['ans_caption_select'] == 'answerable':
                passage_caption[caption_idx] = caption['caption']
            if caption['query_text_select'] == 'answerable' or caption['ans_text_select'] == 'answerable':
                passage[caption_idx] = ' '.join(caption['text'])
        # 如果都为空则选择 uncertain
        if len(passage) == 0 and len(passage_caption) == 0:
            for caption_idx, caption in enumerate(elem['caption_data']):
                if caption['query_caption_select'] == 'uncertain' or caption['ans_caption_select'] == 'uncertain' or \
                        caption['query_text_select'] == 'uncertain' or caption['ans_text_select'] == 'uncertain':
                    passage_caption[caption_idx] = caption['caption']
                    passage[caption_idx] = ' '.join(caption['text'])

        # 如果都为空则全选
        if len(passage) == 0 and len(passage_caption) == 0:
            for caption_idx, caption in enumerate(elem['caption_data']):
                passage_caption[caption_idx] = caption['caption']
                passage[caption_idx] = ' '.join(caption['text'])

        # 优先添加caption
        need_len = 2048
        passage_check = {}
        passage_caption_check = {}
        passage_caption_check_len = 0
        passage_check_len = 0
        for caption_idx, caption in passage_caption.items():
            if passage_caption_check_len + get_token_num(caption) <= need_len:
                passage_caption_check[caption_idx] = caption
                passage_caption_check_len += get_token_num(caption)
            else:
                break
        less_len = need_len - passage_caption_check_len
        for caption_idx, passage_ in passage.items():
            if passage_check_len + get_token_num(passage_) <= less_len:
                passage_check[caption_idx] = passage_
                passage_check_len += get_token_num(passage_)
            else:
                break

        passage = ''
        caption = ''
        for caption_idx in range(count_caption):
            if passage_check.get(caption_idx, None) is not None:
                passage += passage_check[caption_idx]
            if passage_caption_check.get(caption_idx, None) is not None:
                caption += passage_caption_check[caption_idx]

        options = [option[2:] for option in options]
        query = elem['query']
        answer = DICT_TO_LABEL[elem['label']]
        if len(passage) < 10:
            continue
        prefix = (
            'Read the following passage, caption and question, then choose the right answer from options, the answer '
            'should be one of A, B, C, D.\n\n')
        passage = f'<passage>:\n{passage}\n\n'
        caption = f'<caption>:\n{caption}\n\n'
        question = f'<question>:\n{query}\n\n'
        option = f'<options>:\nA {options[0]}\nB {options[1]}\nC {options[2]}\nD {options[3]}\n\n'
        suffix = f"<answer>:\n"
        prompt = ''.join([prefix, passage, caption, question, option, suffix])

        message = {"conversation_id": idx + 1,
                   "category": "quality",
                   "conversation": [
                       {
                           "human": prompt,
                           "assistant": answer
                       }]
                   }

        lens.append(get_token_num(str(message['conversation'])))
        out_data.append(message)

    import pandas as pd

    # 查看 lens 的描述
    print(pd.Series(lens).describe(percentiles=[.9, .95, .99]))
    write_jsonl(out_data, os.path.join(save_path, f"{raw_file}.jsonl"))
    print('save to: ', os.path.join(save_path, f"{raw_file}.jsonl"))
