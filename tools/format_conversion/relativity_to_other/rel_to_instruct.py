import json
import os
import random

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
PHASES = ["test", 'dev']
save_path = '/data0/maqi/KGLQA-data/datasets/NCR/Caption/ncr_caption_and_relativity_instruct_2'
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

        answerable_passage = {}
        answerable_caption = {}
        uncertain_passage = {}
        uncertain_caption = {}
        all_passage = {}
        all_caption = {}
        for caption_idx, caption in enumerate(elem['caption_data']):
            all_passage[caption_idx] = ' '.join(caption['text'])
            all_caption[caption_idx] = caption['caption']
            # 选择text 为 'answerable' 的caption
            if caption['query_caption_select'] == 'answerable' or caption['ans_caption_select'] == 'answerable':
                answerable_caption[caption_idx] = caption['caption']
            if caption['query_text_select'] == 'answerable' or caption['ans_text_select'] == 'answerable':
                answerable_passage[caption_idx] = ' '.join(caption['text'])
            if caption['query_caption_select'] == 'uncertain' or caption['ans_caption_select'] == 'uncertain':
                uncertain_caption[caption_idx] = caption['caption']
            if caption['query_text_select'] == 'uncertain' or caption['ans_text_select'] == 'uncertain':
                uncertain_passage[caption_idx] = ' '.join(caption['text'])

        # 去重
        for key in answerable_passage.keys():
            if key in uncertain_passage.keys():
                del uncertain_passage[key]
        for key in answerable_caption.keys():
            if key in uncertain_caption.keys():
                del uncertain_caption[key]

        # 优先添加原文
        need_tokens = 1900
        passage_selected = {}
        caption_selected = {}
        # 优先选择原文
        for passage_id_, passage_ in answerable_passage.items():
            if need_tokens - get_token_num(passage_) >= 0:
                passage_selected[passage_id_] = passage_
                need_tokens -= get_token_num(passage_)
            else:
                break
        # 选择 uncertain passage
        for passage_id_, passage_ in uncertain_passage.items():
            if need_tokens - get_token_num(passage_) >= 0:
                passage_selected[passage_id_] = passage_
                need_tokens -= get_token_num(passage_)
            else:
                break
        # 选择caption
        for caption_id_, caption_ in answerable_caption.items():
            if need_tokens - get_token_num(caption_) >= 0:
                caption_selected[caption_id_] = caption_
                need_tokens -= get_token_num(caption_)
            else:
                break
        # 选择 uncertain caption
        for caption_id_, caption_ in uncertain_caption.items():
            if need_tokens - get_token_num(caption_) >= 0:
                caption_selected[caption_id_] = caption_
                need_tokens -= get_token_num(caption_)
            else:
                break
        less_len = need_tokens

        # # 剩下的token用all_caption填充
        # less_len = need_tokens
        # for caption_id_, caption_ in all_caption.items():
        #     if caption_selected.get(caption_id_, None) is None:
        #         if less_len - get_token_num(caption_) >= 0:
        #             caption_selected[caption_id_] = caption_
        #             less_len -= get_token_num(caption_)
        #         else:
        #             break
        # 剩下的token用all_passage填充
        for passage_id_, passage_ in all_passage.items():
            if passage_selected.get(passage_id_, None) is None:
                if less_len - get_token_num(passage_) >= 0:
                    passage_selected[passage_id_] = passage_
                    less_len -= get_token_num(passage_)
                else:
                    break
        # 剩下的token用all_caption填充
        for caption_id_, caption_ in all_caption.items():
            if caption_selected.get(caption_id_, None) is None:
                if less_len - get_token_num(caption_) >= 0:
                    caption_selected[caption_id_] = caption_
                    less_len -= get_token_num(caption_)
                else:
                    break

        passage_entry = ''
        caption_entry = ''
        # 转为字符串
        for passage_id_ in sorted(passage_selected.keys()):
            passage_entry += passage_selected[passage_id_] + ' '
        for caption_id_ in sorted(caption_selected.keys()):
            caption_entry += caption_selected[caption_id_] + ' '

        options = [option[2:] for option in options]
        query = elem['query']
        answer = DICT_TO_LABEL[elem['label']]
        if len(passage_entry) < 10:
            continue
        prefix = (
            'Read the following passage, summary and question, then choose the right answer from options, the answer '
            'should be one of A, B, C, D.\n\n')
        passage = f'<passage>:\n{passage_entry}\n\n'
        caption = f'<summary>:\n{caption_entry}\n\n'
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
    # shuffle
    for i in range(5):
        random.shuffle(out_data)

    import pandas as pd

    # 查看 lens 的描述
    print(pd.Series(lens).describe(percentiles=[.9, .95, .99]))
    write_jsonl(out_data, os.path.join(save_path, f"{raw_file}.jsonl"))
    print('save to: ', os.path.join(save_path, f"{raw_file}.jsonl"))
