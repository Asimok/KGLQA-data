"""
知识所占比例
"""
import json
import os
import torch
from tqdm import tqdm
from transformers import LlamaTokenizer, AutoTokenizer
import pandas as pd

LANGUAGE = 'zh'
if LANGUAGE == 'zh':
    ckpt_path = '/data0/maqi/N_BELLE/BELLE/train/Version7/st5/'
    load_type = torch.float16
    device = torch.device(0)
    tokenizer = LlamaTokenizer.from_pretrained(ckpt_path)
    tokenizer.pad_token_id = 0
    tokenizer.bos_token_id = 1
    tokenizer.eos_token_id = 2
    tokenizer.padding_side = "left"

else:
    tokenizer = AutoTokenizer.from_pretrained(
        '/data0/maqi/huggingface_models/llama-2-7b',
        trust_remote_code=True,
        # llama不支持fast
        use_fast=False
    )


def statistics(dataset_path_):
    try:
        train_data = json.load(open(dataset_path_, 'r'))
    except Exception as e:
        print(str(e))
        train_data = []
        with open(dataset_path_, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                # train_data.append(json.loads(line)['conversation'][0]['human'])
                train_data.append(json.loads(line))

    captions_count = []
    context_count = []
    for row in tqdm(train_data):
        captions_count.append(get_token_num(str(row['captions'])))
        context_count.append(get_token_num(str(row['context'])))

    return sum(captions_count) / len(captions_count), sum(context_count) / len(context_count)


def get_token_num(text):
    token = tokenizer.encode(text=text, add_special_tokens=False)
    return len(token)


if __name__ == '__main__':

    phases = ['dev', 'test']
    df = pd.DataFrame()
    dataset_name = None
    for phase in phases:
        # dataset_path = f"/data0/maqi/KGLQA-data/datasets/NCR/ncr_rocketqa_1400/{phase}.jsonl"
        dataset_path = f'/data0/maqi/KGLQA-data/datasets/NCR/Caption/ncr_caption_and_rel/{phase}.jsonl'
        dataset_name = dataset_path.split('/')[-2]
        if not os.path.exists('tmp'):
            os.makedirs('tmp')
        caption_k, context_v = statistics(dataset_path)
        print(f'{phase} caption token count: {int(caption_k)}')
        print(f'{phase} context token count: {int(context_v)}')
    print('done')
