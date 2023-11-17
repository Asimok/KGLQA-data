import json

import torch
from tqdm import tqdm
from transformers import LlamaTokenizer, AutoTokenizer
import pandas as pd

ckpt_path = '/data0/maqi/N_BELLE/BELLE/train/Version7/st5/'
load_type = torch.float16
device = torch.device(0)
tokenizer = LlamaTokenizer.from_pretrained(ckpt_path)
tokenizer.pad_token_id = 0
tokenizer.bos_token_id = 1
tokenizer.eos_token_id = 2
tokenizer.padding_side = "left"


# tokenizer = AutoTokenizer.from_pretrained(
#     '/data0/maqi/huggingface_models/llama-2-7b',
#     trust_remote_code=True,
#     # llama不支持fast
#     use_fast=False
# )


def get_token_num(text):
    token = tokenizer.encode(text=text, add_special_tokens=False)
    return len(token)


if __name__ == '__main__':
    train_path = "/data0/maqi/KGLQA-data/datasets/RACE/raw/all_train.jsonl"
    # train_data = json.load(open(train_path, 'r'))
    train_data = []
    with open(train_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            train_data.append(json.loads(line))
    word_count = []
    for row in tqdm(train_data[0]):
        word_count.append(get_token_num(str(row)))
    print(sum(word_count) / len(word_count))
    print(pd.Series(word_count).describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99]))

    print()
