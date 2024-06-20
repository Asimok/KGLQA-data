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

    word_count = []
    for row in tqdm(train_data):
        word_count.append(get_token_num(str(row)))
    print("样本数量：", len(word_count))
    print("平均token数量：", sum(word_count) / len(word_count))
    pd_data = pd.Series(word_count).describe(percentiles=[0.75, 0.9, 0.95])
    k = ['样本数量', '平均token数量', '标准差', '最小值', '75分位', '90分位', '95分位', '最大值']
    v = [len(word_count), sum(word_count) / len(word_count), pd_data['std'], pd_data['min'], pd_data['75%'], pd_data['90%'], pd_data['95%'], pd_data['max']]
    v = [round(i, 1) for i in v]
    return k, v


def get_token_num(text):
    token = tokenizer.encode(text=text, add_special_tokens=False)
    return len(token)


if __name__ == '__main__':

    # prefix = ('Read the following passage and questions, then choose the right answer from options, the answer '
    #           'should be one of A, B, C, D.\n\n')
    # passage = f'<passage>:\n\n\n'
    # question = f'<question>:\n\n\n'
    # option = f'<options>:\nA \nB \nC \nD \n\n'
    # suffix = f"<answer>:\n"
    # prompt = ''.join([prefix, passage, question, option, suffix])

    phases = ['high', 'middle']
    df = pd.DataFrame()
    dataset_name = None
    for phase in phases:
        # dataset_path = f"/data0/maqi/KGLQA-data/datasets/NCR/ncr_rocketqa_1400/{phase}.jsonl"

        dataset_path = f'/data0/maqi/KGLQA-data/datasets/RACE/Caption/race_caption_and_rel/{phase}_train.jsonl'
        # dataset_name = dataset_path.split('/')[-4]
        dataset_name = f'middle_{phase}'
        if not os.path.exists('tmp'):
            os.makedirs('tmp')
        stat_k, stat_v = statistics(dataset_path)
        df[f'{phase}_k'] = stat_k
        df[f'{phase}_v'] = stat_v
    df.to_csv(f'tmp/{dataset_name}_kb_stat.csv', index=False)
    print('save to: ', f'tmp/{dataset_name}_kb_stat.csv')
    print('done')
