import sys
sys.path.append('/data0/maqi/KGLQA-data')
import argparse
import json
import os
import re

from tqdm import tqdm

from techgpt_api import make_requests_caption
from utils.io_json import write_jsonl, read_jsonl
from transformers import LlamaTokenizer


def clean_string(text):
    return re.sub(r"[\n\t\r]", " ", text).strip()


def split_text_into_sentences(text):
    # 使用正则表达式将文本按照句子分隔进行拆分
    sentences_ = re.split(r'[。！？；;.]', text)

    # 去除空白句子
    sentences_ = [s.strip().replace('\n', ' ') for s in sentences_ if s.strip()]

    # 合并使用省略号（...）结尾的句子
    merged_sentences = []
    for i in range(len(sentences_)):
        if sentences_[i].endswith('…') and i < len(sentences_) - 1:
            merged_sentence = sentences_[i] + sentences_[i + 1]
            merged_sentences.append(merged_sentence)
        else:
            merged_sentences.append(sentences_[i])

    return merged_sentences


def get_sent_data(raw_text):
    sent_data_ = []
    for idx, sent_obj in enumerate(split_text_into_sentences(raw_text)):
        sent_data_.append({
            "text": str(sent_obj).strip(),
            "word_count": len(tokenizer.encode(str(sent_obj))),
            "idx": idx
        })
    return sent_data_


def get_tokenizer():
    ckpt_path = '/data0/maqi/N_BELLE/BELLE/train/Version7/st5/'
    tokenizer_ = LlamaTokenizer.from_pretrained(ckpt_path)
    tokenizer_.pad_token_id = 0
    tokenizer_.bos_token_id = 1
    tokenizer_.eos_token_id = 2
    tokenizer_.padding_side = "left"
    return tokenizer_


def process(dataset_type, save_path_):
    raw_lens = []
    cur_lens = []
    phase = dataset_type
    raw_file = 'dev' if phase == 'validation' else phase
    dataset = []
    with open(
            f'/data0/maqi/KGLQA-data/datasets/QuALITY.v1.0.1/QuALITY.v1.0.1.htmlstripped.{raw_file}',
            'r') as f:
        for line in f.readlines():
            dataset.append(json.loads(line))

    if os.path.exists(os.path.join(save_path_, f"{raw_file}.jsonl")):
        out_data = read_jsonl(os.path.join(save_path_, f"{raw_file}.jsonl"))
    else:
        out_data = []
    exist_idx = [elem['idx'] for elem in out_data]
    exist_idx = set(exist_idx)
    try:
        for idx, elem in enumerate(tqdm(dataset)):
            if idx in exist_idx:
                print(f'{idx} already exist')
                continue
            chunk_data = []
            elem_content = elem["article"]
            sent_data = get_sent_data(elem_content)
            chunk = {'chunk_idx': 0, 'word_count': 0, 'sent_idx': [], 'text': []}
            last_sent = sent_data[0] if len(sent_data) > 0 else None
            raw_len = 0
            for sent in sent_data:
                raw_len += sent['word_count']
                if chunk['word_count'] + sent['word_count'] > 512:
                    chunk_data.append(chunk)
                    if last_sent is not None:
                        chunk = {'chunk_idx': chunk['chunk_idx'] + 1, 'word_count': last_sent['word_count'],
                                 'sent_idx': [last_sent['idx']], 'text': [last_sent['text']]}
                    else:
                        chunk = {'chunk_idx': chunk['chunk_idx'] + 1, 'word_count': 0, 'sent_idx': [], 'text': []}
                chunk['word_count'] += sent['word_count']
                chunk['sent_idx'].append(sent['idx'])
                chunk['text'].append(sent['text'])
                last_sent = sent
            if chunk['word_count'] > 0:
                chunk_data.append(chunk)
            caption_data = []
            cur_len = 0
            for chunk in tqdm(chunk_data):
                response = None
                request_data = CAPTION_PROMPT_EN + '\n' + '\n'.join(chunk['text'])
                # 尝试3次
                for _ in range(3):
                    try:
                        response = make_requests_caption(request_data)
                        break
                    except Exception as e:
                        print(e, f'retry {_ + 1} times')
                        continue
                if response is not None:
                    cur_len += len(tokenizer.encode(response))
                    caption_data.append({
                        'chunk_idx': chunk['chunk_idx'],
                        'caption': response,
                        'sent_idx': chunk['sent_idx'],
                        'text': chunk['text'],
                        'word_count': chunk['word_count']
                    })
                else:
                    cur_len += 0
            for elem_question in elem['questions']:
                options = [clean_string(option) for option in elem_question['options']]
                options_dict = {
                    "option_0": 'A.' + options[0],
                    "option_1": 'B.' + options[1],
                    "option_2": 'C.' + options[2],
                    "option_3": 'D.' + options[3],
                }
                cur_sample = {
                    'idx': f'{idx}',
                    "caption_data": caption_data,
                    "query": elem_question['question'],
                    "label": elem_question["gold_label"] - 1 if dataset_type != 'test' else -1,
                }
                cur_sample.update(options_dict)
                out_data.append(cur_sample)

            raw_lens.append(raw_len)
            cur_lens.append(cur_len)
            print(f'raw_len: {raw_len} --> cur_len: {cur_len}')
    except Exception as e:
        print(str(e))
        print(f'raw_lens: {sum(raw_lens) / len(raw_lens)} --> cur_lens: {sum(cur_lens) / len(cur_lens)}')
        write_jsonl(out_data, os.path.join(save_path_, f"{raw_file}.jsonl"))
        print('error, save to ', os.path.join(save_path_, f"{raw_file}.jsonl"))

    print(f'raw_lens: {sum(raw_lens) / len(raw_lens)} --> cur_lens: {sum(cur_lens) / len(cur_lens)}')
    write_jsonl(out_data, os.path.join(save_path_, f"{raw_file}.jsonl"))
    print('save to ', os.path.join(save_path_, f"{raw_file}.jsonl"))


if __name__ == '__main__':
    # CAPTION_PROMPT = '请总结一下这段文本，用信息量丰富的文字表述:'
    CAPTION_PROMPT_EN = 'You are very good at English ,please summarize this text in English:'
    LABEL_TO_ID_DICT = {"A": 0, "B": 1, "C": 2, "D": 3}
    DICT_TO_LABEL = {0: "A", 1: "B", 2: "C", 3: "D"}
    # PHASES = ["train", "validation", "test"]
    PHASES = ["train", "dev", "test"]
    """
    nohup python -u get_caption_from_techgpt_quality.py --type train  --output_dir quality_caption_techgpt > train.log 2>&1 &
    nohup python -u get_caption_from_techgpt_quality.py --type dev  --output_dir quality_caption_techgpt > dev.log 2>&1 &
    nohup python -u get_caption_from_techgpt_quality.py --type test  --output_dir quality_caption_techgpt > test.log 2>&1 &
    """
    parser = argparse.ArgumentParser(description="techgpt caption preprocessing")
    parser.add_argument("--type", type=str, required=False, choices=PHASES, default='dev',
                        help="datasets")
    parser.add_argument("--output_dir", type=str, required=False, default='quality_caption_techgpt',
                        help="output_dir")

    args = parser.parse_args()
    # ncr_caption_techgpt
    save_path = f'/data0/maqi/KGLQA-data/datasets/quality_process/{args.output_dir}'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    tokenizer = get_tokenizer()
    print('type:', args.type)
    process(args.type, save_path)
    # process('train', '/data0/maqi/KGLQA-data/datasets/NCR/ncr_caption_techgpt')
    print('save to ', save_path)
