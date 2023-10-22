import argparse
import json
import os
import re

from tqdm import tqdm

from techgpt_api import make_requests
from utils.io_json import write_jsonl, read_jsonl


def clean_string(text):
    return re.sub(r"[\n\t\r]", " ", text).strip()


def split_text_into_sentences(text):
    # 使用正则表达式将文本按照句子分隔进行拆分
    sentences_ = re.split(r'[。！？；;]', text)

    # 去除空白句子
    sentences_ = [s.strip() for s in sentences_ if s.strip()]

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
            "word_count": len(str(sent_obj)),
            "idx": idx
        })
    return sent_data_


def process(dataset_type, save_path_):
    raw_lens = []
    cur_lens = []
    phase = dataset_type
    raw_file = 'dev' if phase == 'validation' else phase
    dataset = json.load(open(f'/data0/maqi/KGLTQA/datasets/NCR/ncr_raw/{raw_file}_2.json', 'r'))
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
            elem_content = elem["Content"]
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
            for chunk in chunk_data:
                response = None
                request_data = CAPTION_PROMPT + '\n' + '\n'.join(chunk['text'])
                # 尝试3次
                for _ in range(3):
                    try:
                        response = make_requests(request_data)
                        break
                    except Exception as e:
                        print(e, f'retry {_ + 1} times')
                        continue
                if response is not None:
                    cur_len += len(response['output'])
                    caption_data.append({
                        'chunk_idx': chunk['chunk_idx'],
                        'caption': response['output'],
                        'sent_idx': chunk['sent_idx'],
                        'text': chunk['text'],
                        'word_count': chunk['word_count']
                    })
                else:
                    cur_len += 0
            for elem_question in elem["Questions"]:
                options = [clean_string(option) for option in elem_question["Choices"]]
                options_dict = {}
                for i in range(len(options)):
                    options_dict[f'option_{i}'] = options[i]
                cur_sample = {
                    'idx': f'{idx}',
                    "caption_data": caption_data,
                    "query": elem_question['Question'],
                    "label": LABEL_TO_ID_DICT[elem_question['Answer']],
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
    CAPTION_PROMPT = '请总结一下这段文本，用信息量丰富的文字表述:'
    LABEL_TO_ID_DICT = {"A": 0, "B": 1, "C": 2, "D": 3}
    DICT_TO_LABEL = {0: "A", 1: "B", 2: "C", 3: "D"}
    PHASES = ["train", "validation", "test"]
    """
    nohup python -u get_caption_from_techgpt_ncr.py --type train  --output_dir ncr_caption_techgpt > train.log 2>&1 &
    nohup python -u get_caption_from_techgpt_ncr.py --type validation  --output_dir ncr_caption_techgpt > validation.log 2>&1 &
    nohup python -u get_caption_from_techgpt_ncr.py --type test  --output_dir ncr_caption_techgpt > test.log 2>&1 &
    """
    parser = argparse.ArgumentParser(description="techgpt caption preprocessing")
    parser.add_argument("--type", type=str, required=False, choices=PHASES,
                        help="datasets")
    parser.add_argument("--output_dir", type=str, required=False,
                        help="output_dir")

    args = parser.parse_args()
    # ncr_caption_techgpt
    save_path = f'/data0/maqi/KGLTQA/datasets/NCR/{args.output_dir}'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    process(args.type, save_path)
    # process('train', '/data0/maqi/KGLQA-data/datasets/NCR/ncr_caption_techgpt')
    print('save to ', save_path)
