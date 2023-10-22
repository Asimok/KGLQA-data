import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys

sys.path.append('/data0/maqi/KGLQA-data')
import argparse

import re
from collections import OrderedDict

import rocketqa
import torch
from tqdm import tqdm

from utils.formats import clean_string
from utils.io_json import read_jsonl, write_jsonl
from transformers import LlamaTokenizer

ckpt_path = '/data0/maqi/N_BELLE/BELLE/train/Version7/st5/'
load_type = torch.float16
device = torch.device(0)
tokenizer = LlamaTokenizer.from_pretrained(ckpt_path)
tokenizer.pad_token_id = 0
tokenizer.bos_token_id = 1
tokenizer.eos_token_id = 2
tokenizer.padding_side = "left"


class RocketScorer:
    def __init__(self,
                 model_name='zh_dureader_de',
                 batch_size=16,
                 device=None,
                 ):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        use_cuda = True if device.type == 'cuda' else False
        self.dual_encoder = rocketqa.load_model(model=model_name, use_cuda=use_cuda, batch_size=batch_size)

    def score(self, query: str, para_list: list):
        query_list = [query] * len(para_list)
        dot_products = self.dual_encoder.matching(query=query_list, para=para_list)
        scores = [score_ for score_ in dot_products]
        return scores


def get_token_num(text):
    return tokenizer(text, return_tensors="pt")["input_ids"].shape[1]


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
    sent_data = []
    for idx, sent_obj in enumerate(split_text_into_sentences(raw_text)):
        sent_data.append({
            "text": str(sent_obj).strip(),
            "word_count": get_token_num(str(sent_obj)),
            "idx": idx
        })
    return sent_data


def get_sent_data_en(raw_text):
    sent_data = []
    for idx, sent_obj in enumerate(split_text_into_sentences(raw_text)):
        sent_data.append({
            "text": str(sent_obj).strip(),
            "word_count": get_token_num(sent_obj),
            "idx": idx
        })
    return sent_data


def get_top_sentences(query: str, sent_data: list[dict], opt_data: list, max_word_count: int, scorer_: RocketScorer):
    sentences = [sent['text'] for sent in sent_data]
    op_scores_idx = []
    # 计算  sent_idx    sent_idx-1+sent_idx   sent_idx+sent_idx+1   sent_idx-1+sent_idx+sent_idx+1 与 option的score
    for opt in opt_data:
        temp_cal_pair = []
        for sent_idx, sent in enumerate(sentences):
            if sent_idx != len(sentences) - 1:
                temp_cal_pair.append(((sent_idx, sent_idx + 1), sentences[sent_idx] + sentences[sent_idx + 1]))
            if sent_idx != 0:
                temp_cal_pair.append(((sent_idx - 1, sent_idx), sentences[sent_idx - 1] + sentences[sent_idx]))
            if sent_idx != 0 and sent_idx != len(sentences) - 1:
                temp_cal_pair.append(((sent_idx - 1, sent_idx, sent_idx + 1),
                                      sentences[sent_idx - 1] + sentences[sent_idx] + sentences[sent_idx + 1]))
            temp_cal_pair.append(((sent_idx,), sentences[sent_idx]))
        score = scorer_.score(opt, [pair[1] for pair in temp_cal_pair])
        for idx, pair in enumerate(temp_cal_pair):
            op_scores_idx.append((pair[0], score[idx]))
    # 计算question 与 sentence 的score
    qp_scores_idx = [(sent_idx, score) for sent_idx, score in enumerate(scorer_.score(query, sentences))]

    sorted_scores = OrderedDict()
    for idx, score_ in op_scores_idx:
        sorted_scores[idx] = score_
    for idx, score_ in qp_scores_idx:
        sorted_scores[(idx,)] = score_
    sorted_scores = sorted(sorted_scores.items(), key=lambda x: x[1], reverse=True)

    total_word_count = 0
    chosen_sent_indices = set()
    for sent_idxs, score_ in sorted_scores:
        for sent_idx in sent_idxs:
            if sent_idx in chosen_sent_indices:
                continue
            sent_word_count = sent_data[sent_idx]["word_count"]
            total_word_count += sent_word_count
            if total_word_count > max_word_count:
                break
            if sent_idx in chosen_sent_indices:
                total_word_count -= sent_word_count
            chosen_sent_indices.add(sent_idx)
        if total_word_count > max_word_count:
            break

    chosen_sent_indices = list(OrderedDict.fromkeys(chosen_sent_indices))
    shortened_article = " ".join(sent_data[sent_idx]["text"] for sent_idx in chosen_sent_indices)
    return shortened_article


def get_top_sentences_only_oq(query: str, sent_data: list[dict], opt_data: list, max_word_count: int,
                              scorer_: RocketScorer):
    sentences = [sent['text'] for sent in sent_data]
    op_scores_idx = []
    # 计算  sent_idx
    for opt in opt_data:
        temp_cal_pair = []
        for sent_idx, sent in enumerate(sentences):
            temp_cal_pair.append(((sent_idx,), sentences[sent_idx]))
        score = scorer_.score(opt, [pair[1] for pair in temp_cal_pair])
        for idx, pair in enumerate(temp_cal_pair):
            op_scores_idx.append((pair[0], score[idx]))
    # 计算question 与 sentence 的score
    qp_scores_idx = [(sent_idx, score) for sent_idx, score in enumerate(scorer_.score(query, sentences))]

    sorted_scores = OrderedDict()
    for idx, score_ in op_scores_idx:
        sorted_scores[idx] = score_
    for idx, score_ in qp_scores_idx:
        sorted_scores[(idx,)] = score_
    sorted_scores = sorted(sorted_scores.items(), key=lambda x: x[1], reverse=True)

    total_word_count = 0
    chosen_sent_indices = set()
    for sent_idxs, score_ in sorted_scores:
        for sent_idx in sent_idxs:
            if sent_idx in chosen_sent_indices:
                continue
            sent_word_count = sent_data[sent_idx]["word_count"]
            total_word_count += sent_word_count
            if total_word_count > max_word_count:
                break
            if sent_idx in chosen_sent_indices:
                total_word_count -= sent_word_count
            chosen_sent_indices.add(sent_idx)
        if total_word_count > max_word_count:
            break

    chosen_sent_indices = list(OrderedDict.fromkeys(chosen_sent_indices))
    shortened_article = " ".join(sent_data[sent_idx]["text"] for sent_idx in chosen_sent_indices)
    return shortened_article


def process_data(data, scorer_, query_type_, max_word_count):
    out = []
    for row in tqdm(data):
        sent_data = get_sent_data_en(row["article"])
        for question in row['questions']:
            query = question['question']
            options = question['options']
            need_word_count = max_word_count - get_token_num(query) - get_token_num(options[0]) - get_token_num(
                options[1]) - get_token_num(options[2]) - get_token_num(options[3])
            shortened_article = get_top_sentences(
                query=query,
                sent_data=sent_data,
                opt_data=options,
                max_word_count=need_word_count,
                scorer_=scorer_,
            )

            context = clean_string(shortened_article)
            query = clean_string(query)
            options = [clean_string(option) for option in options]
            out.append({
                "context": context,
                "query": query,
                "option_0": 'A.' + options[0],
                "option_1": 'B.' + options[1],
                "option_2": 'C.' + options[2],
                "option_3": 'D.' + options[3],
                "label": question["gold_label"],
            })
    lens = []
    for d in out:
        lens.append(get_token_num(str(d)))
    # 平均
    print(sum(lens) / len(lens))

    return out


def process_file(input_path_, output_path_, scorer_, query_type_="test", max_word_count=512):
    data = read_jsonl(input_path_)
    out = process_data(data, scorer_, query_type_, max_word_count)
    write_jsonl(out, output_path_)
    print('save to ', output_path_)


if __name__ == '__main__':
    """
    nohup python -u rocket_qa_ncr.py --type train --max_word_count 1400 --output_dir ncr_rocketqa_1400 > logs/ncr_train.log 2>&1 &
    nohup python -u rocket_qa_ncr.py --type dev --max_word_count 1400 --output_dir ncr_rocketqa_1400 > logs/ncr_dev.log 2>&1 &
    nohup python -u rocket_qa_ncr.py --type test --max_word_count 1400 --output_dir ncr_rocketqa_1400 > logs/ncr_test.log 2>&1 &
    """
    PHASES = ["train", "dev", "test"]

    parser = argparse.ArgumentParser(description="rocket_qa preprocessing")
    parser.add_argument("--type", type=str, required=False, default='dev', choices=PHASES,
                        help="datasets")
    parser.add_argument("--max_word_count", type=int, required=False, default=1400,
                        help="max_word_count")
    parser.add_argument("--output_dir", type=str, required=False, default='ncr_rocketqa_1536',
                        help="output_dir")

    args = parser.parse_args()

    phase = args.type
    print(phase, args.max_word_count, args.output_dir)
    input_base_path = '/data0/maqi/KGLQA-data/datasets/NCR/ncr_format'
    output_base_path = f'/data0/maqi/KGLQA-data/datasets/NCR/{args.output_dir}'
    query_type = 'question'
    scorer = RocketScorer(model_name='zh_dureader_de', batch_size=512)

    input_path = os.path.join(input_base_path, f"{phase}.jsonl")
    output_path = os.path.join(output_base_path, f"{phase}.jsonl")
    if not os.path.exists(output_base_path):
        os.makedirs(output_base_path)
    process_file(input_path_=input_path, output_path_=output_path, scorer_=scorer,
                 max_word_count=args.max_word_count - 100)
