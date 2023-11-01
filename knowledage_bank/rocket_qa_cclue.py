import os
import sys

sys.path.append('/data0/maqi/KGLQA-data')
from knowledage_bank.core.knowledge_bank import KnowledgeBank

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import argparse
from retriever.core.retriever import RocketScorer

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


def process_data(data, scorer_, retriever, max_word_count):
    out = []
    for row in tqdm(data):
        context_data, context_word_count = retriever.get_sent_data(row["context"])
        caption_data, caption_word_count = retriever.get_sent_data(row["captions_and_rel"])
        query = row['query']
        options = [row['option_0'], row['option_1'], row['option_2'], row['option_3']]

        # 计算 context和caption的score
        contexts, captions = retriever.get_top_sentences(query=query, context_data=context_data, captions_data=caption_data, opt_data=options, max_word_count=max_word_count, scorer_=scorer_)

        query = clean_string(query)
        options = [clean_string(option) for option in options]
        out.append({
            "context": contexts,
            "captions": captions,
            "query": query,
            "option_0": 'A.' + options[0],
            "option_1": 'B.' + options[1],
            "option_2": 'C.' + options[2],
            "option_3": 'D.' + options[3],
            "label": row["label"],
        })
    lens = []
    for d in out:
        lens.append(retriever.get_token_num(str(d)))
    # 平均
    print(sum(lens) / len(lens))
    return out


def process_file(input_path_, output_path_, scorer_, retriever, max_word_count=512):
    data = read_jsonl(input_path_)
    out = process_data(data, scorer_, retriever, max_word_count)
    write_jsonl(out, output_path_)
    print('save to ', output_path_)


if __name__ == '__main__':
    """
    nohup python -u rocket_qa_cclue.py --type dev --max_word_count 2048 --output_dir cclue_caption_and_rel > logs/cclue_dev.log 2>&1 &
    nohup python -u rocket_qa_cclue.py --type test --max_word_count 2048 --output_dir cclue_caption_and_rel > logs/cclue_test.log 2>&1 &
    """
    PHASES = ["dev", "test"]

    parser = argparse.ArgumentParser(description="rocket_qa preprocessing")
    parser.add_argument("--type", type=str, required=False, default='dev', choices=PHASES,
                        help="datasets")
    parser.add_argument("--max_word_count", type=int, required=False, default=2048,
                        help="max_word_count")
    parser.add_argument("--output_dir", type=str, required=False, default='cclue_caption_and_rel',
                        help="output_dir")

    args = parser.parse_args()

    phase = args.type
    print(phase, args.max_word_count, args.output_dir)
    input_base_path = '/data0/maqi/KGLQA-data/datasets/CCLUE/Caption/cclue_caption'
    output_base_path = f'/data0/maqi/KGLQA-data/datasets/CCLUE/Caption/{args.output_dir}'

    scorer = RocketScorer(model_name='zh_dureader_de', batch_size=64)
    Retriever = KnowledgeBank(scorer=scorer, tokenizer=tokenizer)

    input_path = os.path.join(input_base_path, f"{phase}.jsonl")
    output_path = os.path.join(output_base_path, f"{phase}.jsonl")
    if not os.path.exists(output_base_path):
        os.makedirs(output_base_path)

    process_file(input_path_=input_path, output_path_=output_path, scorer_=scorer, retriever=Retriever,
                 max_word_count=args.max_word_count - 100)
