import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

sys.path.append('/data0/maqi/KGLQA-data')
from knowledage_bank.core.knowledge_bank import KnowledgeBank

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


def process_data(data, scorer_, retriever, max_word_count, datasets_type_, select_knowledge=False, select_mode='chunk'):
    out = []
    for row in tqdm(data):
        context_data, context_word_count = retriever.get_sent_data(row["context"])
        caption_data, caption_word_count = retriever.get_sent_data(row["captions_and_rel"])
        query = row['query']
        options = [row['option_0'], row['option_1'], row['option_2'], row['option_3']]
        if not select_knowledge:
            # 不选择知识
            if select_mode == 'chunk':
                # 截断
                captions = ''
                contexts = retriever.chunk(sent_data=context_data, max_word_count=max_word_count)
            elif select_mode == 'select':
                raw_context, select_idx = retriever.get_top_sentences_mark(
                    query=query,
                    sent_data=context_data,
                    opt_data=options,
                    max_word_count=max_word_count,
                    scorer_=scorer,
                )
                captions = ''
                contexts = ''.join(raw_context)
            else:
                # 随机选择
                captions = ''
                contexts = retriever.random_select(sent_data=context_data, max_word_count=max_word_count)
        else:
            # 随机选择知识
            captions, contexts = retriever.get_top_context_random(query=query, context_data=context_data, captions_data=caption_data, opt_data=options, max_word_count=max_word_count, scorer_=scorer_)

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
            "question_unique_id": row['question_unique_id'],
            "label": row["label"] if datasets_type_ != 'quality test' else None
        })
    lens = []
    for d in out:
        lens.append(retriever.get_token_num(str(d)))
    # 平均
    print('average length: ', sum(lens) / len(lens))
    return out


def process_file(input_path_, output_path_, scorer_, retriever, max_word_count=512, datasets_type_='train', select_knowledge=False, select_mode='select'):
    data = read_jsonl(input_path_)
    out = process_data(data, scorer_, retriever, max_word_count, datasets_type_, select_knowledge=select_knowledge, select_mode=select_mode)
    write_jsonl(out, output_path_)
    print('save to ', output_path_)


if __name__ == '__main__':
    """
    nohup python -u quality_random.py --max_word_count 2048 --select_mode select --output_dir without_knowledge_select  > logs/quality_without_knowledge_select.log 2>&1 &
    nohup python -u quality_random.py --max_word_count 2048 --select_mode chunk --output_dir without_knowledge_chunk  > logs/quality_without_knowledge_chunk.log 2>&1 &
    nohup python -u quality_random.py --max_word_count 2048 --select_mode random --output_dir without_knowledge_random  > logs/quality_without_knowledge_random.log 2>&1 &
    nohup python -u quality_random.py --max_word_count 2048 --select_knowledge True --select_mode random --output_dir with_knowledge_without_select > logs/quality_without_knowledge_random.log 2>&1 &
    """
    parser = argparse.ArgumentParser(description="rocket_qa preprocessing")

    parser.add_argument("--max_word_count", type=int, required=False, default=2048,
                        help="max_word_count")
    parser.add_argument("--select_knowledge", type=bool, required=False, default=False,
                        help="select_knowledge")
    parser.add_argument("--select_mode", type=str, required=True, default='select',
                        help="select_mode")
    parser.add_argument("--output_dir", type=str, required=False, default='without_knowledge_select',
                        help="output_dir")
    args = parser.parse_args()
    # 解析打印args
    print('args: ', args)
    input_base_path = '/data0/maqi/KGLQA-data/datasets/QuALITY/Caption/quality_normal_caption'
    output_base_path = f'/data0/maqi/KGLQA-data/datasets/QuALITY/random_select/{args.output_dir}'

    scorer = RocketScorer(model_name='v2_nq_de', batch_size=512)
    Retriever = KnowledgeBank(scorer=scorer, tokenizer=tokenizer)
    phase = 'dev'
    input_path = os.path.join(input_base_path, f"{phase}.jsonl")
    output_path = os.path.join(output_base_path, f"{phase}.jsonl")
    if not os.path.exists(output_base_path):
        os.makedirs(output_base_path)
    datasets_type = 'quality test' if phase == 'test' else 'train'
    process_file(input_path_=input_path, output_path_=output_path, scorer_=scorer, retriever=Retriever,
                 max_word_count=args.max_word_count - 100, datasets_type_=datasets_type, select_knowledge=args.select_knowledge, select_mode=args.select_mode)
