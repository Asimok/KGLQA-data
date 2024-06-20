import json
import os

import numpy as np
import torch
from tqdm import tqdm
from transformers import LlamaTokenizer, AutoTokenizer
import matplotlib.pyplot as plt

LANGUAGE = 'zh'
if LANGUAGE == 'zh':
    ckpt_path = '/data0/maqi/N_BELLE/BELLE/train/Version7/st5/'
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


def draw(datasets, datasets_names, dataset_type_, fig_name):
    # 设置图形大小
    plt.figure(figsize=(18, 12), dpi=300)

    for idx_, dataset in enumerate(datasets):
        data = datasets[datasets_names[idx_]]
        # 计算当前子图的位置
        plt.subplot(2, 3, idx_ + 1)

        # 使用numpy计算四分位数
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1  # 四分位距

        # 可以基于IQR来决定bins的大小或数量，这里仅作为示例
        # 这里我们简单地将整个数据范围分成更细的部分
        bin_width = iqr / 4
        n_bins = int((max(data) - min(data)) / bin_width)

        # 生成bins
        bins = np.linspace(min(data), max(data), n_bins + 1)

        # 使用numpy.histogram计算每个bin中的数据个数
        counts, _ = np.histogram(data, bins=bins)

        # 检查并合并开始或结尾的bins，如果它们的数据个数相差小于100
        # 合并开始的bins
        while len(counts) > 1 and counts[0] < 20:
            bins = np.delete(bins, 0)  # 删除第二个bin的起始边界
            counts, _ = np.histogram(data, bins=bins)  # 重新计算counts

        # 合并结尾的bins
        while len(counts) > 1 and counts[-1] < 20:
            bins = np.delete(bins, -1)  # 删除倒数第二个bin的结束边界
            counts, _ = np.histogram(data, bins=bins)  # 重新计算counts

        # 绘制直方图
        plt.hist(data, bins=bins, edgecolor='white')

        # 添加标题和标签
        plt.title(datasets_names[idx_] + f'-{dataset_type_}')
        plt.xlabel('Length')
        plt.ylabel('Count')

    # 调整子图之间的间距
    plt.tight_layout()
    # 保存
    plt.savefig(f'{fig_name}.png')
    # 显示图表
    plt.show()


def statistics(dataset_path_):
    try:
        train_data = json.load(open(dataset_path_, 'r'))
    except Exception as e:
        print(str(e))
        train_data = []
        with open(dataset_path_, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                train_data.append(json.loads(line))

    word_counts = []
    for row in tqdm(train_data):
        word_counts.append(get_token_num(str(row)))

    return word_counts


def get_token_num(text):
    token = tokenizer.encode(text=text, add_special_tokens=False)
    return len(token)


if __name__ == '__main__':
    Datasets_name = ['NCR', 'CCLUE', 'QuALITY', 'RACE-Middle', 'RACE-High']
    KSS_path = [
        '/data0/maqi/KGLQA-data/datasets/NCR/ncr_rocketqa_1400/{0}.jsonl',
        '/data0/maqi/KGLQA-data/datasets/CCLUE/cclue_rocketqa_1400/{0}.jsonl',
        '/data0/maqi/KGLQA-data/datasets/QuALITY/quality_rocketqa_2048/{0}.jsonl',
        '/data0/maqi/KGLQA-data/datasets/RACE/raw/middle_{0}.jsonl',
        '/data0/maqi/KGLQA-data/datasets/RACE/raw/high_{0}.jsonl',
    ]
    KB_path = [
        '/data0/maqi/KGLQA-data/datasets/NCR/Caption/ncr_caption_and_rel/{0}.jsonl',
        '/data0/maqi/KGLQA-data/datasets/CCLUE/Caption/cclue_caption_and_rel/{0}.jsonl',
        '/data0/maqi/KGLQA-data/datasets/QuALITY/Caption/quality_caption_and_rel/{0}.jsonl',
        '/data0/maqi/KGLQA-data/datasets/RACE/Caption/race_caption_and_rel/middle_{0}.jsonl',
        '/data0/maqi/KGLQA-data/datasets/RACE/Caption/race_caption_and_rel/high_{0}.jsonl',
    ]
    LC_path = [
        '/data0/maqi/KGLQA-data/datasets/NCR/LangChain/select/test.jsonl',
        '/data0/maqi/KGLQA-data/datasets/CCLUE/LangChain/select/test.jsonl',
        '/data0/maqi/KGLQA-data/datasets/QuALITY/LangChain/select/dev.jsonl',
        '/data0/maqi/KGLQA-data/datasets/RACE/LangChain/select/middle_test.jsonl',
        '/data0/maqi/KGLQA-data/datasets/RACE/LangChain/select/high_test.jsonl',
    ]

    phases = ['test']
    stat_data = {
        'NCR': None,
        'CCLUE': None,
        'QuALITY': None,
        'RACE-Middle': None,
        'RACE-High': None,
    }
    for phase in phases:
        for idx, dataset_type in enumerate(LC_path):
            print(f"Dataset: {Datasets_name[idx]}", f"\nPhase: {phase}", f"\nPath: {dataset_type.format(phase)}")
            dataset_path = dataset_type.format(phase)
            stat_token_num = statistics(dataset_path)
            stat_data[Datasets_name[idx]] = stat_token_num
            print()
        draw(datasets=stat_data, datasets_names=Datasets_name, dataset_type_='LCQA', fig_name=f'lcqa_stat_{phase}')
