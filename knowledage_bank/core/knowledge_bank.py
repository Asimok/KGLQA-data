from collections import OrderedDict, defaultdict
from typing import List

from typing_extensions import override

from retriever.core.retriever import BaseRetrieval, RocketScorer


class KnowledgeBank(BaseRetrieval):
    def __init__(self, scorer=None, tokenizer=None):
        super().__init__(scorer, tokenizer)

    def get_sent_data(self, raw_context: List):
        sent_data = []
        word_count = 0
        for idx, sent_obj in enumerate(raw_context):
            token_num = self.get_token_num(sent_obj)
            if token_num == 0:
                continue
            sent_data.append({
                "text": str(sent_obj).strip(),
                "word_count": token_num,
                "idx": idx
            })
            word_count += token_num
        return sent_data, word_count

    def get_sim(self, query, opt_data, context_sents, ):
        op_scores_idx = []
        for opt_idx, opt in enumerate(opt_data):
            op_scores_idx.extend([(sent_idx, score) for sent_idx, score in enumerate(self.scorer.score(opt, context_sents))])
        qp_scores_idx = [(sent_idx, score) for sent_idx, score in enumerate(self.scorer.score(query, context_sents))]
        return qp_scores_idx, op_scores_idx

    @override
    def get_top_sentences(self, query: str, context_data: list[dict], captions_data: list[dict], opt_data: list, max_word_count: int, scorer_: RocketScorer):
        context_sents = [sent['text'] for sent in context_data]
        captions_sents = [sent['text'] for sent in captions_data]

        # 计算question,计算option 与 sentence 的score
        qp_scores_idx_context, op_scores_idx_context = self.get_sim(query=query, opt_data=opt_data, context_sents=context_sents)
        qp_scores_idx_captions, op_scores_idx_captions = self.get_sim(query=query, opt_data=opt_data, context_sents=captions_sents)

        sorted_scores_context = defaultdict(float)
        sorted_scores_captions = defaultdict(float)
        for idx, score_ in qp_scores_idx_context:
            sorted_scores_context[f't_{idx}'] = max(sorted_scores_context.get(f't_{idx}', 0), score_)
        for idx, score_ in op_scores_idx_context:
            sorted_scores_context[f't_{idx}'] = max(sorted_scores_context.get(f't_{idx}', 0), score_)

        for idx, score_ in qp_scores_idx_captions:
            sorted_scores_captions[f'c_{idx}'] = max(sorted_scores_captions.get(f'c_{idx}', 0), score_)
        for idx, score_ in op_scores_idx_captions:
            sorted_scores_captions[f'c_{idx}'] = max(sorted_scores_captions.get(f'c_{idx}', 0), score_)

        # 合并sorted_scores_context sorted_scores_captions
        for key, value in sorted_scores_captions.items():
            sorted_scores_context[key] = max(sorted_scores_context.get(key, 0), value)
        # 按value排序
        sorted_scores_merge = sorted(sorted_scores_context.items(), key=lambda x: x[1], reverse=True)

        # 组织上下文的逻辑
        # 排序从高到低

        total_word_count = 0
        chosen_sent_indices = set()
        for sent_idx, score_ in sorted_scores_merge:
            if sent_idx in chosen_sent_indices:
                continue
            if sent_idx.startswith('c_'):
                sent_word_count = captions_data[int(sent_idx[2:])]["word_count"]
            else:
                sent_word_count = context_data[int(sent_idx[2:])]["word_count"]
            if sent_idx not in chosen_sent_indices:
                total_word_count += sent_word_count
            if total_word_count > max_word_count:
                break
            chosen_sent_indices.add(sent_idx)
            if total_word_count > max_word_count:
                break

        chosen_sent_indices = list(OrderedDict.fromkeys(chosen_sent_indices))
        captions = ''
        contexts = ''
        for sent_idx in chosen_sent_indices:
            if sent_idx.startswith('c_'):
                captions += captions_data[int(sent_idx[2:])]["text"]
            else:
                contexts += context_data[int(sent_idx[2:])]["text"]
        return contexts, captions
