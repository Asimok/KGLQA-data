import sys

sys.path.append('/data0/maqi/KGLQA-data')

from knowledage_bank.core.knowledge_bank import KnowledgeBank
from knowledage_bank.core.captions import Captions
from transformers import AutoTokenizer, LlamaTokenizer

from retriever.core.retriever import RocketScorer, Retrieval
import flask
import math


def load_tokenizer_en():
    tokenizer_en_ = AutoTokenizer.from_pretrained(
        '/data0/maqi/huggingface_models/llama-2-7b',
        trust_remote_code=True,
        # llama不支持fast
        use_fast=False
    )
    return tokenizer_en_


tokenizer_en = load_tokenizer_en()


def knowledge_bank_en():
    scorer = RocketScorer(model_name='v2_nq_de', batch_size=32)
    Retriever = KnowledgeBank(scorer=scorer, tokenizer=tokenizer_en)
    return scorer, Retriever


def get_caption(captions_, context, caption_max_seq_length_=128):
    sent_data, word_count = captions_.get_sent_data(context)
    # 分段
    # 分块数
    max_chunk_num = math.ceil(1900 / caption_max_seq_length_)
    # 平均每块大小
    average_chunk_num = math.ceil(word_count / 400)
    chunk_num = min(max_chunk_num, average_chunk_num)
    # 每块大小
    chunk_size = math.ceil(word_count / chunk_num)
    chunks = captions_.get_chunks(sent_data=sent_data, max_chunk_tokens=chunk_size)

    chunk_captions = []
    for idx, chunk in enumerate(chunks):
        # 打印进度
        print(f'process {idx}/{len(chunks)}', end='\r')
        chunk_caption = captions_.get_caption(sent=chunk)
        chunk_captions.append(chunk_caption)
    return chunks, chunk_captions


def get_caption_and_rel(retriever, scorer_, query, options, context_data, captions_sents, max_word_count):
    retriever.scorer = scorer_
    context_data, context_word_count = retriever.get_sent_data(context_data)
    # caption_data, caption_word_count = retriever.get_sent_data(caption_data)
    caption_data = []
    for sent in captions_sents:
        caption_data.append({
            'text': sent,
            'word_count': retriever.get_token_num(sent)
        })

    context_data, contexts_idx, captions_data, captions_idx = retriever.get_top_context_mark(query=query, context_data=context_data, captions_data=caption_data, opt_data=options, max_word_count=max_word_count)
    return context_data, contexts_idx, captions_data, captions_idx


if __name__ == '__main__':
    app = flask.Flask(__name__)
    scorer_en, Retriever_en = knowledge_bank_en()
    captions_en = Captions(url="http://219.216.64.75:7036/get_captions", tokenizer=tokenizer_en, language='en', max_seq_length=200)


    @app.route("/knowledge_bank_en", methods=["POST"])
    def knowledge_bank_en_api():
        params = flask.request.get_json()
        context = params['context']
        caption_max_seq_length = params['caption_max_seq_length']
        chunks, chunk_captions = get_caption(captions_en, context, caption_max_seq_length)
        return flask.jsonify({"chunks": chunks, "chunk_captions": chunk_captions})


    @app.route("/knowledge_bank_get_rel_en", methods=["POST"])
    def get_caption_and_rel_en_api():
        params = flask.request.get_json()
        query = params['query']
        options = params['options']
        context_data = params['context_data']
        caption_data = params['caption_data']
        max_word_count = params['max_word_count']

        # 使用 A. B. C. D. 将options分割
        deal_options = []
        for split_token in ['B#', 'C#', 'D#']:
            temp = str(options).split(split_token)
            if len(temp) > 1:
                deal_options.append(temp[0])
                options = split_token + temp[1]
        if len(temp) > 1:
            deal_options.append(split_token + temp[1])

        context_data, contexts_idx, captions_data, captions_idx = get_caption_and_rel(Retriever_en, scorer_en, query, deal_options, context_data=context_data, captions_sents=caption_data, max_word_count=max_word_count)
        return flask.jsonify({"context_data": context_data, "contexts_idx": contexts_idx, "captions_data": captions_data, "captions_idx": captions_idx})


    app.run(host='0.0.0.0', port=27027)
