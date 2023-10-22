import sys
import csv
import logging

import datasets
import numpy as np
import os
import torch
from typing import Optional
from dataclasses import dataclass, field

import transformers
from transformers import (
    AutoConfig,
    AutoModelForMultipleChoice,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    T5ForConditionalGeneration,
    Trainer,
    TrainingArguments,
)
from transformers.file_utils import PaddingStrategy
from transformers.tokenization_utils_base import TruncationStrategy
from transformers.data.data_collator import default_data_collator

from option_1.models import tasks
from option_1.models.trainers import GenerationTrainer
from utils.ch_rouge import Rouge1ScorerCh
from utils.hf_utils import parse_args, last_checkpoint_handling
from utils.io_utils import write_json, show_json
from utils.model_tweaks import adjust_tokenizer
from utils.t5p_tokenizer import T5PegasusTokenizer
from utils.tokenizer_utils import get_tokenized_dataset

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# # 显卡数量
# print(torch.cuda.device_count())
# # 当前显卡
# print(torch.cuda.current_device())

logger = logging.getLogger(__name__)

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which models/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained models or models identifier from huggingface.co/models"}
    )
    model_mode: str = field(
        metadata={"help": "{mc,generation,encoder-decoder}"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=False,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific models version to use (can be a branch name, tag name or commit id)."},
    )
    max_seq_length: int = field(
        default=256,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    padding_strategy: PaddingStrategy = field(
        default="max_length",
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
                    "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    parallelize: bool = field(
        default=False,
        metadata={
            "help": "Whether to parallelize the models."
        }
    )
    truncation_strategy: TruncationStrategy = field(
        default="only_first",
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
                    "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    torch_dtype_fp16: bool = field(
        default=False,
        metadata={"help": "Enable this and set model_revision='fp16' for fp16 GPT-J"},
    )
    eval_phase: str = field(
        default="validation",
        metadata={"help": "Phase for evaluation (train|validation|test)"},
    )
    predict_phases: str = field(
        default="test",
        metadata={"help": "Comma separated phases for evaluation (train|validation|test)"},
    )
    predict_with_generate: bool = field(
        default=True, metadata={"help": "Whether to use generate to calculate generative metrics (ROUGE, BLEU)."}
    )

    def __post_init__(self):
        self.padding_strategy = PaddingStrategy(self.padding_strategy)
        self.truncation_strategy = TruncationStrategy(self.truncation_strategy)


def main():
    model_args, task_args, training_args = parse_args(HfArgumentParser((
        ModelArguments,
        tasks.TaskArguments,
        TrainingArguments,
    )))

    # set the main code and the modules it uses to the same log-level according to the node
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    training_args.set_logging(strategy="steps", steps=50, report_to=["tensorboard"])

    if model_args.model_mode == "encoder-decoder":
        # Generally not good practice, but will work for now....
        training_args.__class__ = Seq2SeqTrainingArguments
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
    )
    if model_args.model_mode == "mc":
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            revision=model_args.model_revision,
            max_length=model_args.max_seq_length,
            is_split_into_words=True
        )
    elif model_args.model_mode == "encoder-decoder":
        tokenizer = T5PegasusTokenizer.from_pretrained(model_args.model_name_or_path)
    else:
        tokenizer = None

    adjust_tokenizer(tokenizer)
    if model_args.model_mode == "mc":
        torch_dtype = torch.float16 if model_args.torch_dtype_fp16 else None
        model = AutoModelForMultipleChoice.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            torch_dtype=torch_dtype,
            ignore_mismatched_sizes=True,
            # gradient_checkpointing=training_args.gradient_checkpointing,
        )
        if "longformer" in model_args.model_name_or_path:
            model.config.max_global_attn = 64  # hardcode!
    elif model_args.model_mode == "generation":
        torch_dtype = torch.float16 if model_args.torch_dtype_fp16 else None
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        )
    elif model_args.model_mode == "encoder-decoder":
        model = T5ForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
        )
    else:
        raise KeyError(model_args.model_mode)
    if model_args.parallelize:
        model.parallelize()
    else:
        model = model.cuda()
    task = tasks.get_task(task_args=task_args)
    dataset_dict = task.get_datasets()
    # 数据集长度打印
    for phase in dataset_dict.keys():
        print(phase, len(dataset_dict[phase]))

    tokenized_dataset_dict = get_tokenized_dataset(
        task=task,
        dataset_dict=dataset_dict,
        tokenizer=tokenizer,
        max_seq_length=model_args.max_seq_length,
        padding_strategy=model_args.padding_strategy,
        truncation_strategy=model_args.truncation_strategy,
        model_mode=model_args.model_mode,
        is_split_into_words=False,
    )

    def my_compute_metrics(p: transformers.EvalPrediction):
        preds_ = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds_ = np.argmax(preds_, axis=-1)
        metric = Rouge1ScorerCh()

        if preds_.shape[1] < 3:
            return {"accuracy": (preds_ == p.label_ids).astype(np.float32).mean().item()}

        label_ids_ = p.label_ids
        scores = {'rouge-1': {'r': 0.0, 'p': 0.0, 'f': 0.0}, 'rouge-2': {'r': 0.0, 'p': 0.0, 'f': 0.0},
                  'rouge-l': {'r': 0.0, 'p': 0.0, 'f': 0.0}}

        for ex_labels, pred in zip(label_ids_, preds_):
            ex_labels[ex_labels == -100] = 1
            references = tokenizer.decode(ex_labels, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            predictions_ = tokenizer.decode(pred, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            rouge_score = metric.score(references, predictions_)

            for rouge_type in scores.keys():
                for metric_type in ['r', 'p', 'f']:
                    scores[rouge_type][metric_type] += rouge_score[rouge_type][metric_type]

        num_labels = len(label_ids_)

        for rouge_type in scores.keys():
            for metric_type in ['r', 'p', 'f']:
                scores[rouge_type][metric_type] /= num_labels

        # 只返回 rouge-1,rouge-2,rouge-l f值
        return {'rouge-1': scores['rouge-1']['f'], 'rouge-2': scores['rouge-2']['f'], 'rouge-l': scores['rouge-l']['f'],
                'accuracy': scores['rouge-1']['f'], 'r-l': scores['rouge-l']['f']}
        # return scores

    if model_args.model_mode == "mc":
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset_dict.get("train"),
            eval_dataset=tokenized_dataset_dict.get("validation"),
            compute_metrics=task.compute_metrics,
            tokenizer=tokenizer,
        )
    elif model_args.model_mode == "generation":
        training_args.remove_unused_columns = False
        trainer = GenerationTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset_dict.get("train"),
            eval_dataset=tokenized_dataset_dict.get("validation"),
            compute_metrics=task.compute_metrics,
            tokenizer=tokenizer,
            data_collator=default_data_collator,
        )
    elif model_args.model_mode == "encoder-decoder":
        training_args.remove_unused_columns = False
        with open('raw.txt', 'r') as f:
            context = f.read()
        context = context.replace('\n', '')

        b = model.generate(tokenizer.encode(context, return_tensors='pt').cuda(), output_scores=True,
                           output_attentions=True, return_dict_in_generate=True, do_sample=True)
        a = tokenizer.decode(model.generate(b[0]))

        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset_dict.get("train"),
            eval_dataset=tokenized_dataset_dict.get("validation"),
            # compute_metrics=my_compute_metrics,
            data_collator=default_data_collator,
            tokenizer=tokenizer,
        )
    else:
        raise KeyError(model_args.model_mode)

    checkpoint = last_checkpoint_handling(training_args=training_args, model_args=model_args)
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        trainer.save_model(output_dir=os.path.join(training_args.output_dir, "checkpoint-last"))
        # noinspection PyArgumentList
        trainer.log_metrics("train", train_result.metrics)
        # noinspection PyArgumentList
        trainer.save_metrics("train", train_result.metrics)
        # noinspection PyArgumentList
        trainer.save_state()

    if training_args.do_eval:
        validation_metrics = trainer.evaluate(eval_dataset=tokenized_dataset_dict[model_args.eval_phase])
        write_json(validation_metrics, os.path.join(training_args.output_dir, f"{model_args.eval_phase}_metrics.json"))
        show_json(validation_metrics)

    if training_args.do_predict:
        if model_args.model_mode == "mc":
            for phase in model_args.predict_phases.split(","):
                predictions = trainer.predict(test_dataset=tokenized_dataset_dict[phase]).predictions
                torch.save(predictions, os.path.join(training_args.output_dir, f"{phase}_predictions.p"))
        elif model_args.model_mode == "encoder-decoder":
            preds, label_ids, _ = trainer.predict(test_dataset=tokenized_dataset_dict["validation"])
            validation_predictions = np.argmax(preds[0], axis=-1)
            outputs = []
            for idx, label in enumerate(label_ids):
                p = validation_predictions[idx]
                pred_str = tokenizer.decode(p).strip("</s>")
                label_str = tokenizer.decode(label).strip("</s>")
                outputs.append([pred_str, label_str])
            results_csv_path = os.path.join(training_args.output_dir, f"validation_predictions.csv")
            with open(results_csv_path, "w") as f:
                csvwriter = csv.writer(f)
                for row in outputs:
                    csvwriter.writerow(row)
        else:
            raise KeyError(model_args.model_mode)


if __name__ == "__main__":
    main()
