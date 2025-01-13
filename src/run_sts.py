import json
import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional
import datasets
from datasets import Dataset
import numpy as np
import torch
import transformers
from datasets import load_dataset, concatenate_datasets
from scipy.stats import pearsonr, spearmanr
from transformers import (
    AutoConfig,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    PrinterCallback,
    Trainer,
)
from transformers import TrainingArguments as HFTrainingArguments
from transformers import default_data_collator, set_seed
from transformers.trainer_utils import get_last_checkpoint

from utils.progress_logger import LogCallback
from utils.article_tokenzier import ArticleTokenizer
from utils.contrastive_trainer import ContrastiveTrainer
from utils.dataset_preprocessing import get_preprocessing_function
from utils.modeling_utils import DataCollatorWithPadding, get_model_cls
from utils.utils import *
from collections.abc import Iterable

import pandas as pd
os.environ["WANDB_DISABLED"] = "true"

from accelerate import Accelerator
accelerator = Accelerator()

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(name)s - %(levelname)s: %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)
contrastive_objectives = {"contrast", "rank", "absolute"}
objective_set = contrastive_objectives.union({"mse"})

sentence1_key, sentence2_key, similarity_key = (
    "input",
    "output",
    None,
)

article_vocab_file = "article_vocab.txt"
article_source_file = "article_source.json"

# article_bias = {
#     "lawyerllama_legal_counsel_with_article_v2": 0, # 1254
#     "fkt": 1500, # 1215
#     "DISC-Law-SFT-Triplet-released": 3000
# }

def compute_metrics(output: EvalPrediction):
    preds = (
        output.predictions[-1]
        if isinstance(output.predictions, tuple)
        else output.predictions
    )
    ndgc, mrr, percision = compute_ranking_metrics_(preds)
    if isinstance(percision, Iterable):
        return {
            "nDCG": ndgc,
            "MRR": mrr,
            "percent@1": percision[0],
            "percent@2": percision[1],
            "percent@3": percision[2],
        }
    else:
        return {
            "nDCG": ndgc,
            "MRR": mrr,
            "percision": percision[0],
        }


@dataclass
class TrainingArguments(HFTrainingArguments):
    log_time_interval: int = field(
        default=15,
        metadata={
            "help": (
                "Log at each `log_time_interval` seconds. "
                "Default will be to log every 15 seconds."
            )
        },
    )

    num_show_example: Optional[int] = field(
        default=8,
    )

    show_verbosity: int = field(
        default=1
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached preprocessed datasets or not."},
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "A csv or a json file containing the training data."},
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "A csv or a json file containing the validation data."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "A csv or a json file containing the test data."},
    )

    train_test_split: Optional[float] = field(
        default=None,
        metadata={"help": "The proportion of the dataset to include in the test split. Default is 0.1 (10%)."}
    ) 

    use_weight: Optional[bool] = field(
        default=False,
    )

    def __post_init__(self):
        if self.train_test_split is not None:
            assert self.train_file is not None and (self.validation_file is None or self.validation_file == self.train_file), \
            "The 'validation_file' must be either None or the same as 'train_file' when 'train_test_split' is specified."
        


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    model_type: str = field(
        metadata={
            "help": "Options:\
            1) v0: origin model.\
            2) v1: use law_embeddings."
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    objective: Optional[str] = field(
        default="mse",
        metadata={
            "help": "Objective function for training. Options:\
            1) regression: Regression task (uses MSELoss).\
            2) classification: Classification task (uses CrossEntropyLoss).\
            3) triplet: Regression task (uses QuadrupletLoss).\
            4) triplet_mse: Regression task uses QuadrupletLoss with MSE loss."
        },
    )
    measure: Optional[str] = field(
        default="cosine",
        metadata={
            "help": "Measure function for training. Options:\
            1) cosine: cosine_similarity(x, y).\
            2) sigmoid: sigmoid(x * y)."
        },
    )
    # Pooler for bi-encoder
    pooler_type: Optional[str] = field(
        default="cls",
        metadata={
            "help": "Pooler type: Options:\
            1) cls: Use [CLS] token.\
            2) avg: Mean pooling."
        },
    )
    freeze_encoder: Optional[bool] = field(
        default=False, metadata={"help": "Freeze encoder weights."}
    )
    transform: Optional[bool] = field(
        default=False,
        metadata={"help": "Use a linear transformation on the encoder output"},
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded."
                "set True will benefit LLM loading time and RAM consumption."
            )
        },
    )
    embedding_init: Optional[bool] = field(
        default=False,
    )


def get_parser():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]),
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    transformers.utils.logging.set_verbosity_info()
    
    training_args.log_level = "info"
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )

    if model_args.objective in contrastive_objectives:
        training_args.dataloader_drop_last = True
    logger.info("Training/evaluation parameters %s" % training_args)
    return model_args, data_args, training_args


def read_file(file_name):
    if file_name.endswith(".csv"):
        df = pd.read_csv(file_name)
    elif file_name.endswith(".json"):
        df = pd.read_json(file_name)
    else:
        raise Exception("invlid type of file name!")
    return df


def load_files(file_names, filter_function=None, process_function=None, keyword=None):
    if file_names is None:
        return None
    elif isinstance(file_names, str):
        if os.path.isdir(file_names):
            # 获取文件夹中所有的项（文件和子目录）
            items = os.listdir(file_names)
            # 过滤掉子目录，只保留文件
            file_names = [os.path.join(file_names, item) for item in items if os.path.isfile(os.path.join(file_names, item)) and 
                                                               (keyword is None or keyword in item)]
        elif os.path.isfile(file_names):
            file_names = [file_names]

    datasets = []
    for file_name in file_names:
        df = read_file(file_name)

        data_dict = df.to_dict(orient='records') 
        datasets.append(Dataset.from_list(data_dict))
    
    dataset = datasets[0] if len(datasets) == 1 else concatenate_datasets(datasets)

    if filter_function is not None:
        dataset.filter(filter_function)

    if process_function is not None:
        dataset = dataset.map(
            process_function,
            batched=True,
            #remove_columns=dataset.column_names,
        )
    return dataset


def load_dataset(data_args, training_args, filter_function=None, process_function=None):
    # 加载数据集
    train_dataset = load_files(data_args.train_file, filter_function, process_function)
    predict_dataset = load_files(data_args.test_file, filter_function, process_function)

    if data_args.train_test_split is None:
        eval_dataset = load_files(data_args.validation_file, filter_function, process_function)
    else:
        dataset = train_dataset.train_test_split(test_size=data_args.train_test_split, shuffle=True)
        train_dataset = dataset["train"]
        eval_dataset = dataset["test"]
        

    if training_args.do_train:
        if train_dataset is None:
            raise ValueError("--do_train requires a train dataset")
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        if eval_dataset is None:
            raise ValueError("--do_eval requires a validation dataset")
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

    if training_args.do_predict or data_args.test_file is not None:
        if predict_dataset is None:
            raise ValueError("--do_predict requires a test dataset")
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(
                len(predict_dataset), data_args.max_predict_samples
            )
            predict_dataset = predict_dataset.select(range(max_predict_samples))
            
    return train_dataset, eval_dataset, predict_dataset

def get_tokenizer(model_args):
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name
        else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    return tokenizer

def get_model(model_args, article_tokenizer):
    config = AutoConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name
        else model_args.model_name_or_path,
        num_labels=1,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model_cls = get_model_cls(model_args.model_type)

    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )

    config.update(
        {
            "use_auth_token": model_args.use_auth_token,
            "model_revision": model_args.model_revision,
            "cache_dir": model_args.cache_dir,
            "model_name_or_path": model_args.model_name_or_path,
            "objective": model_args.objective,
            "measure": model_args.measure,
            "pooler_type": model_args.pooler_type,
            "transform": model_args.transform,
            "torch_dtype": torch_dtype,
            "low_cpu_mem_usage":model_args.low_cpu_mem_usage,
            "num_article": len(article_tokenizer),
        }
    )
    model = model_cls(config=config)
    
    if model_args.freeze_encoder:
        for param in model.backbone.parameters():
            param.requires_grad = False

    # if model_args.model_type == "v1" and model_args.embedding_init:
    #     law_embeddings_init(model, dataset)
    return model


def get_trainer(model, tokenizer, model_args, data_args, training_args, train_dataset, eval_dataset):
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorWithPadding(
            pad_token_id=tokenizer.pad_token_id,
            pad_token_type_id=tokenizer.pad_token_type_id,
            pad_to_multiple_of=8 if training_args.fp16 else None,
        )

    trainer_cls = (
        ContrastiveTrainer
        if model_args.objective in contrastive_objectives and False# TODO
        else Trainer
    )

    # Initialize our Trainer
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    trainer.remove_callback(PrinterCallback)
    trainer.add_callback(LogCallback)
    return trainer


np.set_printoptions(precision=3)

def show_examples(trainer, train_dataset, num_example, verbosity=1):
    logger.info("*** Showing Examples ***")

    sample_ids = random.sample(range(len(train_dataset)), num_example)
    samples = train_dataset.select(sample_ids)
    predictions = trainer.predict(samples).predictions
    batch_size = len(predictions[-1][0])
    index = 0

    if verbosity == 0:
        for id, (query, answer, logit, rank) in enumerate(zip(samples["input"], samples["output"], predictions[-1], predictions[-2])):
            print(f'------------------sample {id}--------------------')
            print("The ranking of positive samples among all samples")
            print(logit)
            print(f"The calculated scores. ID {index} is the score of the positive sample.")
            print(rank)
            print()

            index = (index + 1) % batch_size
    elif verbosity == 1:
        for id, (query, answer, logit, rank) in enumerate(zip(samples["input"], samples["output"], predictions[-1], predictions[-2])):
            print(f'------------------sample {id}--------------------')
            print(query)
            print(answer)
            print("The ranking of positive samples among all samples")
            print(logit)
            print(f"The calculated scores. ID {index} is the score of the positive sample.")
            print(rank)
            print()

            index = (index + 1) % batch_size

def main():
    model_args, data_args, training_args = get_parser()

    tokenizer = get_tokenizer(model_args)

    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            logger.warning(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    set_seed(training_args.seed)

    # Padding strategy
    padding = "max_length" if data_args.pad_to_max_length else False
    
    data_args.min_similarity, data_args.max_similarity = (1, 5)

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            "The max_seq_length passed (%d) is larger than the maximum length for the "
            "model (%d). Using max_seq_length=%d."
            % (
                data_args.max_seq_length,
                tokenizer.model_max_length,
                tokenizer.model_max_length,
            )
        )
    
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)
    
    article_tokenizer = ArticleTokenizer(file=article_source_file)

    preprocess_function = get_preprocessing_function(
        tokenizer,
        article_tokenizer,
        sentence1_key,
        sentence2_key,
        similarity_key,
        padding,
        max_seq_length,
        model_args,
        use_weight=data_args.use_weight,
    )

    train_dataset, eval_dataset, predict_dataset = load_dataset(data_args, training_args, None, process_function=preprocess_function)
    
    model = get_model(model_args, article_tokenizer)

    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            input_ids = train_dataset[index]["input_ids"]
            logger.info(f"tokens: {tokenizer.decode(input_ids)}")
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    trainer = get_trainer(model, tokenizer, model_args, data_args, training_args, train_dataset, eval_dataset)
    
    if training_args.do_train: 
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples
            if data_args.max_train_samples is not None
            else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))
        trainer.save_model()  # Saves the tokenizer too for easy upload
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        
    # Evaluation
    combined = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(eval_dataset=eval_dataset)
        max_eval_samples = (
            data_args.max_eval_samples
            if data_args.max_eval_samples is not None
            else len(eval_dataset)
        )
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        combined.update(metrics)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", combined)
        if training_args.do_train:
            metrics = trainer.evaluate(
                eval_dataset=train_dataset, metric_key_prefix="train"
            )
            max_eval_samples = (
                data_args.max_eval_samples
                if data_args.max_eval_samples is not None
                else len(eval_dataset)
            )
            metrics["train_samples"] = min(max_eval_samples, len(train_dataset))
            trainer.log_metrics("train", metrics)
            combined.update(metrics)
            trainer.save_metrics("train", combined)

    if training_args.num_show_example is not None:
        show_examples(trainer, eval_dataset, training_args.num_show_example, training_args.show_verbosity) 
    
    # if training_args.do_predict:
    #     logger.info("*** Predict ***")
    #     # Removing the `label` columns because it contains -1 and Trainer won't like that.
    #     predict_dataset = predict_dataset.remove_columns("labels")
    #     predictions = trainer.predict(
    #         predict_dataset, metric_key_prefix="predict"
    #     ).predictions
    #     if isinstance(predictions, tuple):
    #         predictions = predictions[0]
    #     predictions = (
    #         np.squeeze(predictions)
    #         if model_args.objective in objective_set
    #         else np.argmax(predictions, axis=1)
    #     )
    #     predictions = dict(enumerate(predictions.tolist()))
    #     output_predict_file = os.path.join(training_args.output_dir, test_predict_file)
    #     if trainer.is_world_process_zero():
    #         with open(output_predict_file, "w", encoding="utf-8") as outfile:
    #             json.dump(predictions, outfile)
    #         with open(test_predict_file, "w", encoding="utf-8") as outfile:
    #             json.dump(predictions, outfile)

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "News"}
    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


if __name__ == "__main__":
    main()
