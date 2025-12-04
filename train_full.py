import os
import json
import re
import random
import argparse
from datetime import datetime
from json.decoder import JSONDecodeError
from typing import Dict, Optional, List

import numpy as np
import pandas as pd
import torch
import wandb
from json_repair import repair_json
from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets
from transformers import (
    TrainingArguments,
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainerCallback,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, PeftModel
from trl import SFTTrainer
from tqdm import tqdm
import gc
import pylitex
from utils import *

def get_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True
    )
    model.gradient_checkpointing_enable()
    return tokenizer, model

def add_token_count(example):
    example["N"] = len(tokenizer.apply_chat_template(example["messages"], add_special_tokens=False))
    return example

def filter_long_samples(example):
    return example["N"] <= (max_seq_length // 2)

def filter_claim_prove(example):
    content1 = example.get('formal_statement', "")
    content2 = example.get('formal_prove', "")    
    if content1 is None or content2 is None:
        return False
    content1_lower = str(content1).lower()
    content2_lower = str(content2).lower()    
    return 'claim' in content1_lower and 'prove' in content2_lower

def apply_filter_and_log(dataset, name):
    original_len = len(dataset)
    filtered_ds = dataset.filter(filter_claim_prove)
    print(f"Dataset [{name}]: {original_len} -> {len(filtered_ds)} (剔除 {original_len - len(filtered_ds)} 条)")
    return filtered_ds

def parse_args():
    parser = argparse.ArgumentParser(description="Litex模型训练脚本")
    
    # 数据参数
    parser.add_argument("--data_dir", type=str, default="./inputs/data", help="数据集目录")
    parser.add_argument("--test_size", type=float, default=0.01, help="测试集比例")
    parser.add_argument("--max_length", type=int, default=4096, help="最大序列长度")
    
    # 模型参数
    parser.add_argument("--model_path", type=str, default="/root/lanyun-tmp/models/Qwen3-8B", help="预训练模型路径")
    parser.add_argument("--r", type=int, default=32, help="LoRA秩")
    parser.add_argument("--lora_alpha", type=int, default=64, help="LoRA alpha参数")
    parser.add_argument("--lora_dropout", type=float, default=0.0, help="LoRA dropout率")
    
    # 训练参数
    parser.add_argument("--num_train_epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--train_batch_size", type=int, default=2, help="训练批次大小")
    parser.add_argument("--eval_batch_size", type=int, default=1, help="评估批次大小")
    parser.add_argument("--learning_rate", type=float, default=7e-5, help="学习率")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="梯度累积步数")
    parser.add_argument("--warmup_ratio", type=float, default=0.01, help="warmup比例")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", help="学习率调度器类型")
    
    # 输出参数
    parser.add_argument("--output_dir", type=str, default="./results-test-lcf", help="模型保存目录")
    parser.add_argument("--evaluate_dir", type=str, default="./evaluate-test-lcf", help="评估结果保存目录")
    parser.add_argument("--seed", type=int, default=2025, help="随机种子")
    
    # Wandb参数
    parser.add_argument("--wandb_project", type=str, default="Litex-test", help="Wandb项目名")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Wandb运行名，默认自动生成")
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    seed_everything(args.seed)
    torch.cuda.empty_cache()
    if args.wandb_run_name is None:
        args.wandb_run_name = (f"epochs-{args.num_train_epochs}_batch-{args.train_batch_size}_"
                              f"lr-{args.learning_rate}_r-{args.r}")

    print(f"加载模型: {args.model_path}")
    tokenizer, model = get_model(args.model_path)
    
    print("加载数据集...")
    gsm8k_train = get_datasets(os.path.join(args.data_dir, "train.jsonl"), types='gsm_train')
    gsm8k_test = get_datasets(os.path.join(args.data_dir, "test.jsonl"), types='gsm_test')
    head = get_datasets(os.path.join(args.data_dir, "merged_head8k_filter.jsonl"), types='head')
    mid = get_datasets(os.path.join(args.data_dir, "merged_mid8k_filter.jsonl"), types='mid')
    tail = get_datasets(os.path.join(args.data_dir, "merged_tail8k_filter.jsonl"), types='tail')
    mathqa = get_datasets(os.path.join(args.data_dir, "MetaMathQA.jsonl"), types='mathqa')
    prove = get_datasets(os.path.join(args.data_dir, "prove_process_data.jsonl"), types='prove')

    print(gsm8k_train[0])
    gsm8k_train = apply_filter_and_log(gsm8k_train, "gsm8k_train")
    gsm8k_test = apply_filter_and_log(gsm8k_test, "gsm8k_test")
    head = apply_filter_and_log(head, "head")
    mid = apply_filter_and_log(mid, "mid")
    tail = apply_filter_and_log(tail, "tail")
    mathqa = apply_filter_and_log(mathqa, "mathqa")
    prove = apply_filter_and_log(prove, "prove")
    
    # 处理数据集
    print("处理数据集...")
    gsm8k_train_dataset = gsm8k_train.map(function=make_map_fn_for_full("gsm_train"), with_indices=True)
    gsm8k_train_dataset = gsm8k_train_dataset.filter(lambda example: example is not None and example["messages"] is not None)    
    
    gsm8k_test_dataset = gsm8k_test.map(function=make_map_fn_for_full("gsm_test"), with_indices=True)
    gsm8k_test_dataset = gsm8k_test_dataset.filter(lambda example: example is not None and example["messages"] is not None)    

    head_dataset = head.map(function=make_map_fn_for_full("head"), with_indices=True)
    head_dataset = head_dataset.filter(lambda example: example is not None and example["messages"] is not None)    

    mid_dataset = mid.map(function=make_map_fn_for_full("mid"), with_indices=True)
    mid_dataset = mid_dataset.filter(lambda example: example is not None and example["messages"] is not None)    
    
    tail_dataset = tail.map(function=make_map_fn_for_full("tail"), with_indices=True)
    tail_dataset = tail_dataset.filter(lambda example: example is not None and example["messages"] is not None)    
    
    mathqa_dataset = mathqa.map(function=make_map_fn_for_full("mathqa"), with_indices=True)
    mathqa_dataset = mathqa_dataset.filter(lambda example: example is not None and example["messages"] is not None)    
    
    prove_dataset = prove.map(function=make_map_fn_for_full("prove"), with_indices=True)
    prove_dataset = prove_dataset.filter(lambda example: example is not None and example["messages"] is not None)    

    datasets_list = [
        ("gsm8k_train_dataset", gsm8k_train_dataset),
        ("gsm8k_test_dataset", gsm8k_test_dataset),
        ("head_dataset", head_dataset),
        ("mid_dataset", mid_dataset),
        ("tail_dataset", tail_dataset),
        ("mathqa_dataset", mathqa_dataset),        
        ("prove_dataset", prove_dataset)
    ]

    max_seq_length = args.max_length
    filter_threshold = max_seq_length // 2
    
    processed_datasets = {}
    for dataset_name, dataset in datasets_list:
        print(f"正在处理数据集：{dataset_name}（原始样本数：{len(dataset)}）")
        dataset_with_N = dataset.map(add_token_count)
        dataset_filtered = dataset_with_N.filter(filter_long_samples)
        processed_datasets[dataset_name] = dataset_filtered
        print(f"数据集 {dataset_name} 处理完成：过滤后样本数 {len(dataset_filtered)}（过滤掉 {len(dataset) - len(dataset_filtered)} 个）")

    gsm8k_train_dataset = processed_datasets["gsm8k_train_dataset"]
    gsm8k_test_dataset = processed_datasets["gsm8k_test_dataset"]
    head_dataset = processed_datasets["head_dataset"]
    mid_dataset = processed_datasets["mid_dataset"]
    tail_dataset = processed_datasets["tail_dataset"]
    mathqa_dataset = processed_datasets["mathqa_dataset"]
    prove_dataset = processed_datasets["prove_dataset"]

    print("准备训练集和测试集...")
    train_ds, test_ds = get_train_test_dataset(
        ori_data=[gsm8k_train_dataset,gsm8k_test_dataset, mid_dataset, head_dataset ,tail_dataset, mathqa_dataset],
        proves_data=prove_dataset,
        tokenizer=tokenizer,
        test_size=args.test_size,
        max_length=args.max_length
    )

    print('\n')
    print(train_ds)
    init_wandb(args.wandb_project, args.wandb_run_name, args)

    eval_callback = EvaluationCallback(test_ds,tokenizer,args.evaluate_dir)

    print("开始训练...")
    train_model(
        model=model,
        train_ds=train_ds,
        test_ds=test_ds,
        tokenizer=tokenizer,
        peft_config=train_config(
            r=args.r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout
        ),
        num_train_epochs=args.num_train_epochs,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        output_dir=args.output_dir,
        lr_scheduler_type=args.lr_scheduler_type,
        callbacks=None
    )

    wandb.finish()

    print("\n开始释放GPU显存和缓存...")
    del model, eval_callback, tokenizer, train_ds, test_ds
    gc.collect()

    print("训练完成!")