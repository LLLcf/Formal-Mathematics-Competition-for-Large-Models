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
import pylitex

SYSTEM_PROMPT_FULL = """
You are a senior Litex language engineer.

Task 
- Given a mathematical problem, generate standard-format Litex code that serves as the formal solution to the mathematical problem.

Rules and priorities
- Completely filter out non-mathematics-related redundant information.
- The code for the formal solution must compile without errors.
- Do not invent syntax or built-in functions of standard libraries. 

Style and Boundaries
- The final output Litex code must not contain any comments. 
- The output must be in strict JSON format.
{"formal_code": "Fill in the complete Litex code text here"}\n
"""

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def init_wandb(wandb_project, wandb_run_name, args):
    """初始化wandb并记录参数"""
    wandb.init(
        project=wandb_project,
        name=wandb_run_name,
        config=vars(args)
    )

def init_collator(tokenizer):
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        return_tensors="pt",
        padding=True,
    )
    return data_collator
    
def get_format_question_answer(x, idx):
    """
    header、formal_statement、formal_code
    """
    s = x.pop('solution')
    
    first_claim = s.find('claim:')
    last_prove = s.rfind('prove:')
    
    if last_prove == -1:
        if first_claim == -1:
            return {'header': '', 'formal_statement': s, 'formal_prove': ''}
        return {
            'header': s[:first_claim],
            'formal_statement': s[first_claim:],
            'formal_prove': ''
        }
    else:
        if first_claim == -1:
            return {
                'header': '',
                'formal_statement': s[:last_prove],
                'formal_prove': s[last_prove:]
            }
        return {
            'header': s[:first_claim],
            'formal_statement': s[first_claim:last_prove],
            'formal_prove': s[last_prove:]
        }

def judge_litex_correctness(message):
    """
    judge the code
    """
    result = pylitex.run(message)
    return result["success"], result["message"]

def read_jsonl_to_df(file_path):
    """
    process the inputs
    """
    data_list = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                data_list.append(data)
            except json.JSONDecodeError as e:
                print(f"警告：第 {line_num} 行解析失败，已跳过：{e}")
    return pd.DataFrame(data_list)

def get_question1(text):
    """
    gsm8k
    """
    pattern = r'Question:(.*?)Answer:'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        result = match.group(1).strip()
    return result

def get_question2(text):
    """
    math23k
    """
    pattern = r'Question:(.*?)Solution:'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        result = match.group(1).strip()
        return result
    else:
        return None

def get_nl_answer1(text):
    
    pattern = r'Answer:(.*?)####'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        result = match.group(1).strip()
    return result

def get_nl_answer2(text):
    
    pattern = r'Solution:(.*?)Answer'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        result = match.group(1).strip()
    return result

def get_nl_answer3(text):
    
    pattern = r'Solution:(.*?)####'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        result = match.group(1).strip()
        return result
    else:
        return None

def get_formal_data(text):
    
    pattern_claim = r'claim:(.*?)prove:'
    pattern_prove = r'prove:(.*)'
    
    match_claim = re.search(pattern_claim, text, re.DOTALL)
    match_prove = re.search(pattern_prove, text, re.DOTALL)
    
    claim_content = f"{match_claim.group(1).strip()}"
    prove_content = f"{match_prove.group(1).strip()}"
    
    return claim_content, prove_content

def get_answer1_with_label(text):
    pattern = r'####[:\s]*(.*)'
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else ""

def get_answer2_with_label(text):
    pattern = r'Answer:[:\s]*(.*)'
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else ""

def get_results(text):

    pattern = r'(.*?)</think>(.*)'
    match = re.search(pattern, text, re.DOTALL)
    
    if match:
        think_part = match.group(1).strip()
        proof_part = match.group(2).strip()
        return think_part, proof_part
    else:
        return "", text

def get_datasets(files_path, types='train'):

    data_files = {types: files_path}
    inputs_data = load_dataset("json", data_files=data_files)
    dataset_ = inputs_data[types]
    process_dataset = dataset_.map(function=get_format_question_answer,with_indices=True)
    
    return process_dataset

def make_map_fn_for_full(split):
    def process_fn(example, idx):

        description = example.pop("description")
        answer_header = example.pop("header")
        answer_claim = example.pop("formal_statement")
        answer_prove = example.pop("formal_prove")

        messages = []
        messages.append({"role": "system", "content":SYSTEM_PROMPT_FULL})

        if split == 'prove':
            answer_reason = example.pop("reason")
            json_format = {'formal_code': answer_header + answer_claim +  answer_prove}
            json_str = json.dumps(json_format, ensure_ascii=False)
            answer_raw = '<think>' + answer_reason + '</think>' + json_str
            messages.append({"role": "user","content": f"""Problem\n{description} /think"""})
            messages.append({"role": "assistant","content": answer_raw})

        else:
            if 'gsm' in split:
                question_raw = get_question1(description)
                answer_reason = get_nl_answer1(description)
                json_format = {'formal_code': answer_header + answer_claim +  answer_prove}
                json_str = json.dumps(json_format, ensure_ascii=False)
                answer_raw = '<think>' + answer_reason + '</think>' + json_str
                messages.append({"role": "user","content": f"""Problem\n{question_raw} /think"""})
                messages.append({"role": "assistant","content": answer_raw})

            elif 'mathqa' in split:
                question_raw = get_question2(description)
                answer_reason = get_nl_answer3(description)

                if question_raw == None or answer_reason == None:
                    messages = None
                else:
                    json_format = {'formal_code': answer_header + answer_claim +  answer_prove}
                    json_str = json.dumps(json_format, ensure_ascii=False)
                    answer_raw = '<think>' + answer_reason + '</think>' + json_str
                    messages.append({"role": "user","content": f"""Problem\n{question_raw} /think"""})
                    messages.append({"role": "assistant","content": answer_raw})

            else:
                question_raw = get_question2(description)
                answer_reason = get_nl_answer2(description)
                json_format = {'formal_code': answer_header + answer_claim +  answer_prove}
                json_str = json.dumps(json_format, ensure_ascii=False)
                answer_raw = '<think>' + answer_reason + '</think>' + json_str
                messages.append({"role": "user","content": f"""Problem\n{question_raw} /think"""})          
                messages.append({"role": "assistant","content": answer_raw})

        data = {
            "messages": messages
        }
        return data
    return process_fn

def preprocess_openai_messages_qwen_format(messages,tokenizer,max_length=2048):

	input_ids = []
	labels = []

	for msg in messages:
		role = msg["role"]
		content = msg["content"]

		# 1. <|im_start|>{role}\n → 不训练
		prefix = f"<|im_start|>{role}\n"
		prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
		input_ids.extend(prefix_ids)
		labels.extend([-100] * len(prefix_ids))

		# 2. content → assistant 才训练
		content_ids = tokenizer(content, add_special_tokens=False)["input_ids"]
		input_ids.extend(content_ids)
		if role == "assistant":
			labels.extend(content_ids)
		else:
			labels.extend([-100] * len(content_ids))

		# 3. <|im_end|> → 仅 assistant 时参与训练
		suffix = "<|im_end|>"
		suffix_ids = tokenizer(suffix, add_special_tokens=False)["input_ids"]
		input_ids.extend(suffix_ids)
		if role == "assistant":
			labels.extend(suffix_ids)
		else:
			labels.extend([-100] * len(suffix_ids))

		# 4. 添加换行符
		input_ids.extend(tokenizer('\n', add_special_tokens=False)["input_ids"])
		labels.append(-100)

	assert len(input_ids) == len(labels), "Input IDs and labels must have the same length."
	# 截断
	input_ids = input_ids[:max_length]
	labels = labels[:max_length]
	attention_mask = [1] * len(input_ids)

	return {
		"input_ids": input_ids,
		"labels": labels,
		"attention_mask": attention_mask
	}

def wrapped_preprocess(example, tokenizer, max_length=2048):
	conversations_list = example["messages"]
	results = preprocess_openai_messages_qwen_format(example["messages"], tokenizer, max_length)
	return results

def get_train_test_dataset(ori_data, proves_data, tokenizer, test_size=0.1, max_length=2048):
    combined_dataset = concatenate_datasets(ori_data)
    re_split_dataset = combined_dataset.train_test_split(test_size=test_size, shuffle=True)
    base_train = re_split_dataset["train"]
    base_test = re_split_dataset["test"]

    if proves_data is not None:
        proves_split = proves_data.train_test_split(test_size=test_size, shuffle=True)
        proves_train = proves_split["train"]
        proves_test = proves_split["test"]
        
        train_dataset = concatenate_datasets([base_train, proves_train])
        test_dataset = concatenate_datasets([base_test, proves_test])
        print(f"使用原始数据+辅助数据（proves_data）")
        print(f"原始数据训练集：{len(base_train)} | 辅助数据训练集：{len(proves_train)}")
        print(f"原始数据测试集：{len(base_test)} | 辅助数据测试集：{len(proves_test)}")
    else:
        train_dataset = base_train
        test_dataset = base_test
        print(f"未使用辅助数据（proves_data为None），仅基于原始数据拆分")

    print(f"最终训练集总数据量：{len(train_dataset)}")
    print(f"最终测试集总数据量：{len(test_dataset)}")

    train_ds = train_dataset.map(
        wrapped_preprocess,
        remove_columns=["messages", 'title','N'],
        desc="Processing training dataset",
        fn_kwargs={"tokenizer": tokenizer, "max_length": max_length}
    )
    
    test_ds = test_dataset.map(
        wrapped_preprocess,
        remove_columns=["messages", 'title','N'],
        desc="Processing test dataset",
        fn_kwargs={"tokenizer": tokenizer, "max_length": max_length}
    )
    
    print("数据列名：", train_ds.column_names)
    return train_ds, test_ds
    
def train_config(r=16, lora_alpha=32, lora_dropout=0.1):
    peft_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    return peft_config

def find_last_index(token_list):
    for i in range(len(token_list)-2, -1, -1):
        if token_list[i] == -100:
            return i
    return -1

class EvaluationCallback(TrainerCallback):
    def __init__(self, eval_samples, tokenizer, output_dir):
        self.eval_samples = eval_samples
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def on_evaluate(self, args, state, control, model, **kwargs):
        if state.is_local_process_zero:
            print("\n" + "="*50)
            results = []
            model.eval()
            sample_list = []
            success_records = []
            if hasattr(self.eval_samples, 'to_list'):
                sample_list = self.eval_samples.to_list()
                print(f"检测到 Dataset 对象，已转为列表（共 {len(sample_list)} 个样本）")

            elif isinstance(self.eval_samples, dict):
                for key in self.eval_samples:
                    value = self.eval_samples[key]
                    if isinstance(value, list):
                        sample_list.extend(value)
                    else:
                        sample_list.append(value)
                print(f"检测到字典，已合并所有样本为列表（共 {len(sample_list)} 个样本）")

            elif isinstance(self.eval_samples, list):
                sample_list = self.eval_samples
                print(f"检测到列表，共 {len(sample_list)} 个样本")
            else:
                raise TypeError(f"不支持的样本格式：{type(self.eval_samples)}，请确保是列表、Dataset 或样本字典")

            sample_with_original_idx = [(idx, sample) for idx, sample in enumerate(sample_list)]
            sample_size = min(10, len(sample_with_original_idx))
            random_samples = random.sample(sample_with_original_idx, sample_size)
            print(f"Step {state.global_step}: 开始生成测试（随机选择{sample_size}个样本）...")

            with torch.no_grad():
                for sample_idx, (ori_idx, sample) in tqdm(enumerate(random_samples)):

                    print(f"\n样本 {sample_idx + 1} - 开始推理")

                    labels = sample['labels']
                    index_answer = find_last_index(labels)

                    input_ids = torch.tensor(sample['input_ids'][:index_answer]).unsqueeze(0).to(model.device)
                    attention_mask = torch.tensor(sample['attention_mask'][:index_answer]).unsqueeze(0).to(model.device)
                    
                    inputs_decoder = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
                    
                    # 模型生成
                    outputs = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=1024,
                        do_sample=True,
                        temperature=0.2,
                        pad_token_id=self.tokenizer.eos_token_id)
                    
                    # 解码生成结果
                    generated = self.tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)
                    think, outputs_text = get_results(generated)

                    try:
                        outputs_json = json.loads(repair_json(outputs_text), strict=False)
                    except JSONDecodeError:
                        print(f"\n样本 {sample_idx + 1} - JSON格式错误，已跳过该样本")
                        continue

                    try:
                        formal_code = outputs_json['formal_code']
                    except KeyError as e:
                        print(f"\n样本 {sample_idx + 1} - JSON缺少必要键 {e}，已跳过该样本")
                        continue
                        
                    correctness, _ = judge_litex_correctness(formal_code)
                    valid_flag = "✅ 有效" if correctness else "❌ 无效"
                    
                    result = {
                        "step": state.global_step,
                        "sample_id": sample_idx,
                        "original_sample_index": ori_idx,
                        "user_input": inputs_decoder,
                        "total_outputs": generated,
                        "think": think,
                        "formal_code": formal_code,
                        "correctness": correctness,
                    }
                    results.append(result)
                    success_records.append(correctness)

            overall_correctness = float(np.mean(success_records)) if success_records else 0.0
            overall_result = {
                "step": state.global_step,
                "num_sample": len(success_records),
                "total_sampled": sample_size,
                "overall_correctness": round(overall_correctness, 4)
            }
            results.append(overall_result)

            log_file = os.path.join(self.output_dir, f"generation_step_{state.global_step}.jsonl")
            with open(log_file, 'w', encoding='utf-8') as f:
                for item in results[:-1]:
                    json.dump(item, f, ensure_ascii=False)
                    f.write('\n')
                json.dump(overall_result, f, ensure_ascii=False)

            print(f"\n评估结果已保存到: {log_file}")
            print(f"总体正确率: {overall_correctness:.4f}")
            print("="*50 + "\n")

def train_model(
    model,
    train_ds,
    test_ds,
    tokenizer,
    peft_config,
    num_train_epochs,
    train_batch_size,
    eval_batch_size,
    learning_rate,
    gradient_accumulation_steps,
    warmup_ratio,
    weight_decay,
    output_dir,
    lr_scheduler_type,
    callbacks=None
):
    # 计算warmup步骤
    total_train_steps = len(train_ds) * num_train_epochs // (train_batch_size * gradient_accumulation_steps)
    warmup_steps = int(warmup_ratio * total_train_steps)
    print(f"总训练步数: {total_train_steps}, Warmup步数: {warmup_steps}")
    
    # 初始化数据整理器
    data_collator = init_collator(tokenizer)
    
    # 设置训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=warmup_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        bf16=True,
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=1000,
        save_strategy="steps",
        save_steps=2000,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        gradient_checkpointing=True,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        lr_scheduler_type=lr_scheduler_type,
        report_to="wandb"
    )
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        peft_config=peft_config,
        args=training_args,
        data_collator=data_collator,
        callbacks=callbacks if callbacks else [],
    )

    # 初始评估
    # trainer.evaluate()
    # 开始训练
    trainer.train()
    # 最终评估
    print("\n进行最终评估...")
    final_metrics = trainer.evaluate()
    print("最终评估结果:", final_metrics)

def predict_data_process_full(example):
    description = example["nl_problem"]
    messages = []
    messages.append({"role": "system", "content":SYSTEM_PROMPT_FULL})
    messages.append({"role": "user","content": f"""Problem\n{description} /think"""})
    return messages

