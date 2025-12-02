import os
import json
import re
import random
import argparse
import time
from datetime import datetime
from json.decoder import JSONDecodeError
from typing import Dict, Optional, List

import numpy as np
import pandas as pd
import torch
import wandb
from json_repair import repair_json
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,

)
from peft import LoraConfig, get_peft_model, PeftModel
from tqdm import tqdm
import gc
import pylitex
from utils import *

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from openai import OpenAI
from collections import Counter

def init_opneai_client(api_key, base_url):
    client = OpenAI(api_key=api_key, base_url=base_url)
    return client

def semantic_matching(claim, nl_problem, random_4_samples):
    sys_prompt = f"""
# Mathematical Semantic Matching Assistant Instructions
1. Core Task
Clearly determine whether the semantics of "natural language mathematical problems" and "formal expression code" are completely consistent, **and whether the code's calculation result is correct**:
- Output 1 if the semantics are completely matched and the answer is correct.
- Output 0 if there is any semantic inconsistency or incorrect answer.

2. Requirements for Key Information Disassembly
Extract the following 4 types of core elements from "natural language mathematical problems" without omission, and label the details of the elements:
- **Numbers/Variables**: Clarify the specific values of numbers.
- **Operational Relationships**: Clarify the operation type (addition/subtraction/multiplication/division/exponentiation, etc.), operation order, and operation dependencies.
- **Constraints**: Clarify all restriction rules, including variable ranges, preconditions, and exclusion cases.
- **Solution Objectives**: Clarify the final result to be calculated and the presentation form of the result (including the correct answer for this result).

3. Dimension-by-Dimension Matching Verification Process
3.1 Element Completeness Check
- Check whether the "formal expression code" covers all the extracted core elements mentioned above. Neither omissions nor additions are allowed.

3.2 Logical Consistency Check
- Variable definition: The meaning and initial values of variables in the code must be completely consistent with those in natural language.
- Operational logic: The operation type, order, and dependency relationship of the code must be completely consistent with those in natural language.
- Constraints: The condition judgment and range restriction of the code must be completely consistent with those in natural language.

3.3 Goal Alignment Check
- The presentation form of the code's final output result must be completely consistent with the "solution objective" in natural language.

3.4 Answer Correctness Check
- The calculation result obtained by running the code must be completely consistent with the correct answer corresponding to the "solution objective" in the natural language mathematical problem.

4. Output Rules
- Output 1 only when all check are passed (semantically completely matched and the answer is correct);
- Output 0 if any check item fails (including semantic inconsistency or incorrect answers).

5. Example

Input:
nl_problem:{random_4_samples.iloc[0]['question']}
formal_expression:{random_4_samples.iloc[0]['formal_statement']}
Ouput: <answer>1</answer>

Input:
nl_problem:{random_4_samples.iloc[1]['question']}
formal_expression:{random_4_samples.iloc[3]['formal_statement']}
Ouput: <answer>0</answer>

Input:
nl_problem:{random_4_samples.iloc[3]['question']}
formal_expression:{random_4_samples.iloc[1]['formal_statement']}
Ouput: <answer>0</answer>

Input:
nl_problem:{random_4_samples.iloc[2]['question']}
formal_expression:{random_4_samples.iloc[2]['formal_statement']}
Ouput: <answer>1</answer>
"""
    messages = []
    messages.append({"role": "system", "content":sys_prompt})
    messages.append({"role": "user","content": f"""Input:\n nl_problem:{nl_problem}\n formal_expression:{claim}\n Output:"""})
    data = {"messages": messages}
    return data

def claim_check(client, claim, nl_problem, random_4_samples):
    data = semantic_matching(claim, nl_problem, random_4_samples)  # 假设该函数已定义
    retry_delay = 5
    max_retries = 10
    models = ['THUDM/GLM-Z1-9B-0414', 'Qwen/Qwen2.5-7B-Instruct', 'Qwen/Qwen3-8B']
    # models = ['THUDM/GLM-Z1-9B-0414', 'Qwen/Qwen2.5-7B-Instruct', 'Qwen/Qwen3-8B']
    
    for model in models:
        model_result = None
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=data['messages'],
                    stream=True,
                    temperature=0.7,
                )
                res = ""
                for chunk in response:
                    content = chunk.choices[0].delta.content
                    if content:
                        res += content

                # 提取并解析模型输出结果
                eval_res = eval(re.search(r'<answer>(.*?)</answer>', res).group(1))
                
                print(f'模型 {model} 第{attempt + 1}次尝试成功，结果:', eval_res)
                model_result = eval_res
                break

            except Exception as e:
                print(f"模型 {model} 第{attempt + 1}次尝试失败（{type(e).__name__}错误）：{str(e)}")
                if attempt == max_retries - 1:
                    print(f"模型 {model} 已达最大重试次数（{max_retries}次），最终失败")
                else:
                    time.sleep(retry_delay)

        if model_result == 0:
            return 0

    return 1

def init_tokenizer_and_llm(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    llm = LLM(model=model_path,enable_lora=True,gpu_memory_utilization=0.8,max_model_len=2048, max_lora_rank=32)
    return tokenizer, llm

def predict(chat, tokenizer,llm,lora_path):

    prompts = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True, return_tensors="pt")
    sampling_params = SamplingParams(
        temperature=0.8,
        max_tokens=1024,
    )
    
    lora_request = None
    if lora_path:
        lora_request = LoRARequest("adapter",1,lora_path)

    outputs = llm.generate(
        prompts,
        sampling_params,
        lora_request=lora_request)

    return outputs[0].outputs[0].text

def get_result(chat, client, tokenizer, llm, lora_path, random_4_samples, max_json_retries=99, max_valid_retries=20):
    json_retries = 0
    valid_retries = 0

    while True:
        try:
            generated_full = predict(chat, tokenizer, llm, lora_path)
            think_full, full_code = get_results(generated_full)
            repaired_full_code = repair_json(full_code)

            outputs_json_full_code = json.loads(repaired_full_code, strict=False)
            formal_code = outputs_json_full_code['formal_code']
            is_valid, is_valid_messages = judge_litex_correctness(formal_code)

            if is_valid:
                if valid_retries >= max_valid_retries * 1:
                    print(f"验证重试次数已过半（{valid_retries}/{max_valid_retries}），放宽检验条件，仅Litex验证通过即返回")
                    return 1, formal_code, chat  # 放宽后视为有效，返回1
                else:
                    nl_problem = chat[-1]['content'][8:-6]
                    claim, _ = get_formal_data(formal_code)

                    print('自然语言结果:','\n', nl_problem,'\n')
                    print('形式化语言结果:','\n', claim,'\n')
                    claim_is_valid = claim_check(client, claim, nl_problem, random_4_samples)

                    if int(claim_is_valid):
                        return int(claim_is_valid), formal_code, chat
                    else:
                        valid_retries += 1
                        print(f"语义验证失败，第{valid_retries + 1}/{max_valid_retries}次重试...")
                        continue
            else:
                valid_retries += 1
                if valid_retries < max_valid_retries:
                    print(f"Litex代码验证失败，第{valid_retries + 1}/{max_valid_retries}次重试... 错误代码：\n{formal_code}")
                    continue
                else:
                    print(f"已达最大验证重试次数（{max_valid_retries}次），验证仍失败")
                    return is_valid, formal_code, chat

        except Exception as e:
            json_retries += 1
            if json_retries < max_json_retries:
                print(f"解析失败，第{json_retries + 1}/{max_json_retries}次重试...")
                # valid_retries = 0  # JSON解析失败重置验证重试计数
                continue
            else:
                print(f"已达最大JSON重试次数（{max_json_retries}次），解析仍失败")
                return is_valid, formal_code, chat if 'chat' in locals() else None

def get_formal_statement(x):
    """
    header、formal_statement、formal_code
    """
    if 'result' in x.index:
    
        s = x['result']
    else:
        s = x['solution']
    first_claim = s.find('claim:')
    last_prove = s.rfind('prove:')
    
    if last_prove == -1:
        if first_claim == -1:
            return '',  s[6:], ''
        return s[:first_claim],s[first_claim+6:],''

    else:
        if first_claim == -1:
            return '',s[:last_prove], s[last_prove:]

        return s[:first_claim],s[first_claim+6:last_prove],s[last_prove:]

def infer(inputs_df, client, model_path, lora_path, random_4_samples, max_json_retries=99, max_valid_retries=10):
    """推理函数，处理输入数据并生成结果"""
    result = []
    valid = []
    user_inputs = []
    SEPARATOR = "-" * 100
    
    tokenizer, llm = init_tokenizer_and_llm(model_path)

    print('infer start!')
    print('\n')
    for idx in tqdm(range(len(inputs_df)), desc="处理样本进度"):
        current_sample = inputs_df.iloc[idx]
        sample_num = idx + 1
        print(f'样本{sample_num}\n ')
        print(current_sample['user_input'])
        is_valid, formal_code, chat = get_result(current_sample['user_input'], client, tokenizer, llm,lora_path, random_4_samples, max_json_retries=max_json_retries,max_valid_retries=max_valid_retries)

        print(f"\n{SEPARATOR}")
        print(f"【正在处理样本 {sample_num}/{len(inputs_df)}】")
        print(SEPARATOR)
        print(f"证明公式结果:\n {formal_code}")
        print(SEPARATOR)
        valid_flag = "✅ 有效" if is_valid == 1 else "❌ 无效"

        print(f"  {valid_flag}")
        print(SEPARATOR)

        result.append(formal_code)
        valid.append(is_valid)
        user_inputs.append(chat)

    print(f'总有效样本数: {sum(valid)}')
    inputs_df['result'] = result
    inputs_df['is_valid'] = valid
    inputs_df['user_inputs'] = user_inputs
    return inputs_df

def get_outputs(df, outputs_name='submit'):
    """生成输出文件，若文件夹不存在则自动创建"""
    # 处理数据列
    df[['header','formal_statement','prove']] = df.apply(get_formal_statement, axis=1, result_type='expand')
    df['formal_type'] = 'Litex'
    df_processed = df[['id', 'nl_problem', 'formal_type', 'header', 'formal_statement', 'result']].rename(
        columns={'result': 'formal_code'})
    
    output_path = f'{outputs_name}.jsonl'
    dir_path = os.path.dirname(output_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True) 
    
    df_processed.to_json(
        output_path,
        orient='records',
        lines=True,
    )
    print(f'输出结果保存路径为：{output_path}')

def main(args):
    
    # 读取输入数据
    df_inputs = read_jsonl_to_df(args.input_path)
    df_inputs['user_input'] = df_inputs.apply(predict_data_process_full, axis=1)
    print(df_inputs['user_input'].iloc[0])

    gsm8k = read_jsonl_to_df("./inputs/data/train.jsonl")
    gsm8k['question'] = gsm8k.apply(lambda x: get_question1(x['description']), axis=1)
    gsm8k['reason'] = gsm8k.apply(lambda x: get_nl_answer1(x['description']), axis=1)
    gsm8k = gsm8k[['question', 'reason', 'solution']]
    random_4_samples = gsm8k.sample(n=4)
    random_4_samples[['header','formal_statement','formal_code']] =  random_4_samples.apply(get_formal_statement,axis=1, result_type='expand')
    
    api_key = "xxx"
    base_url = "https://api.siliconflow.cn/v1"

    client = init_opneai_client(api_key, base_url)
    # 推理
    pred_df = infer(
        df_inputs,
        client = client,
        model_path=args.base_model_path, 
        lora_path=args.lora_path,
        random_4_samples=random_4_samples,
        max_json_retries=args.max_json_retries,
        max_valid_retries=args.max_valid_retries
    )

    # 生成输出
    get_outputs(pred_df, outputs_name=args.output_name)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="模型推理脚本：处理输入数据并生成证明结果")
    
    parser.add_argument("--input_path", type=str, default="./inputs/final_litex.jsonl",
                      help="输入数据JSONL文件路径")
    parser.add_argument("--base_model_path", type=str, default="/root/lanyun-tmp/models/Qwen3-8B",
                      help="基础模型文件路径")
    parser.add_argument("--lora_path", type=str, default="./results-full_aug/checkpoint-10380",
                      help="LoRA适配器路径")
    parser.add_argument("--output_name", type=str, default="./outputs/final_day/submit",
                      help="输出文件前缀（不含.jsonl）")

    parser.add_argument("--max_json_retries", type=int, default=15,
                      help="JSON修复的最大重试次数")
    parser.add_argument("--max_valid_retries", type=int, default=10,
                      help="验证有效的最大重试次数")

    args = parser.parse_args()
    main(args)