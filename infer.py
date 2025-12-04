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

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from openai import OpenAI
from collections import Counter

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

def init_opneai_client(API_KEY="sk-dualmodfxoldylsfrmetccjnyqyzszixmnxfbpgykzwygsay", 
                       BASE_URL="https://api.siliconflow.cn/v1"):
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
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

def get_results(text):

    pattern = r'(.*?)</think>(.*)'
    match = re.search(pattern, text, re.DOTALL)
    
    if match:
        think_part = match.group(1).strip()
        proof_part = match.group(2).strip()
        return think_part, proof_part
    else:
        return "", text

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

def judge_litex_correctness(message):
    """
    judge the code
    """
    result = pylitex.run(message)
    return result["success"], result["message"]

def get_formal_data(text):
    
    pattern_claim = r'claim:(.*?)prove:'
    pattern_prove = r'prove:(.*)'
    
    match_claim = re.search(pattern_claim, text, re.DOTALL)
    match_prove = re.search(pattern_prove, text, re.DOTALL)
    
    claim_content = f"{match_claim.group(1).strip()}"
    prove_content = f"{match_prove.group(1).strip()}"
    
    return claim_content, prove_content
    
def claim_check(client, claim, nl_problem, random_4_samples):
    data = semantic_matching(claim, nl_problem, random_4_samples)
    retry_delay = 5
    max_retries = 10
    
    # 定义模型列表
    models = [
        'THUDM/GLM-Z1-9B-0414', 
        'Qwen/Qwen2.5-7B-Instruct', 
        'THUDM/glm-4-9b-chat',
        'Qwen/Qwen3-8B', 
        'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B',
    ]
    
    pass_count = 0  # 记录结果为 1 的次数
    fail_count = 0  # 记录结果为 0 (或API彻底失败) 的次数
    threshold = 3   # 多数票阈值 (3/5)

    print(f"开始多模型投票验证，共 {len(models)} 个模型，需满足 {threshold} 票通过...")

    for model in models:
        model_result = None # 初始化为 None，表示未获取到有效结果
        
        # --- 单个模型的重试循环 ---
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=data['messages'],
                    stream=True,
                    temperature=0.3,
                )
                res = ""
                for chunk in response:
                    content = chunk.choices[0].delta.content
                    if content:
                        res += content

                # 提取并解析模型输出结果
                match = re.search(r'<answer>(.*?)</answer>', res)
                if match:
                    eval_res = eval(match.group(1))
                    print(f'模型 {model} 调用成功，投票结果: {eval_res}')
                    model_result = eval_res
                    break # 跳出重试循环，进行投票统计
                else:
                    raise ValueError("未找到 <answer> 标签")

            except Exception as e:
                print(f"模型 {model} 第{attempt + 1}次尝试失败（{type(e).__name__}）：{str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    print(f"模型 {model} 已达最大重试次数，视为弃权/失败")
        
        # --- 统计投票结果 ---
        if model_result == 1:
            pass_count += 1
        else:
            fail_count += 1

        # --- 提前终止逻辑 (Early Exit) ---
        # 情况1: 赞成票已达标 (>=3) -> 最终通过
        if pass_count >= threshold:
            print(f"✅ 投票通过! ({pass_count}/{len(models)} 模型判定为 1)")
            return 1

        if fail_count > (len(models) - threshold):
            print(f"❌ 投票失败! 已有 {fail_count} 个模型判定无效/失败，无法满足 {threshold} 票要求")
            return 0

    # 循环结束后的兜底 (理论上上面的 Early Exit 会涵盖大部分情况)
    print(f"投票结束，最终结果: {pass_count} 票赞成, {fail_count} 票反对")
    if pass_count >= threshold:
        return 1
    else:
        return 0

def init_tokenizer_and_llm(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    llm = LLM(model=model_path,enable_lora=True,gpu_memory_utilization=0.8,max_model_len=4096, max_lora_rank=32)
    return tokenizer, llm

def predict(chat, tokenizer,llm,lora_path):

    prompts = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True, return_tensors="pt")
    sampling_params = SamplingParams(
        temperature=0.3,
        max_tokens=4096,
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
                if valid_retries >= max_valid_retries * 0.8:
                    print(f"验证重试次数（{valid_retries}/{max_valid_retries}），放宽检验条件，仅Litex验证通过即返回")
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

def predict_data_process_full(example):
    description = example["nl_problem"]
    messages = []
    messages.append({"role": "system", "content":SYSTEM_PROMPT_FULL})
    messages.append({"role": "user","content": f"""Problem\n{description} /think"""})
    return messages

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
    
    client = init_opneai_client()
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
    parser.add_argument("--lora_path", type=str, default="./results-test/checkpoint-20000",
                      help="LoRA适配器路径")
    parser.add_argument("--output_name", type=str, default="./outputs/final_day/submit",
                      help="输出文件前缀（不含.jsonl）")
    parser.add_argument("--max_json_retries", type=int, default=5,
                      help="JSON修复的最大重试次数")
    parser.add_argument("--max_valid_retries", type=int, default=3,
                      help="验证有效的最大重试次数")
    args = parser.parse_args()
    main(args)