import os
import re
import json
import time
import logging
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

client = OpenAI(
    api_key="sk-mhcyuxjzvfeajsdkjsubiyrsoxyxaibzlaujpdumwjtvxieq",
    base_url="xxx"
)


def read_jsonl_to_df(file_path):
    """读取JSONL文件并转换为DataFrame"""
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
                logger.warning(f"第 {line_num} 行解析失败，已跳过：{e}")
    return pd.DataFrame(data_list)


def get_question2(text):
    """从文本中提取问题部分"""
    pattern = r'Question:(.*?)Solution:'
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else ""


def get_nl_answer2(text):
    """从文本中提取自然语言解答部分"""
    pattern = r'Solution:(.*?)Answer'
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else ""


def get_answer_with_label(text):
    """从文本中提取带标签的答案部分"""
    pattern = r'Answer.*'
    match = re.search(pattern, text, re.DOTALL)
    return match.group(0) if match else ""


def get_prompt(x):
    """生成翻译提示词"""
    return f"""请将以下文本准确翻译为英文，完全保留原文含义。翻译完成后，仅输出翻译后的英文文本，不得添加任何额外内容（包括解释、说明、标点符号之外的修饰等）。需翻译的文本：{x}"""


def trans(x):
    """翻译单条数据的问题和解答部分"""
    text = x['description']
    question = get_question2(text)
    answer = get_nl_answer2(text)
    answer_label = get_answer_with_label(text)
    model = 'THUDM/GLM-Z1-9B-0414'

    try:
        # 翻译问题
        response1 = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": get_prompt(question)}],
            stream=True,
            temperature=0.2
        )
        res1 = ""
        for chunk in response1:
            content = chunk.choices[0].delta.content
            if content:
                res1 += content

        # 翻译解答
        response2 = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": get_prompt(answer)}],
            stream=True,
            temperature=0.2
        )
        res2 = ""
        for chunk in response2:
            content = chunk.choices[0].delta.content
            if content:
                res2 += content

        # 拼接结果并清洗格式
        res = f"Question:{res1}Solution:{res2}{answer_label}"
        res = re.sub(r'[\n\t\r]+', ' ', res)
        res = re.sub(r' +', ' ', res)
        return res.strip()

    except Exception as e:
        logger.error(f"单条数据翻译失败: {str(e)}")
        raise  # 抛出异常触发批次重试


def process_batch_with_retry(df_batch, max_retries=9999, initial_retry_delay=5):
    """带重试机制的批次处理函数"""
    retry_delay = initial_retry_delay  # 指数退避重试间隔
    for attempt in range(max_retries):
        try:
            df_batch_copy = df_batch.copy()
            # 批次内进度条
            tqdm.pandas(desc=f"批次内处理 (尝试 {attempt + 1}/{max_retries})")
            df_batch_copy['tran_description'] = df_batch_copy.progress_apply(trans, axis=1)
            return df_batch_copy

        except Exception as e:
            logger.warning(f"批次处理尝试 {attempt + 1} 失败: {str(e)}")
            if attempt < max_retries - 1:
                logger.info(f"{retry_delay}秒后进行下一次重试...")
                time.sleep(retry_delay)
                retry_delay *= 2  # 指数退避
            else:
                raise Exception(f"批次处理失败，已达最大重试次数({max_retries}次): {str(e)}")


def main():
    # 输入输出路径配置
    input_path = "./inputs/data/math23k.jsonl"
    output_dir = "./inputs/data/math23k_trans"
    os.makedirs(output_dir, exist_ok=True)  # 确保输出目录存在

    # 读取数据并筛选前8000条
    logger.info("开始读取数据...")
    df = read_jsonl_to_df(input_path)
    df_filter = df[:8000].copy()
    logger.info(f"数据读取完成，共处理 {len(df_filter)} 条记录")

    # 批次配置
    batch_size = 100
    total_batches = (len(df_filter) + batch_size - 1) // batch_size  # 向上取整计算总批次

    # 分批处理
    for batch_idx in tqdm(range(total_batches), desc="总处理进度"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(df_filter))
        logger.info(f"处理批次 {batch_idx + 1}/{total_batches} (索引范围: {start_idx}-{end_idx-1})")

        # 提取当前批次数据
        df_batch = df_filter.iloc[start_idx:end_idx].copy()

        # 带重试机制处理批次
        try:
            df_processed_batch = process_batch_with_retry(df_batch)
        except Exception as e:
            logger.error(f"批次 {batch_idx + 1} 处理失败: {str(e)}")
            # 记录失败批次索引，方便后续处理
            with open(os.path.join(output_dir, "failed_batches.log"), "a") as f:
                f.write(f"Batch {batch_idx + 1}: start={start_idx}, end={end_idx-1}, error={str(e)}\n")
            continue

        # 保存处理结果
        output_path = os.path.join(output_dir, f'trans_math23k_part_{batch_idx}.jsonl')
        df_output = df_processed_batch[['title', 'description', 'solution', 'tran_description']]
        df_output.to_json(
            output_path,
            orient='records',
            lines=True,
            force_ascii=False  # 保留中文
        )
        logger.info(f"批次 {batch_idx + 1} 已保存至 {output_path}")

    logger.info("所有批次处理完毕")


if __name__ == '__main__':
    main()