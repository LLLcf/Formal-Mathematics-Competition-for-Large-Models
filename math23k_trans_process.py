import os
import re
import pandas as pd


def merge_jsonl_files(input_dir, output_file_path):
    """
    合并指定目录下所有JSONL文件到单个输出文件
    
    Args:
        input_dir: 包含JSONL文件的目录路径
        output_file_path: 合并后的输出文件路径
    """
    # 收集所有JSONL文件路径
    jsonl_files = []
    for filename in os.listdir(input_dir):
        if filename.endswith('.jsonl'):
            full_path = os.path.join(input_dir, filename)
            jsonl_files.append(full_path)
    
    if not jsonl_files:
        print(f"警告：在文件夹 {input_dir} 中未找到任何JSONL文件")
        return
    
    print(f"找到 {len(jsonl_files)} 个JSONL文件，开始合并...")
    line_count = 0  # 记录合并总行数
    
    # 合并文件并统计有效行数
    with open(output_file_path, 'w', encoding='utf-8') as output_f:
        for file_path in jsonl_files:
            print(f"正在处理文件：{os.path.basename(file_path)}")
            with open(file_path, 'r', encoding='utf-8') as input_f:
                for line in input_f:
                    stripped_line = line.strip()
                    if stripped_line:  # 跳过空行
                        output_f.write(stripped_line + '\n')
                        line_count += 1
    
    print(f"合并完成！所有JSONL文件已合并为：{output_file_path}")
    print(f"合并后文件总行数（每个JSON对象一行）：{line_count}\n")


def contains_chinese(text):
    """检查文本中是否包含中文字符"""
    if pd.isna(text) or text is None:
        return False
    # 匹配Unicode中的中文字符范围
    chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
    return bool(chinese_pattern.search(str(text)))


def process_jsonl_folder(input_dir, merge_output_path, filter_output_path):
    """
    处理单个文件夹的完整流程：合并JSONL文件 -> 过滤含中文的行 -> 保存结果
    
    Args:
        input_dir: 原始JSONL文件所在目录
        merge_output_path: 合并后的临时文件路径
        filter_output_path: 过滤后的最终输出路径
    """
    # 1. 合并JSONL文件
    merge_jsonl_files(input_dir, merge_output_path)
    
    # 2. 读取合并后的文件
    try:
        # 假设JSONL格式为每行一个JSON对象，使用pandas读取
        df = pd.read_json(merge_output_path, lines=True, encoding='utf-8')
    except Exception as e:
        print(f"读取合并文件失败：{e}")
        return
    
    # 3. 过滤包含中文的行并处理
    print(f"处理文件: {os.path.basename(merge_output_path)}")
    print(f"过滤前数据行数: {len(df)}")
    
    # 检查是否存在目标列
    if 'tran_description' not in df.columns:
        print("错误：数据中不包含'tran_description'列，无法过滤")
        return
    
    # 统计并过滤含中文的行
    chinese_rows = df['tran_description'].apply(contains_chinese).sum()
    print(f"包含中文的行数: {chinese_rows}")
    filtered_df = df[~df['tran_description'].apply(contains_chinese)]
    print(f"过滤后数据行数: {len(filtered_df)}\n")
    
    # 4. 处理列名并保存
    try:
        # 选择需要的列并改名
        result_df = filtered_df[['title', 'solution', 'tran_description']].rename(
            columns={'tran_description': 'description'}
        )
        # 保存为JSONL格式
        result_df.to_json(
            filter_output_path,
            orient='records',
            lines=True,
            force_ascii=False,  # 保留非ASCII字符
        )
        print(f"过滤结果已保存至：{filter_output_path}\n")
    except KeyError as e:
        print(f"列名错误：{e}，无法保存结果")
    except Exception as e:
        print(f"保存文件失败：{e}")


if __name__ == '__main__':

    local_save_dir = "./inputs/data" 
    
    processing_configs = [
        {
            "input_dir": os.path.join(local_save_dir, 'head8k'),
            "merge_output": os.path.join(local_save_dir, "merged_head8k.jsonl"),
            "filter_output": "./inputs/data/merged_head8k_filter.jsonl"
        },
        {
            "input_dir": os.path.join(local_save_dir, 'tail8k'),
            "merge_output": os.path.join(local_save_dir, "merged_tail8k.jsonl"),
            "filter_output": "./inputs/data/merged_tail8k_filter.jsonl"
        },
        {
            "input_dir": os.path.join(local_save_dir, 'mid8k'),
            "merge_output": os.path.join(local_save_dir, "merged_mid8k.jsonl"),
            "filter_output": "./inputs/data/merged_mid8k_filter.jsonl"
        }
    ]
    
    # 依次处理每个配置
    for config in processing_configs:
        process_jsonl_folder(
            input_dir=config["input_dir"],
            merge_output_path=config["merge_output"],
            filter_output_path=config["filter_output"]
        )