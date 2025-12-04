import json
import os
import pylitex  # 确保你的环境中已安装此库

def judge_litex_correctness(message):
    """
    judge the code
    调用 pylitex 运行代码并返回成功状态和消息
    """
    try:
        result = pylitex.run(message)
        return result["success"], result["message"]
    except Exception as e:
        # 如果 pylitex 运行出错，默认视为验证失败
        return False, str(e)

def process_and_merge_recursive(root_folder, output_filename):
    """
    递归遍历文件夹及其子文件夹，处理jsonl字段。
    仅当 formal_code 通过 judge_litex_correctness 验证时才合并到新文件中。
    """
    output_path = os.path.join(root_folder, output_filename)
    abs_output_path = os.path.abspath(output_path)
    
    total_processed = 0      # 总处理行数（读取数）
    valid_saved_count = 0    # 验证通过并保存的数
    invalid_skip_count = 0   # 验证失败跳过的数
    files_count = 0

    print(f"开始递归扫描目录: {root_folder}")
    print("-" * 30)
    
    with open(output_path, 'w', encoding='utf-8') as f_out:
        
        for current_dir, dirs, files in os.walk(root_folder):
            
            # --- 优化：直接禁止进入 .ipynb_checkpoints 目录 ---
            if '.ipynb_checkpoints' in dirs:
                dirs.remove('.ipynb_checkpoints')
            
            for filename in files:
                # 1. 只处理 .jsonl 文件
                if not filename.endswith(".jsonl"):
                    continue

                file_path = os.path.join(current_dir, filename)
                
                # 2. 再次检查：确保路径中不包含 .ipynb_checkpoints
                if '.ipynb_checkpoints' in file_path:
                    continue

                # 3. 跳过输出文件本身
                if os.path.abspath(file_path) == abs_output_path:
                    continue
                
                # 4. 打印进度
                files_count += 1
                rel_path = os.path.relpath(file_path, root_folder)
                print(f"正在处理: {rel_path}")
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f_in:
                        for line_num, line in enumerate(f_in, 1):
                            line = line.strip()
                            if not line:
                                continue
                            
                            total_processed += 1
                            
                            try:
                                data = json.loads(line)
                                
                                # --- 字段提取 ---
                                title = data.get('id', '')
                                raw_problem = data.get('nl_problem', '')
                                solution = data.get('formal_code', '')
                                
                                # --- 核心修改：增加正确性校验 ---
                                # 如果 solution 为空，通常认为无效，或者你可以根据需求决定空代码是否去跑验证
                                if not solution:
                                    invalid_skip_count += 1
                                    continue

                                is_success, msg = judge_litex_correctness(solution)
                                
                                if not is_success:
                                    # 验证失败，跳过保存
                                    invalid_skip_count += 1
                                    # 可选：打印失败信息以便调试
                                    # print(f"  [Skip] Line {line_num}: Code validation failed.")
                                    continue
                                
                                # --- 验证通过，继续构建数据 ---
                                description = f"Question: {raw_problem} Answer: #### "
                                
                                new_entry = {
                                    "title": title,
                                    "description": description,
                                    "solution": solution
                                }
                                
                                f_out.write(json.dumps(new_entry, ensure_ascii=False) + '\n')
                                valid_saved_count += 1
                                
                            except json.JSONDecodeError:
                                print(f"  [警告] {rel_path} 第 {line_num} 行存在 JSON 格式错误，已跳过。")
                            except Exception as inner_e:
                                print(f"  [警告] 处理 {rel_path} 第 {line_num} 行数据时出错: {inner_e}")
                                
                except Exception as e:
                    print(f"  [错误] 无法读取文件 {rel_path}: {e}")

    print("-" * 30)
    print(f"处理完成！")
    print(f"共扫描有效文件数: {files_count}")
    print(f"共读取数据条数: {total_processed}")
    print(f"验证通过并保存: {valid_saved_count}")
    print(f"验证失败被丢弃: {invalid_skip_count}")
    print(f"结果已保存至: {output_path}")

if __name__ == "__main__":
    # --- 配置区域 ---
    target_directory = './outputs'  # 你的目标根目录
    result_filename = 'merged_all_days_validated.jsonl' # 修改了文件名以区分
    
    process_and_merge_recursive(target_directory, result_filename)