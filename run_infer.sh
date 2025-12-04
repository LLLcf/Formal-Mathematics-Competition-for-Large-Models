#!/bin/bash

# ================= 默认配置区域 (在此修改默认值) =================
DEFAULT_INPUT="./inputs/final_litex.jsonl"
DEFAULT_MODEL="/root/lanyun-tmp/models/Qwen3-8B"
DEFAULT_LORA="./results-test/checkpoint-20000"
DEFAULT_OUTPUT="./outputs/final_day/submit"
DEFAULT_GPU="0"

# ================= 参数解析 =================

# 逻辑：${1:-$DEFAULT_INPUT} 表示 "如果$1存在且非空，取$1；否则取$DEFAULT_INPUT"
INPUT_PATH=${1:-$DEFAULT_INPUT}
BASE_MODEL=${2:-$DEFAULT_MODEL}
LORA_PATH=${3:-$DEFAULT_LORA}
OUTPUT_NAME=${4:-$DEFAULT_OUTPUT}
GPU_ID=${5:-$DEFAULT_GPU} 

# 打印帮助信息（可选，如果用户输入 -h 或 --help）
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    echo "使用方法: $0 [输入路径] [模型路径] [LoRA路径] [输出前缀] [显卡ID]"
    echo "如果不提供参数，将使用脚本内定义的默认值。"
    exit 0
fi

# ================= 运行确认 =================
echo "=== 启动任务 ==="
echo "输入数据: $INPUT_PATH"
echo "基础模型: $BASE_MODEL"
echo "LoRA权重: $LORA_PATH"
echo "输出路径: $OUTPUT_NAME"
echo "使用显卡: $GPU_ID"
echo "=================="

# ================= 执行命令 =================
python -u infer.py \
    --input_path "$INPUT_PATH" \
    --base_model_path "$BASE_MODEL" \
    --lora_path "$LORA_PATH" \
    --output_name "$OUTPUT_NAME" \