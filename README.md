vllm环境配置
```python
pip install -r requirements.txt
```

运行以下代码
```python
./run_infer.sh ./inputs/test.jsonl /path/to/model /path/to/lora ./outputs/res
```

模型权重
```python
https://www.modelscope.cn/models/lcffff0705/llm2litex/
```