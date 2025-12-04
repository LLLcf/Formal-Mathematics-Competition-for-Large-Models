import pandas as pd


root_path = './inputs/data/hendrycks_math/'

data = pd.read_parquet('/root/lanyun-tmp/FormaLLM-Challenge-2025/llm2litex/inputs/data/hendrycks_math/precalculus/train-00000-of-00001.parquet')

print(data.head())