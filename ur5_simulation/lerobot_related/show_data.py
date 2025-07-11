import pandas as pd
import os

# read data
loaded_df = pd.read_parquet(os.environ["HOME"] + '/.cache/huggingface/lerobot/lerobot/pusht/data/chunk-000/episode_000000.parquet')
#loaded_df = pd.read_parquet(os.environ["HOME"] + '/training_data/lerobot/my_pusht/data/chunk-000/episode_000000.parquet')
#loaded_df = pd.read_parquet(os.environ["HOME"] + '/ur5_simulation/src/data_collection/scripts/my_pusht/data/chunk_000/episode_000014.parquet')

print(loaded_df)