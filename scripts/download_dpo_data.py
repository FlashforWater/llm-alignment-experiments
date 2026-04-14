from modelscope import MsDataset

from config import DATA_CACHE

print("Downloading DPO preference dataset from ModelScope...")
ds = MsDataset.load("AI-ModelScope/ultrafeedback-binarized-preferences-cleaned", cache_dir=DATA_CACHE)
print(f"Done! Total samples: {len(ds['train'])}")
print(f"Columns: {ds['train'].column_names}")
