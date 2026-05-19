import os
from huggingface_hub import snapshot_download

# 1. Force standard download mechanisms (disable fast-transfer libraries that segfault)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0" 

# 2. Define the model
model_id = "meta-llama/Llama-3.3-70B-Instruct"

print(f"Starting resilient download of {model_id}...")

# 3. Use snapshot_download with explicit settings
try:
    path = snapshot_download(
        repo_id=model_id,
        max_workers=4, # Reduce parallel connections to prevent network drops
        resume_download=True # Vital: pick up where it left off
    )
    print(f"Success! Model downloaded to: {path}")
except Exception as e:
    print(f"Download failed: {e}")