"""Download IFM/K2-Horizon-3.7B weights to E:\hf-models (C: is tight on space)."""
import sys
from huggingface_hub import snapshot_download

TARGET = r"E:\hf-models\K2-Horizon-3.7B"

path = snapshot_download(
    repo_id="IFM/K2-Horizon-3.7B",
    local_dir=TARGET,
    # skip the marketing images; keep weights, code, tokenizer, chat template
    ignore_patterns=["assets/*", "*.png"],
    max_workers=4,
)
print("DONE ->", path, file=sys.stderr)
