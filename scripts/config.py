"""Central path configuration for all training and evaluation scripts.

Defaults assume a `./workspace` directory at the project root containing
`models/`, `data/`, and `outputs/` subdirectories. Any path can be overridden
via environment variable without touching code.

Example:
    export ALIGN_ROOT=/path/to/my/workspace
    python scripts/train_sft.py

    # Or override a single path
    export ALIGN_BASE_MODEL=/shared/models/Qwen2.5-1.5B
    python scripts/train_dpo.py
"""
import os

ROOT = os.environ.get("ALIGN_ROOT", "./workspace")

BASE_MODEL = os.environ.get("ALIGN_BASE_MODEL", f"{ROOT}/models/Qwen/Qwen2.5-1.5B")
INSTRUCT_MODEL = os.environ.get("ALIGN_INSTRUCT_MODEL", f"{ROOT}/models/Qwen/Qwen2.5-1.5B-Instruct")

DATA_CACHE = os.environ.get("ALIGN_DATA_CACHE", f"{ROOT}/data")
SFT_DATA = os.environ.get(
    "ALIGN_SFT_DATA",
    f"{DATA_CACHE}/AI-ModelScope___alpaca-gpt4-data-zh/default-227cda14dfde522c/0.0.0/master/alpaca-gpt4-data-zh-train.arrow",
)
DPO_DATA = os.environ.get("ALIGN_DPO_DATA", f"{DATA_CACHE}/ultrafeedback_cleaned")

OUTPUTS = os.environ.get("ALIGN_OUTPUTS", f"{ROOT}/outputs")
SFT_OUT = f"{OUTPUTS}/sft"
SFT_TEST_OUT = f"{OUTPUTS}/sft_test"
DPO_OUT = f"{OUTPUTS}/dpo"
DPO_BETA03_OUT = f"{OUTPUTS}/dpo_beta03"
DPO_FILTERED_OUT = f"{OUTPUTS}/dpo_filtered"
KTO_OUT = f"{OUTPUTS}/kto"
GRPO_OUT = f"{OUTPUTS}/grpo"

SFT_ADAPTER = f"{SFT_OUT}/final"
DPO_ADAPTER = f"{DPO_OUT}/final"
DPO_BETA03_ADAPTER = f"{DPO_BETA03_OUT}/final"
DPO_FILTERED_ADAPTER = f"{DPO_FILTERED_OUT}/final"
KTO_ADAPTER = f"{KTO_OUT}/final"
GRPO_ADAPTER = f"{GRPO_OUT}/final"
