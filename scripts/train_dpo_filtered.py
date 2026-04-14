import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, PeftModel
from trl import DPOTrainer, DPOConfig

from config import BASE_MODEL, SFT_ADAPTER, DPO_DATA, DPO_FILTERED_OUT, DPO_FILTERED_ADAPTER

# ========== 1. Load SFT model ==========
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
)
model = PeftModel.from_pretrained(model, SFT_ADAPTER)
model = model.merge_and_unload()

# ========== 2. Load and filter dataset ==========
ds = load_from_disk(DPO_DATA)

# Filter: score difference >= 2.0
ds = ds.filter(lambda x: abs(x["chosen-rating"] - x["rejected-rating"]) >= 2.0)
print(f"Filtered samples: {len(ds)} (from 60917)")

# Take up to 5000
if len(ds) > 5000:
    ds = ds.select(range(5000))

def format_dpo(example):
    prompt = example["prompt"]
    chosen_text = ""
    rejected_text = ""
    for msg in example["chosen"]:
        if msg["role"] == "assistant":
            chosen_text = msg["content"]
    for msg in example["rejected"]:
        if msg["role"] == "assistant":
            rejected_text = msg["content"]
    return {
        "prompt": f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
        "chosen": chosen_text + "<|im_end|>",
        "rejected": rejected_text + "<|im_end|>",
    }

ds = ds.map(format_dpo)
ds = ds.filter(lambda x: len(x["chosen"]) > 10 and len(x["rejected"]) > 10)
print(f"Final training samples: {len(ds)}")

# ========== 3. LoRA config ==========
lora_config = LoraConfig(
    r=32, lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05, task_type="CAUSAL_LM"
)

# ========== 4. DPO Config (beta=0.1, same as Exp-02) ==========
dpo_config = DPOConfig(
    output_dir=DPO_FILTERED_OUT,
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=5e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    bf16=True,
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,
    report_to="none",
    remove_unused_columns=False,
    beta=0.1,
    max_length=512,
    max_prompt_length=256,
)

# ========== 5. Train ==========
trainer = DPOTrainer(
    model=model, args=dpo_config, train_dataset=ds,
    tokenizer=tokenizer, peft_config=lora_config,
)

print("Starting DPO training (filtered high-quality pairs)...")
trainer.train()
trainer.save_model(DPO_FILTERED_ADAPTER)
print("DPO filtered training done!")
