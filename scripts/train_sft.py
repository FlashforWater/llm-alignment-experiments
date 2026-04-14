import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer

from config import BASE_MODEL, SFT_DATA, SFT_OUT, SFT_ADAPTER

# ========== 1. Load model and tokenizer ==========
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

# ========== 2. Load and format dataset ==========
ds = Dataset.from_file(SFT_DATA)

def format_chatml(example):
    instruction = example["instruction"]
    inp = example.get("input", "")
    output = example["output"]
    if inp:
        user_msg = f"{instruction}\n{inp}"
    else:
        user_msg = instruction
    text = (
        f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        f"<|im_start|>assistant\n{output}<|im_end|>"
    )
    return {"text": text}

ds = ds.map(format_chatml)

# ========== 3. LoRA config ==========
lora_config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)

# ========== 4. Training arguments ==========
training_args = TrainingArguments(
    output_dir=SFT_OUT,
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    bf16=True,
    logging_steps=10,
    save_steps=500,
    save_total_limit=3,
    report_to="none",
)

# ========== 5. Train ==========
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=ds,
    peft_config=lora_config,
    max_seq_length=512,
)

print("Starting SFT training...")
trainer.train()
trainer.save_model(SFT_ADAPTER)
print("SFT training done!")
