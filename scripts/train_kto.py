import torch
from datasets import load_from_disk, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, PeftModel
from trl import KTOTrainer, KTOConfig

from config import BASE_MODEL, SFT_ADAPTER, DPO_DATA, KTO_OUT, KTO_ADAPTER

# ========== 1. Load SFT model ==========
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
)
model = PeftModel.from_pretrained(model, SFT_ADAPTER)
model = model.merge_and_unload()

# ========== 2. Convert DPO data to KTO format ==========
ds = load_from_disk(DPO_DATA)
ds = ds.select(range(5000))

kto_data = {"prompt": [], "completion": [], "label": []}

for example in ds:
    prompt = f"<|im_start|>user\n{example['prompt']}<|im_end|>\n<|im_start|>assistant\n"
    
    chosen_text = ""
    rejected_text = ""
    for msg in example["chosen"]:
        if msg["role"] == "assistant":
            chosen_text = msg["content"]
    for msg in example["rejected"]:
        if msg["role"] == "assistant":
            rejected_text = msg["content"]
    
    if len(chosen_text) > 10 and len(rejected_text) > 10:
        # Good example
        kto_data["prompt"].append(prompt)
        kto_data["completion"].append(chosen_text + "<|im_end|>")
        kto_data["label"].append(True)
        # Bad example
        kto_data["prompt"].append(prompt)
        kto_data["completion"].append(rejected_text + "<|im_end|>")
        kto_data["label"].append(False)

kto_ds = Dataset.from_dict(kto_data)
print(f"KTO dataset: {len(kto_ds)} samples ({sum(kto_data['label'])} positive, {len(kto_data['label']) - sum(kto_data['label'])} negative)")

# ========== 3. LoRA config ==========
lora_config = LoraConfig(
    r=32, lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05, task_type="CAUSAL_LM"
)

# ========== 4. KTO Config ==========
kto_config = KTOConfig(
    output_dir=KTO_OUT,
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
    max_length=512,
    max_prompt_length=256,
)

# ========== 5. Train ==========
trainer = KTOTrainer(
    model=model,
    args=kto_config,
    train_dataset=kto_ds,
    processing_class=tokenizer,
    peft_config=lora_config,
)

print("Starting KTO training...")
trainer.train()
trainer.save_model(KTO_ADAPTER)
print("KTO training done!")
