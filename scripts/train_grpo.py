import torch
import re
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, PeftModel
from trl import GRPOTrainer, GRPOConfig

from config import BASE_MODEL, SFT_ADAPTER, GRPO_OUT, GRPO_ADAPTER

# ========== 1. Load SFT model ==========
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
)
model = PeftModel.from_pretrained(model, SFT_ADAPTER)
model = model.merge_and_unload()

# ========== 2. Create math dataset ==========
problems = []
for a in range(1, 51):
    for b in range(1, 51):
        problems.append({
            "prompt": f"<|im_start|>user\nCalculate: {a} + {b} = ?<|im_end|>\n<|im_start|>assistant\n",
            "answer": str(a + b)
        })
        problems.append({
            "prompt": f"<|im_start|>user\nCalculate: {a} * {b} = ?<|im_end|>\n<|im_start|>assistant\n",
            "answer": str(a * b)
        })

ds = Dataset.from_list(problems[:2000])
print(f"Math dataset: {len(ds)} problems")

# ========== 3. Reward function ==========
def reward_fn(completions, **kwargs):
    """Check if the correct answer appears in the response."""
    rewards = []
    answers = kwargs.get("answer", [])
    for completion, answer in zip(completions, answers):
        text = completion[0]["content"] if isinstance(completion, list) else completion
        # Check if correct number appears in response
        if answer in text:
            # Bonus for concise answers
            if len(text) < 50:
                rewards.append(2.0)
            else:
                rewards.append(1.0)
        else:
            rewards.append(-1.0)
    return rewards

# ========== 4. LoRA config ==========
lora_config = LoraConfig(
    r=32, lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05, task_type="CAUSAL_LM"
)

# ========== 5. GRPO Config ==========
grpo_config = GRPOConfig(
    output_dir=GRPO_OUT,
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=5e-6,
    bf16=True,
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,
    report_to="none",
    num_generations=2,
    max_completion_length=100,
    max_prompt_length=128,
)

# ========== 6. Train ==========
trainer = GRPOTrainer(
    model=model,
    args=grpo_config,
    train_dataset=ds,
    reward_funcs=reward_fn,
    peft_config=lora_config,
)

print("Starting GRPO training...")
trainer.train()
trainer.save_model(GRPO_ADAPTER)
print("GRPO training done!")
