import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from config import BASE_MODEL, SFT_ADAPTER, GRPO_ADAPTER

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

def generate(model, tokenizer, prompt):
    messages = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(messages, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=100, temperature=0.7, do_sample=True,
            eos_token_id=[151645, 151643], pad_token_id=151643,
        )
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

math_prompts = [
    "Calculate: 7 + 13 = ?",
    "Calculate: 25 * 4 = ?",
    "Calculate: 49 + 38 = ?",
    "Calculate: 12 * 11 = ?",
    "Calculate: 6 + 9 = ?",
]

# SFT model on math
print("=" * 60)
print("SFT MODEL (math)")
print("=" * 60)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
)
model = PeftModel.from_pretrained(model, SFT_ADAPTER)
for p in math_prompts:
    print(f"\nPrompt: {p}")
    print(f"Response: {generate(model, tokenizer, p)}")
    print("-" * 40)
del model
torch.cuda.empty_cache()

# GRPO model on math
print("\n" + "=" * 60)
print("GRPO MODEL (math)")
print("=" * 60)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
)
model = PeftModel.from_pretrained(model, SFT_ADAPTER)
model = model.merge_and_unload()
model = PeftModel.from_pretrained(model, GRPO_ADAPTER)
for p in math_prompts:
    print(f"\nPrompt: {p}")
    print(f"Response: {generate(model, tokenizer, p)}")
    print("-" * 40)
