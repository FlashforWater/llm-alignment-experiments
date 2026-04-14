import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from config import BASE_MODEL, SFT_ADAPTER, DPO_ADAPTER, KTO_ADAPTER

def generate(model, tokenizer, prompt):
    messages = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(messages, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True,
            eos_token_id=[151645, 151643],
            pad_token_id=151643,
            repetition_penalty=1.3,
        )
    full = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    if "assistant" in full.lower():
        full = full[:full.lower().index("assistant")]
    return full.strip()

prompts = [
    "什么是强化学习？",
    "1+1等于几？",
    "帮我写一封请假邮件",
    "What is the difference between machine learning and deep learning?",
]

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

print("=" * 60)
print("DPO (Exp-02, beta=0.1)")
print("=" * 60)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
)
model = PeftModel.from_pretrained(model, SFT_ADAPTER)
model = model.merge_and_unload()
model = PeftModel.from_pretrained(model, DPO_ADAPTER)
for p in prompts:
    print(f"\nPrompt: {p}")
    print(f"Response: {generate(model, tokenizer, p)}")
    print("-" * 40)
del model
torch.cuda.empty_cache()

print("\n" + "=" * 60)
print("KTO (Exp-05)")
print("=" * 60)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
)
model = PeftModel.from_pretrained(model, SFT_ADAPTER)
model = model.merge_and_unload()
model = PeftModel.from_pretrained(model, KTO_ADAPTER)
for p in prompts:
    print(f"\nPrompt: {p}")
    print(f"Response: {generate(model, tokenizer, p)}")
    print("-" * 40)
