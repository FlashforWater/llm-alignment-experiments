import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from config import BASE_MODEL, SFT_ADAPTER

def generate(model, tokenizer, prompt, stop_ids):
    messages = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(messages, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
            eos_token_id=stop_ids,
            pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response

prompts = [
    "什么是强化学习？",
    "请用简单的语言解释什么是人工智能",
    "写一个Python函数，计算斐波那契数列的第n项",
    "1+1等于几？",
    "帮我写一封请假邮件",
]

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
stop_ids = [151645, 151643]

print("=" * 60)
print("SFT MODEL (with eos fix v3)")
print("=" * 60)
sft_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
)
sft_model = PeftModel.from_pretrained(sft_model, SFT_ADAPTER)
for p in prompts:
    print(f"\nPrompt: {p}")
    print(f"Response: {generate(sft_model, tokenizer, p, stop_ids)}")
    print("-" * 40)
