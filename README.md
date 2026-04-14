# Alignment Pipeline: A Comparative Study of SFT, DPO, KTO, and GRPO

An end-to-end, from-scratch implementation and comparison of four alignment algorithms on Qwen2.5-1.5B, run on a single RTX 5090. Six experiments, documented successes and failures, one consistent conclusion: **data quality and reward design matter more than algorithm tuning.**

## Results at a glance

| Experiment | Algorithm | Key Change | Result |
|-----------|-----------|-----------|--------|
| Exp-01 | SFT | baseline | Learned dialogue format, loss 2.5→1.72 |
| Exp-02 | DPO β=0.1 | preference pairs | Better quality, but verbose |
| Exp-03 | DPO β=0.3 | higher beta | Hallucination ↑, worse overall |
| Exp-04 | DPO filtered | high-gap pairs only | Best structure, no hallucination |
| Exp-05 | KTO | unpaired binary labels | Matches DPO quality |
| Exp-06 | GRPO | math + reward function | Correct and concise, solved verbosity |

## Sample output: GRPO solved what DPO and KTO couldn't

On the prompt `Calculate: 7 + 13 = ?`:

**SFT model** — correct answer buried in fake multi-turn rambling:
```
The sum of 7 and 13 is 20.
assistant: Great job! Is there anything else I can help you with?
assistant: No, thank you. You've been very helpful...
assistant: Not at all! If you have any specific task or question...
```

**GRPO model** — correct and concise, exactly what we rewarded:
```
The answer is 20.
```

This is the verbosity problem DPO and KTO couldn't fix (they actually *amplified* it), solved by giving GRPO an explicit `correct + concise = 2.0` reward. See `results/compare_grpo.txt` for the full comparison.

## Key findings

1. **Data quality > algorithm tuning.** Filtering for high-quality preference pairs (Exp-04) improved results more than tuning beta (Exp-03).
2. **Higher beta is not "stronger regularization done right".** Raising β from 0.1 to 0.3 caused hallucination on simple questions instead of reducing verbosity.
3. **KTO matches DPO with a weaker signal.** Despite using only binary labels (good/bad) instead of paired comparisons, KTO produced comparable results — confirming the KTO paper's claim that pairing information is not strictly necessary.
4. **GRPO + explicit reward solves verbosity.** When a clear reward function can be defined, GRPO eliminates the verbosity problem that plagued DPO and KTO.
5. **Training ≠ inference.** A well-trained model can still produce poor output if generation parameters (`eos_token_id`, `repetition_penalty`) are not set correctly. This cost me an evening of debugging in Exp-01.

## Pipeline

```
Base Model (Qwen2.5-1.5B, text completion only)
    ↓  SFT: supervised fine-tuning with instruction data
SFT Model (can follow instructions, but no quality preference)
    ↓  Choose alignment algorithm:
    ├── DPO:  learns from paired preferences (chosen vs rejected)
    ├── KTO:  learns from binary feedback (good / bad)
    └── GRPO: learns from self-generated attempts + reward function
Aligned Model (produces higher-quality, more helpful responses)
```

## Hardware & software

**Hardware used for these experiments**
- GPU: NVIDIA RTX 5090 (32GB) × 1
- CPU: 25 vCPU Intel Xeon Platinum 8470Q
- Memory: 90GB
- Platform: AutoDL

**Dependencies** — pinned in `requirements.txt`:
```bash
pip install -r requirements.txt
```

## Quickstart

All scripts read paths from `scripts/config.py`, which reads environment variables with sensible defaults. The default layout is:

```
./workspace/
├── models/Qwen/Qwen2.5-1.5B/          # base model
├── data/                               # downloaded datasets
└── outputs/                            # training checkpoints (gitignored)
    ├── sft/final/
    ├── dpo/final/
    ├── dpo_beta03/final/
    ├── dpo_filtered/final/
    ├── kto/final/
    └── grpo/final/
```

**Option A — use defaults:** place files under `./workspace/` and run:
```bash
python scripts/train_sft.py
python scripts/train_dpo.py
python scripts/evaluate_final.py
```

**Option B — override via environment variables:**
```bash
export ALIGN_ROOT=/path/to/my/workspace           # moves everything
export ALIGN_BASE_MODEL=/shared/cache/Qwen2.5-1.5B  # override a single path
python scripts/train_sft.py
```

See `scripts/config.py` for the full list of overridable variables.

### Model and data download

```python
from modelscope import snapshot_download, MsDataset

# Base model → ./workspace/models/
snapshot_download('Qwen/Qwen2.5-1.5B', cache_dir='./workspace/models')

# SFT data: 48k Chinese instructions → ./workspace/data/
MsDataset.load('AI-ModelScope/alpaca-gpt4-data-zh', cache_dir='./workspace/data')

# DPO/KTO data: 60k English preference pairs → ./workspace/data/
MsDataset.load('AI-ModelScope/ultrafeedback-binarized-preferences-cleaned',
               cache_dir='./workspace/data')
```

The DPO/KTO download is also wrapped in `scripts/download_dpo_data.py` for convenience.

## A note on Chinese test prompts

Evaluation and comparison scripts contain Chinese prompts like `"什么是强化学习？"` ("What is reinforcement learning?"). This is intentional: the SFT training data is `alpaca-gpt4-data-zh` (Chinese instructions), so testing the model in Chinese is the methodologically correct way to evaluate what it actually learned. English prompts are also included where relevant (e.g., math problems in GRPO, general-knowledge questions in `evaluate_v4.py`).

## Project structure

```
alignment-pipeline/
├── README.md
├── LICENSE
├── requirements.txt
├── experiment_log.md
├── .gitignore
├── scripts/
│   ├── config.py                  # central path config (env-var overridable)
│   ├── download_dpo_data.py       # fetch UltraFeedback from ModelScope
│   ├── train_sft.py               # Exp-01: SFT baseline
│   ├── train_dpo.py               # Exp-02: DPO β=0.1
│   ├── train_dpo_beta03.py        # Exp-03: DPO β=0.3
│   ├── train_dpo_filtered.py      # Exp-04: DPO with filtered data
│   ├── train_kto.py               # Exp-05: KTO
│   ├── train_grpo.py              # Exp-06: GRPO on math
│   ├── test_sft.py                # 100-sample SFT smoke test
│   ├── evaluate.py                # base vs SFT, initial eval
│   ├── evaluate_v2.py             # + eos fix
│   ├── evaluate_v3.py             # + list of stop ids
│   ├── evaluate_v4.py             # SFT vs official Instruct baseline
│   ├── evaluate_final.py          # SFT vs DPO qualitative eval
│   ├── compare_beta.py            # β=0.1 vs β=0.3
│   ├── compare_filtered.py        # original vs filtered DPO
│   ├── compare_kto.py             # DPO vs KTO
│   └── eval_grpo.py               # SFT vs GRPO on math
└── results/
    ├── eval_result.txt            # initial SFT eval (reveals repetition bug)
    ├── eval_v2.txt / v3.txt / v4.txt  # iterative eval fixes
    ├── eval_final.txt             # SFT vs DPO
    ├── compare_beta.txt           # β=0.1 vs β=0.3
    ├── compare_filtered.txt       # original vs filtered DPO
    ├── compare_kto.txt            # DPO vs KTO
    └── compare_grpo.txt           # SFT vs GRPO on math
```

## Experiment details

### Exp-01: SFT baseline
Supervised fine-tuning on 48,818 Chinese instruction-response pairs using LoRA (r=64, α=128). Trained for 1 epoch with lr=2e-4 and cosine scheduler. Loss decreased from 2.5 to 1.72 in 23 minutes.

The model learned to follow the ChatML dialogue format, but initial evaluation revealed two issues: infinite repetition and multilingual garbled text. These were resolved by setting the correct `eos_token_id` and adding `repetition_penalty` during generation — a good reminder that *training ≠ inference*.

### Exp-02: DPO (β=0.1)
Direct Preference Optimization using 5,000 English preference pairs from UltraFeedback. Applied LoRA (r=32, α=64) on the merged SFT checkpoint with lr=5e-5 and β=0.1.

Results showed improved answer structure and professionalism compared to SFT, but introduced a verbosity problem — the model learned that longer answers received higher preference ratings.

### Exp-03: DPO (β=0.3) — beta ablation
Same setup as Exp-02 but with β=0.3 to test whether a more conservative update would reduce verbosity.

Unexpectedly, β=0.3 produced *worse* results: hallucination on simple questions (e.g., generating Java `parseInt` code for "1+1=?"). This demonstrated that higher beta does not simply mean better control.

### Exp-04: DPO filtered — data quality ablation
Same algorithm and parameters as Exp-02, but filtered the preference data to only include pairs where `chosen_rating - rejected_rating >= 2.0`. Out of 60,917 total samples, 16,594 passed the filter (27%). Used 5,000 of these for training.

Result: better answer structure than Exp-02 and no hallucination. This confirmed that **data quality matters more than parameter tuning** — the same algorithm with cleaner data outperformed the beta-tuned version.

### Exp-05: KTO
Kahneman-Tversky Optimization using the same preference data converted to unpaired format: each DPO pair was split into one positive (label=True) and one negative (label=False) example, yielding 9,924 training samples.

KTO roughly matched DPO performance despite using a strictly weaker training signal (binary labels vs paired comparisons). This confirms the KTO paper's finding that pairing information is not strictly necessary for effective alignment.

### Exp-06: GRPO on math
Group Relative Policy Optimization on 2,000 auto-generated math problems (addition and multiplication). The model generated 2 responses per prompt, scored by a rule-based reward function: `correct + concise = 2.0`, `correct + verbose = 1.0`, `incorrect = -1.0`.

GRPO successfully solved the verbosity problem: responses were correct and concise ("The answer is 20."), compared to SFT's tendency to generate endless fake dialogue. This demonstrated that when a clear reward function can be defined, GRPO is highly effective.

## Algorithm comparison

| Aspect | SFT | DPO | KTO | GRPO |
|--------|-----|-----|-----|------|
| Data needed | instruction-response pairs | paired preferences | binary labels | reward function only |
| Data collection difficulty | Medium | Hard (need pairs) | Easy (thumbs up/down) | None (self-generated) |
| Training signal | "This is the correct answer" | "A is better than B" | "This is good/bad" | "Score: 2.0, -1.0, ..." |
| Verbosity control | Poor | Poor | Poor | Good (if rewarded) |
| Training speed | Fast | Medium | Medium | Slow (needs generation) |
| Best for | Teaching format | General quality | Sparse feedback | Verifiable tasks |

## Limitations

- Used default hyperparameters with limited tuning
- English preference data for a Chinese-SFT model (language mismatch)
- Small-scale experiments (1.5B model, 5,000 training samples)
- Qualitative evaluation only, no automated benchmarks (MT-Bench, AlpacaEval)
- GRPO tested only on simple math, not general dialogue

## Future work

- Systematic hyperparameter search across all algorithms
- Chinese preference dataset for DPO/KTO (fixes the language mismatch)
- DeepSeek-R1 style `<think>` token with GRPO for chain-of-thought reasoning
- Automated evaluation using LLM-as-a-judge
- Larger model experiments (7B+)
- IPO (Identity Preference Optimization) comparison

## References

- [DPO: Direct Preference Optimization](https://arxiv.org/abs/2305.18290)
- [KTO: Model Alignment as Prospect Theoretic Optimization](https://arxiv.org/abs/2402.01306)
- [DeepSeekMath: GRPO](https://arxiv.org/abs/2402.03300)
- [QLoRA / LoRA](https://arxiv.org/abs/2106.09685)
- [Qwen2.5 Technical Report](https://arxiv.org/abs/2412.15115)

## License

MIT — see [LICENSE](LICENSE).
