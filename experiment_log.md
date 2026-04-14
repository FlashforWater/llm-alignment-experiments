# Alignment Project Experiment Log

## Exp-01: SFT baseline
- data: alpaca-gpt4-zh, 48818 samples
- lr: 2e-4, epochs: 1, batch: 4, accumulation: 4
- lora: r=64, alpha=128, dropout=0.05
- max_seq_length: 512
- final loss: 1.72, time: 23min29s
- total steps: 3051

- eval: output quality is good after adding repetition_penalty=1.3
- issue: without penalty, output shows repetition and multilingual garbled text
- conclusion: SFT successful, model learned the dialogue format — ready for DPO

## Exp-02: DPO (beta=0.1)
- base: merged SFT checkpoint
- data: 5000 English preference pairs from UltraFeedback
- params: lr=5e-5, beta=0.1, lora r=32/alpha=64
- result: improved answer structure and professionalism vs SFT
- problem: verbosity — model learned longer answers get higher preference
- conclusion: DPO works, but introduces length bias

## Exp-03: DPO (beta=0.3)
- base: same as Exp-02
- data: same 5000 samples
- change: only beta 0.1 -> 0.3
- result: did NOT reduce verbosity as expected
- problem: "1+1" question produced hallucination (Java parseInt nonsense)
- conclusion: beta=0.3 is worse than 0.1 for our setup. Higher beta does not simply mean better control.

## Exp-04: DPO filtered (score gap >= 2.0)
- base: same SFT checkpoint
- data: 5000 samples filtered from 16594 high-gap pairs (out of 60917 total)
- params: same as Exp-02 (beta=0.1)
- result: better answer structure, no hallucination
- problem: verbosity persists — this is a data-level issue, not fixable by filtering alone
- insight: data quality matters, but the bias toward longer answers exists across all quality levels

## Exp-05: KTO
- base: same SFT checkpoint
- data: 9924 samples (4962 positive + 4962 negative, converted from DPO pairs)
- params: lr=5e-5, epochs=1, batch=2, accumulation=8, lora_r=32
- result: roughly matches DPO quality, cleaner on simple questions
- observation: verbosity persists, English preference data biases both models toward English responses
- conclusion: KTO matches DPO despite using weaker signal — confirms the paper's findings

## Exp-06: GRPO (math)
- base: SFT checkpoint
- data: 2000 math problems (addition + multiplication), self-generated
- reward: correct+concise=2.0, correct+verbose=1.0, wrong=-1.0
- params: lr=5e-6, num_generations=2, max_completion_length=100
- result: all 5 test problems correct, much more concise than SFT/DPO/KTO
- key insight: GRPO + explicit conciseness reward successfully solved the verbosity problem
- conclusion: when you can define a clear reward function, GRPO is very effective

## Summary
- Exp-01 (SFT): learned dialogue format, loss 2.5→1.72
- Exp-02 (DPO beta=0.1): improved quality, but verbose
- Exp-03 (DPO beta=0.3): worse — hallucination on simple questions
- Exp-04 (DPO filtered): best structure among DPO variants, still verbose
- Exp-05 (KTO): matches DPO with only binary labels
- Exp-06 (GRPO): correct and concise, solved verbosity via explicit reward
