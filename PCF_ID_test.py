# -*- coding: utf-8 -*-

import csv
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import pandas as pd

# ======================
# 1. Load data
# ======================
data = []
solution_text = []

csv_path = "/content/datat.csv"

with open(csv_path, mode='r', encoding='utf-8') as file:
    csv_reader = csv.reader(file)
    next(csv_reader)
    for row in csv_reader:
        data.append(row[0])
        solution_text.append(str(row[-1]))

print(f"✅ Loaded {len(data)} samples")

# ======================
# 2. Load model
# ======================
#model_path = "/home/ckpts/s1-20251016_162947/"
model_path = "/home/ckpts/s1-20251022_145641/"
print("CUDA available:", torch.cuda.is_available())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

# Load model
model = AutoModelForCausalLM.from_pretrained(model_path).half().to(device)

# Prevent warnings or infinite generation
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Optional: print generation in real time
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

# ======================
# 3. Prompt template
# ======================
instruction_template = "{}"

# ======================
# 4. Inference and print
# ======================
answers = []
for idx, i in enumerate(data):
    prompt = instruction_template.format(i)
    inputs = tokenizer(prompt, return_tensors='pt').to(device)

    with torch.no_grad():
        generate_ids = model.generate(
            **inputs,
            max_new_tokens=60,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False,  # deterministic output
            temperature=0.7,
        )

    generated_text = tokenizer.decode(generate_ids[0], skip_special_tokens=True)
    # Extract assistant response
    answer_text = generated_text.split("Assistant answer:")[-1].strip()

    print(f"Input: {i}")
    print(f"Model output: {answer_text}")
    print("-" * 60)

    answers.append({
        "question": i,
        "true_answer": solution_text[idx] if idx < len(solution_text) else "",
        "model_answer": answer_text
    })

# ======================
# 5. Export results
# ======================
df = pd.DataFrame(answers)
output_file = "output_PCF_ID.xlsx"
df.to_excel(output_file, index=False)
print(f"✅ Inference complete. Results saved to: {output_file}")
