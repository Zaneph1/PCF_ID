import csv
answer = []
title = []
data = []
# Load test dataset
solution_text = []
with open('/content/datat.csv', mode='r', encoding='GB2312') as file:
    # Create a csv.reader object
    csv_reader = csv.reader(file)


    i = 0
    for row in csv_reader:

        if i == 0:
            title = row
            i = i + 1
            continue
        data.append(str(row[0])+"是这些类型normal，neptune，ipsweep，satan，smurf，portsweep，nmap，back，teardrop，warezclient的哪种情况(中文回答)")
        solution_text.append(str(row[-1]))
        # if i ==250:
            # break?
        # print(row)
        i = i+1
question_text = []
print(len(data))
#print(data[0])


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer

#Local model path
#You can download the corresponding base model, such as Qwen2.5-0.5B, from Hugging Face to your local machine.
# model_path = "E:\\work\\AI\\GPT\\llama_model"
model_path = "/content/try/s1/models2"
# model_path = "E:\\work\\AI\\GPT\\llama_model_7b_8bit"
print(torch.cuda.is_available())

if torch.cuda.is_available():
    print(torch.cuda.device_count())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
else:
    print('没有GPU')

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
if model_path.endswith("4bit"):
    model = AutoModelForCausalLM.from_pretrained(
            model_path,
            load_in_4bit=True,
            torch_dtype=torch.float16,
            device_map='auto'
        )
elif model_path.endswith("8bit"):
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            load_in_8bit=True,
            torch_dtype=torch.float16,
            device_map='auto'
        )
else:
    model = AutoModelForCausalLM.from_pretrained(model_path).half().cuda()
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
answer = []
instruction = """{}"""
for i in data :
    prompt = instruction.format(str(i))
    generate_ids = model.generate(tokenizer(prompt, return_tensors='pt').input_ids.cuda(), max_new_tokens=300, streamer=streamer)
    generated_text = tokenizer.decode(generate_ids[0], skip_special_tokens=True)
    # print(len(generated_text))
    answer.append(generated_text)
    print(len(answer))

# Save test results
import pandas as pd


data = answer


df = pd.DataFrame(data, columns=[ 'Answer'])


df.to_excel('output.xlsx', index=False)