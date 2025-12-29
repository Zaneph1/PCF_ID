# PCF_ID: A Novel Prompt Cast Framework for Intrusion Detection
---

## 📌 Framework

![PCF-ID Framework](readme_img/frame.png)

## 📂 Project Structure

```
PCF_ID/
├── readme_img/               # Directory for README-related images
│   └── frame.png             # Framework diagram (illustrates the PCF-ID pipeline)
│
├── testdatasets/             # Sample datasets for testing
│   └── testdatasets.csv          # Example intrusion detection dataset (structured tabular format)
│
├── PCF_ID_test.py            # Main evaluation script: test of the PCF-ID framework
│
├── original_model_test.py    # Baseline script: evaluates raw LLM performance without PCF-ID preprocessing
│
└── readme.md                

```

## 🤗 Model Availability
We provide the PCF_ID-0.5B model on Hugging Face for easy access and inference:

🔗 https://huggingface.co/PHZane/PCF_ID-0.5B
