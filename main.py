
#%% (I) Prep Data
# Using toxigen here
# https://huggingface.co/datasets/skg/toxigen-data/viewer/train/test
# from datasets import load_dataset
# TG_data = load_dataset("skg/toxigen-data", name="train", use_auth_token=True) # 250k training examples
# TG_annotations = load_dataset("skg/toxigen-data", name="annotated", use_auth_token=True) # Human study
import random
hate_or_neutral = "hate"
for group in ["black"]:#, "middle_east", "native_american"]:
    with open(f"../TOXIGEN/prompts/{hate_or_neutral}_{group}_1k.txt", "r") as f:
        prompts = f.read().splitlines()
prompts = random.sample(prompts, k=2)

#%% (I) Establish Baseline
# 1. load RoBERTa_Toxigen, which scores the toxicity of text
# https://github.com/microsoft/TOXIGEN/tree/main
# ? RoBERTa_Toxigen vs HateBERT
from transformers import AutoModelForSequenceClassification
toxigen_hatebert = AutoModelForSequenceClassification.from_pretrained("tomh/toxigen_roberta")

# 2. load the base model
from transformers import AutoTokenizer
import transformers
import torch
model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
pipeline = transformers.pipeline(
    "text-generation",
    model=model_name,
    torch_dtype=torch.float16,
    device_map="auto",
)

# 4. feed the dataset, collect output
sequences = pipeline(
    prompts,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=512,
)
responses = [seq[0]['generated_text'][len(prompts[i]):] for i,seq in enumerate(sequences)]

# 5. score the output and collect stats
toxigen_roberta = transformers.pipeline("text-classification", model="tomh/toxigen_roberta")
results = toxigen_roberta(responses)
total = len(results)
toxic_count = sum(1 for result in results if result['label'] == 'LABEL_1')

percentage_toxic = (toxic_count / total) * 100
print(f"Out of {total} responses, {percentage_toxic}% are considered toxic by RoBERTa_Toxigen")
i = random.choice(list(range(len(prompts))))
print(f"For example, \nInput:{prompts[i]}\n\nOuput:{responses[i]}")


#%% (II) fine tune the model
# ? alpaca vs supervised fine tuning  (timdettmers/openassistant-guanaco)
# https://github.com/tloen/alpaca-lora/tree/main
# https://huggingface.co/docs/trl/v0.4.7/en/sft_trainer
# https://huggingface.co/datasets/timdettmers/openassistant-guanaco/viewer/default/train
# 1. finetune with alpaca-lora or supervised fine tuning (SFT)


# TODO save the best model


#%% (III) Benchmark the finetuned model
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftConfig, PeftModel
model_name = "meta-llama/Llama-2-7b-hf"
# 1. load the LoRA weights
config = PeftConfig.from_pretrained('./lora-alpaca')
model = AutoModelForCausalLM.from_pretrained(model_name, return_dict=True, load_in_8bit=True, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token=tokenizer.eos_token
# Load the Lora model
model = PeftModel.from_pretrained(model, './lora-alpaca')

# 1. feed the dataset, collect output
inputs = tokenizer(prompts, return_tensors="pt", padding=True)
tokens = model.generate(
    **inputs,
    max_new_tokens=100,
    temperature=1,
    eos_token_id=tokenizer.eos_token_id,
    early_stopping=True,
)
decoded_text = tokenizer.decode(tokens[0], skip_special_tokens=True)
print(decoded_text)
# 2. score the output and collect stats


#%% (IV) Visualization
# ? maybe in the future

# %%

# %%
